echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#export ASCEND_MF_STORE_URL="tcp://192.168.25.215:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

P_IP=('192.168.25.215' '192.168.25.217')

D_IP=('192.168.25.216' '192.168.25.218')

export ASCEND_USE_FIA=1

MODEL_PATH=/home/weights/Qwen3.5-397B-A17B-w4a8-mtp/

#export SGLANG_NPU_USE_MLAPO=1
#export SGLANG_USE_FIA_NZ=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
		export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24669"

        #export HCCL_BUFFSIZE=3500
		export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
		export HCCL_BUFFSIZE=3000
		export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
		export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

		#export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        #export TASK_QUEUE_ENABLE=2

        #export HCCL_SOCKET_IFNAME=enp196s0f0
        #export GLOO_SOCKET_IFNAME=enp196s0f0

		export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill \
		--host ${P_IP[$i]} --port 32125 --disaggregation-bootstrap-port $((8998+$i)) \
		--trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.62 \
		--attention-backend ascend --device npu \
		--enable-multimodal --quantization modelslim \
        --disaggregation-transfer-backend ascend \
		--max-running-requests 24 --chunked-prefill-size -1 --max-prefill-tokens 131072 \
		--moe-a2a-backend deepep --deepep-mode normal \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
		--speculative-draft-model-quantization unquant \
		--dtype bfloat16 --mamba-ssm-dtype bfloat16 --disable-cuda-graph --disable-overlap-schedule \
		--enable-dp-lm-head --dp-size 2 --enable-dp-attention \
		--tokenizer-worker-num 4 --mamba-scheduler-strategy extra_buffer --mm-enable-dp-encoder \
		--max-mamba-cache-size 120

        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
		export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24669"

        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_BUFFSIZE=2400
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=144
        #export TASK_QUEUE_ENABLE=1
        #export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0

		python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
		--host ${D_IP[$i]} --port 33125 --trust-remote-code \
		--dist-init-addr ${D_IP[0]}:5000 --nnodes 2 --node-rank $i \
		--tp-size 32 --ep-size 32 \
        --mem-fraction-static 0.7 --max-running-requests 144 \
		--attention-backend ascend --device npu \
		--enable-multimodal --quantization modelslim \
        --moe-a2a-backend deepep --enable-dp-attention --deepep-mode low_latency \
		--enable-dp-lm-head --dp-size 4 --enable-dp-attention \
        --cuda-graph-bs 8 16 24 32 36 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 \
        --speculative-algorithm NEXTN --speculative-draft-model-quantization unquant \
		--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --tokenizer-worker-num 4 --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
        --load-balance-method round_robin --disable-radix-cache
        NODE_RANK=$i
        break
    fi
done


pkill -9 sglang

