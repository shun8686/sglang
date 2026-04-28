echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/home/weights/Qwen3.5-397B-A17B-w4a8-mtp/

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export PYTHONPATH=/home/q30063557/code/sglang-qdj/python:$PYTHONPATH

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0
export HCCL_OP_EXPANSION_MODE="AIV"

# zbal
export HCCL_BUFFSIZE=0
unset PYTORCH_NPU_ALLOC_CONF
export SGLANG_ZBAL_LOCAL_MEM_SIZE=57600
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://192.168.25.215:24671"
# zbal if use mix alloc
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
# zbal if support graph
export ZBAL_ENABLE_GRAPH=1

MIX_IP=('192.168.25.215' '192.168.25.216')

for i in "${!MIX_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${MIX_IP[$i]}" || "$LOCAL_HOST2" == "${MIX_IP[$i]}" ]];
    then
        echo "${MIX_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
        export SGLANG_ENABLE_SPEC_V2=1

        python -m sglang.launch_server --model-path ${MODEL_PATH} \
        --host 192.168.25.215 --port 31125 --trust-remote-code \
        --nnodes 2 --node-rank $i --tp-size 32 \
	--dp-size 32 --enable-dp-attention --enable-dp-lm-head \
	--mem-fraction-static 0.7 --max-running-requests 512 \
        --attention-backend ascend --device npu \
	--quantization modelslim --enable-multimodal \
        --moe-a2a-backend deepep --deepep-mode auto --cuda-graph-bs 1 2 4 6 8 10 12 14 16 18 20 22 24 \
        --dist-init-addr ${MIX_IP[0]}:5000 --chunked-prefill-size -1 --max-prefill-tokens 65536 \
        --speculative-algorithm NEXTN --speculative-draft-model-quantization unquant \
        --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
        --max-total-tokens 200000 --tokenizer-worker-num 4 \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
	--disable-radix-cache
        NODE_RANK=$i
        break
    fi
done
