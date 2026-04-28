# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

#cd /home/lws/sglang_27b
#cd /home/lws/ascend_sgl
#export PYTHONPATH=${PWD}/python:$PYTHONPATH
# cd /home/hexq/rjw/sglang
export STREAMS_PER_DEVICE=32
#export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
#export HCCL_BUFFSIZE=3000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
#export SGLANG_NPU_PROFILING=1
#export SGLANG_NPU_PROFILING_STAGE="prefill"
#export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
#export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
#export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24669"
#export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export ASCEND_RT_VISIBLE_DEVICES=8,9
#export ASCEND_LAUNCH_BLOCKING=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
#export SGLANG_NPU_PROFILING=1
#export SGLANG_NPU_PROFILING_BS=3
python3 -m sglang.launch_server \
        --model-path /home/weights/Qwen3.5-27B-W8A8 \
        --attention-backend ascend \
        --device npu \
        --tp-size 2 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 65000 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 --max-running-requests 32 --max-mamba-cache-size 32 \
        --mem-fraction-static 0.85 \
        --port 8001 \
        --cuda-graph-bs 2 3 4 5 6 \
        --enable-multimodal \
        --quantization modelslim \
        --mm-attention-backend ascend_attn \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --max-total-tokens 310000 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4


