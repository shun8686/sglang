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
export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=2500
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0
export SGLANG_NPU_PROFILING=0
#export SGLANG_NPU_PROFILING_STAGE
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=4096
export ASCEND_MF_STORE_URL="tcp://61.47.19.75:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python3 -m sglang.launch_server \
        --model-path /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-0.6B \
        --quantization modelslim \
        --attention-backend ascend \
        --device npu \
        --tp-size 4 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 65536 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 0.0.0.0 --max-running-requests 256 \
        --mem-fraction-static 0.85 \
        --port 8000 \
        --cuda-graph-bs 1 2 3 4 8 9 10 11 12 13 14 15 16 \
        --enable-multimodal  \
        --mm-attention-backend  --moe-a2a-backend deepep --deepep-mode auto \
        --max-total-tokens 1200000 \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --disaggregation-mode decode --disaggregation-transfer-backend ascend \
        --speculative-draft-model-quantization unquant --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4





