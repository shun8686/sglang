# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
# cd /home/rjw/sglang-v4
cd /home/rjw/sglang-qwen3-next
export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# export PATH="/usr/local/python3.11.13/bin:$PATH"
# source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
# source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# 通信buffer
# export SGLANG_DEEPEP_BF16_DISPATCH=0
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16      # 16
export HCCL_BUFFSIZE=2000

# export DEEPEP_NORMAL_LONG_SEQ_ROUND=10
# export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
# export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

#--------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE=AIV
# export HCCL_OP_EXPANSION_MODE=HOST
# export HCCL_DETERMINISTIC=true
# export CLOSE_MATMUL_K_SHIFT=1
export TASK_QUEUE_ENABLE=1


MODEL_PATH=/home/weights/Qwen3-Next-80B-A3B-Instruct-W8A8


# export ASCEND_LAUNCH_BLOCKING=1
# export INF_NAN_MODE_FORCE_DISABLE=1

export SGLANG_WARMUP_TIMEOUT=3600
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
# export FORCE_DRAFT_MODEL_NON_QUANT=1


python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.4 \
    --disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 65535 --context-length 65535 \
    --max-running-requests 64 \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim 2>&1 | tee launch.log &
    
    # --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
    # --dtype bfloat16 \
    # --disable-cuda-graph \
    # --cuda-graph-bs 1 2 4 \
    # --disable-overlap-schedule \
    # --dp-size 2 --enable-dp-attention --enable-dp-lm-head \
    # --skip-server-warmup
    
    
