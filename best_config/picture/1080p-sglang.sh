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
unset ASCEND_LAUNCH_BLOCKING
# export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_USE_FIA=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCCL_BUFFSIZE=3000
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python3 -m sglang.launch_server \
--model-path  /home/weights/Qwen3.5-397B-A17B-w4a8-mtp \
--attention-backend ascend \
--device npu \
--tp-size 16 \
--chunked-prefill-size -1 --max-prefill-tokens 16384 \
--disable-radix-cache \
--trust-remote-code \
--host 127.0.0.1 --max-running-requests 256 \
--mem-fraction-static 0.85 \
--port 31125 \
--cuda-graph-bs 1 2 4 6 8 12 16 20 24 28 32 40 48 56 64 \
--quantization modelslim \
--enable-multimodal --moe-a2a-backend deepep --deepep-mode auto \
--mm-attention-backend ascend_attn \
--dtype bfloat16 --mamba-ssm-dtype bfloat16 --max-total-tokens 800000 \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--speculative-draft-model-quantization unquant \
--dp-size 4 --enable-dp-attention --enable-dp-lm-head 

