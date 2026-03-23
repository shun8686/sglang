echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0r
sysctl -w kernel.numa_balancing=0ec
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set/env.sh
source /usr/local/Ascend/nnal/atb/set/env.sh
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=2500
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=enp196sOf0
export GL0O_SOCKET_IFNAME=enp196s0f0
export SGLANG_NPU_PROFILING=0
#export SGLANG_NPU_PROFILING_STAGE
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS-4096
export ASCEND_MF_STORE_URL="tcp:/127.0.0.1:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
python3 -m sglang.launch server \
--model-path /home/weights/Qwen3.5-397B-A17B-w8a8 \
--attention-backend ascendr \
--device npu \
--tp-size 16 --nnodes 1 --node-rank 0 \
--chunked-prefill-size -1 --max-prefill-tokens 16384 \
--disable radix-cache \
--trust-remote-code \
--host 0.0.0.0 --max-running-requests 64 \
--mem-fraction-static 0.85 \
--port 8000 \
--cuda-graph-bs 2 4 6 8 10 16 20 24 28 32 \
--quantization modelslim \
--enable-multimodal	--moe-a2a-backend deepep --deepep-mode auto \
--mm-attention-backend ascend attn --moe-a2a-backend deepep --deepep-mode auto \
--dtype bfloat16 --dp-size 2 --enable-dp-attention \
--disaggregation-bootstrap-port 8998 --disaggregation-mode prefill --disaggregation-transfer-backend ascend
