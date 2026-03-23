echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa balancing=0e
sysctl -w kernel.sched migration cost ns=50000
# bind cou
export SGLANG SET CPU AFFINITY=1
unset https proxy
unset http proxy
unset HTTPS PROXY
unset HTTP PROXY
export ASCEND LAUNCH BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set env.sh
source /usr/local/Ascend/nnal/atb/set env.sh
export SGLANG DISAGGREGATION WAITING TIMEOUT-3600
export STREAMS PER DEVICE=32
#export SGLANG DEEPEP NUM MAX DISPATCH TOKENS PER RANK=128
export HCCL BUFFSIZE=1200
export HCCL OP EXPANSION MODE=AIV
export HCCL_SOCKET_IFNAME=enp196sOf0
export GL0O SOCKET IFNAME=enp196sOf0 
export SGLANG NPU PROFILING=0的
#export SGLANG NPU PROFILING STAGE="prefill"
#export DEEPEP NORMAL LONG SEQ ROUND=32
#export DEEPEP NORMAL LONG SEQ PER ROUND TOKENS=3584
export ASCEND MF STORE URL="tcp://61.47.19.76:24669"
export SGLANG DISAGGREGATION BOOTSTRAP TIMEOUT=3600
0098-LN03WIL ONILIVM NOILV93YO9VSIG ONVTOS 1JOdxa
python3 -m sglang.launch server \
--model-path /home/weights/Qwen3.5-397B-A17B-w8a8
--attention-backend ascendr
--device npu \
--tp-size 16 --nnodes 1 --node-rank 0 \
--chunked-prefill-size -1 --max-prefill-tokens 16384 \
--disable radix-cache
--trust-remote-code \
--host 0.0.0.0 --max-running requests 64 \
--mem-fraction-static 0.85 \
--port 8000 \
--cuda-graph-bs 2 4 6 8 10 16 20 24 28 32 \
--quantization modelslim \
--enable-multimodal	--moe-a2a-backend deepep --deepep-mode auto \
--mm-attention-backend ascend attn --moe-a2a-backend deepep --deepep-mode auto \
--dtype bfloat16 --dp-size 2 --enable-dp-attention \
--disaggregation-bootstrap-port 8998 --disaggregation-mode prefill --disaggregation-transfer-backend ascend
