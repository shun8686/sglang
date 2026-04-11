export PYTHONPATH=/home/x00452483/sglang/python:$PYTHONPATH

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE=1500
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=4096
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

sglang serve \
   --model-path /home/weights/MiniMax-M2.5-w8a8-QuaRot \
   --host 127.0.0.1 \
   --port 32000 \
   --tp-size 16 \
   --enable-dp-attention \
   --dp-size 8 \
   --ep-size 16 \
   --disable-radix-cache \
   --mem-fraction-static 0.6 \
   --prefill-delayer-max-delay-passes 200 \
   --enable-prefill-delayer \
   --max-running-requests 24 \
   --chunked-prefill-size -1 --max-prefill-token 32768 \
   --cuda-graph-bs 1 2 3 4 \
   --moe-a2a-backend deepep --deepep-mode auto --quantization modelslim \
