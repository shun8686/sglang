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

export HCCL_BUFFSIZE=800
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_NPU_FUSED_MOE_MODE=2
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=204800
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

MODEL_PATH=/home/weights/MiniMax-M2.5-w8a8-QuaRot
EAGLE_MODEL_PATH=/home/weights/MiniMax-M2.5-eagel-model-0318
export PYTHONPATH=${EAGLE_MODEL_PATH}:$PYTHONPATH
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3

sglang serve \
   --model-path $MODEL_PATH \
   --host 127.0.0.1 \
   --port 32000 \
   --tp-size 16 \
   --enable-dp-attention \
   --dp-size 16 \
   --ep-size 16 \
   --mem-fraction-static 0.75 \
   --max-running-requests 512 \
   --disable-radix-cache \
   --prefill-delayer-max-delay-passes \
   --enable-prefill-delayer \
   --chunked-prefill-size -1 --max-prefill-token 4096 \
   --cuda-graph-bs 1 7 8 9 10 16 \
   --moe-a2a-backend ascend_fuseep --deepep-mode auto --quantization modelslim \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path $EAGLE_MODEL_PATH \
   --speculative-num-steps 3 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 4 \
   --speculative-draft-model-quantization unquant \
   --dtype bfloat16
