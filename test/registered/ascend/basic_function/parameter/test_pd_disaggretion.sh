# cpu高性能
from Tools.scripts.nm2def import export_list
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
# cd /home/l00890003/codes/sglang-npu-nn-xx
# export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# unset ASCEND_LAUNCH_BLOCKING
export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=3000
export HCCL_OP_EXPANSION_MODE=AIV
# 网卡
export HCCL_SOCKET_INFNAME=lo
export GLOO_SOCKET_INFNAME=lo
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_STAGE="prefill"
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export ASCEND_MF_STORE_url="tcp://127.0.0.1:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

MODEL_PATH=/root/.cache/modelscope/hub/model/Qwen/Qwen3.5-27B-W8A8

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--quantization modelslim \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--nnodes 1 --node-rank 0 \
--tp-size 4 \
--mem-fraction-static 0.75 \
--chunked-prefill-size -1 --max-prefill-tokens 100352 \
--disable-radix-cache \
--max-running-requests 128 \
--host 127.0.0.1 --port 10000 \
--cuda-graph-bs 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 \
--enable-multimodal \
--enable-multimodal \
--mm-attention-backend ascend_attn --max-total-tokens 600000 \
--dtype bfloat16 --mamba-ssm-dtype bfloat16 --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
--base-gpu-id 0 \
--watchdog-timeout 9000 \
--enable-dp-attention --dp-size 2 --enable-dp-lm-head \

exit 1
