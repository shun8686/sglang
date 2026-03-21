# 单机混布
# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
cd /home/l00890003/codes/sglang-npu-nn-xx
export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# 通信buffer
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=80
export HCCL_BUFFSIZE=1600
export DEEPEP_NORMAL_LONG_SEQ_ROUND=10
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

# mtp quant path
MODEL_PATH=/home/weights/deepseekr1_w4a8_pertoken
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
#export SGLANG_NPU_USE_MULTI_STREAM=1

export SGLANG_USE_FIA_NZ=1
#export ENABLE_MOE_NZ=1
# export SGLANG_NPU_PROFILING=1
# export SGLANG_NPU_PROFILING_BS=20
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 16 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--quantization modelslim \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6699 \
--cuda-graph-bs 4 8 16 20 \
--mem-fraction-static 0.755 \
--max-running-requests 320 \
--disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 1500 \
--moe-a2a-backend deepep --deepep-mode auto \
--enable-dp-attention --dp-size 16 --enable-dp-lm-head \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--dtype bfloat16

exit 1
