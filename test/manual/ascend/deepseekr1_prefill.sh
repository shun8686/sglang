pkill -9 sglang
pkill -9 python

# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1
export HCCL_BUFFSIZE=1536
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export TASK_QUEUE_ENABLE=2
export HCCL_SOCKET_TFNAME=enp23s0f3
export GLOO_SOCKET_IFNAME=enp23s0f3

export PYTHONPATH=/usr/local/python3.11.13/lib/python3.11/site-packages/sglang:$PWD/python/:$PYTHONPATH
export MODEL_PATH="/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8"
export logfile="./launch_prefill_$(date +'%Y-%m-%d-%H:%M').log"


# P节点
python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
--host 192.168.0.184 --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
--nnodes 1 \
--node-rank 0 \
--disaggregation-mode prefill \
--tp-size 16 \
--mem-fraction-static 0.8 \
--quantization modelslim \
--max-running-requests 8 \
--disable-radix-cache \
--chunked-prefill-size -1 \
--max-prefill-tokens 28680 \
--moe-a2a-backend deepep \
--deepep-mode normal \
--speculative-algorithm NEXTN \
--speculative-num-steps 1 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 2 \
--dp-size 2 \
--enable-dp-attention \
--disable-shared-experts-fusion \
--dtype bfloat16 \
