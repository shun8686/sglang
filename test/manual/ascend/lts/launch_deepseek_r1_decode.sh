#!/bin/bash

pkill -9 sglang
pkill -9 python

#export PYTHONPATH=/data/d00662834/lts-test/randgun/sglang/python:$PYTHONPATH
#export PYTHONPATH=/data/d00662834/lts-test/release1230/sglang/python:$PYTHONPATH

cann_version=$(cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info | grep "^version=")
echo "CANN: ${cann_version}"
if [[ ${cann_version} == version=8.3.* ]];then
    echo "Set env for CANN 8.3"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
    source /usr/local/Ascend/8.5.0/bisheng_toolkit/set_env.sh
else
    echo "Set env for CANN 8.5"
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# NIC Name
NIC_NAME="enp194s0f0"
#NIC_NAME="enp23s0f3"

# node ip
export node_ip="141.61.29.201"

# pd传输, IP设置为p节点首节点
export ASCEND_MF_STORE_URL="tcp://141.61.39.231:24667"
# export ASCEND_MF_STORE_URL="tcp://192.168.0.184:24667"

MODEL_PATH="/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"

mkdir -p log
DECODE_LOG_FILE="./log/launch_decode_$(date +'%Y-%m-%d-%H:%M').log"

# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
# sysctl -w kernel.sched_migration_cost_ns=50000

# 绑核
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# enable mlapo
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1
# decode环境变量
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export HCCL_BUFFSIZE=720
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export TASK_QUEUE_ENABLE=0
export HCCL_SOCKET_IFNAME=$NIC_NAME
export GLOO_SOCKET_IFNAME=$NIC_NAME

# D节点
# --context-length 8192 \ 长序列场景，注释该参数
nohup \
python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
--host ${node_ip} --port 8001 --trust-remote-code \
--nnodes 1 \
--node-rank 0 \
--tp-size 16 \
--dp-size 16 \
--mem-fraction-static 0.8 \
--max-running-requests 384 \
--attention-backend ascend \
--device npu \
--quantization modelslim \
--moe-a2a-backend deepep \
--enable-dp-attention \
--deepep-mode low_latency \
--enable-dp-lm-head \
--cuda-graph-bs 8 10 12 14 16 18 20 22 24 \
--disaggregation-transfer-backend ascend \
--watchdog-timeout 9000 \
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--prefill-round-robin-balance \
--disable-shared-experts-fusion \
--dtype bfloat16 \
--tokenizer-worker-num 4 \
--dist-init-addr ${node_ip}:5000 \
> $DECODE_LOG_FILE 2>&1 &
