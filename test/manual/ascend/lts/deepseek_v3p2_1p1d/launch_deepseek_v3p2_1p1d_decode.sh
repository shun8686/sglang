#!/bin/bash

pkill -9 sglang
pkill -9 python

#export PYTHONPATH=/data/d00662834/lts-test/randgun/sglang/python:$PYTHONPATH
#export PYTHONPATH=/data/d00662834/lts-test/release1230/sglang/python:$PYTHONPATH

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

MODEL_PATH="/root/.cache/modelscope/hub/models/DeepSeek-V3.2-W8A8"

export SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_NPU_USE_MLAPO=1
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export TASK_QUEUE_ENABLE=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export HCCL_BUFFSIZE=400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8

PIPs=('your prefill ip1' 'your prefill ip2')

DIPs=('your decode ip1' 'your decode ip2')
IFNAMES=('xxx' 'xxx')

# get IP in current node
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
echo "LOCAL_HOST = " ${LOCAL_HOST}
# get node index
for i in "${!DIPs[@]}";
do
  echo "LOCAL_HOST=${LOCAL_HOST}, DIPs[${i}]=${DIPs[$i]}"
  if [ "$LOCAL_HOST" == "${DIPs[$i]}" ]; then
      echo "Node Rank : ${i}"
      VC_TASK_INDEX=$i
      break
  fi
done

export HCCL_SOCKET_IFNAME=${IFNAMES[$VC_TASK_INDEX]}
export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
export ASCEND_MF_STORE_URL=tcp://${PIPs[0]}:24667

nnodes=${#DIPs[@]}
tp_size=`expr 16 \* ${nnodes}`

mkdir -p log
DECODE_LOG_FILE="./log/launch_decode_$(date +'%Y-%m-%d-%H:%M').log"
nohup python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp $tp_size \
--dp 8 \
--ep $tp_size \
--moe-dense-tp-size 1 \
--enable-dp-attention \
--enable-dp-lm-head \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host ${DIPs[$VC_TASK_INDEX]} --port 8001 \
--mem-fraction-static 0.79 \
--disable-radix-cache \
--chunked-prefill-size -1 --max-prefill-tokens 68000 \
--max-running-requests 32 \
--cuda-graph-max-bs 4 \
--moe-a2a-backend deepep \
--deepep-mode low_latency \
--quantization modelslim \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--disaggregation-transfer-backend ascend \
--disaggregation-mode decode \
--prefill-round-robin-balance \
--load-balance-method round_robin \
--nnodes $nnodes --node-rank $VC_TASK_INDEX \
--dist-init-addr ${DIPs[0]}:10000 --load-balance-method decode_round_robin \
> $DECODE_LOG_FILE 2>&1 &
