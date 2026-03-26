#!/bin/bash

pkill -9 sglang
pkill -9 python

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

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export HCCL_BUFFSIZE=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=5
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

MODEL_PATH="/root/.cache/modelscope/hub/models/DeepSeek-V3.2-W8A8"

export SGLANG_NPU_USE_MLAPO=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_OP_EXPANSION_MODE=AIV

PIPs=('your prefill ip1' 'your prefill ip2')
IFNAMES=('xxx' 'xxx')

DIPs=('your decode ip1' 'your decode ip2')

# get IP in current node
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
echo "LOCAL_HOST = " ${LOCAL_HOST}
# get node index
for i in "${!PIPs[@]}";
do
  echo "LOCAL_HOST=${LOCAL_HOST}, PIPs[${i}]=${PIPs[$i]}"
  if [ "$LOCAL_HOST" == "${PIPs[$i]}" ]; then
      echo "Node Rank : ${i}"
      VC_TASK_INDEX=$i
      break
  fi
done

export HCCL_SOCKET_IFNAME=${IFNAMES[$VC_TASK_INDEX]}
export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
echo "HCCL_SOCKET_IFNAME : ${HCCL_SOCKET_IFNAME}"
nnodes=${#PIPs[@]}
tp_size=`expr 16 \* ${nnodes}`
export ASCEND_MF_STORE_URL=tcp://${PIPs[0]}:24667

mkdir -p log
PREFILL_LOG_FILE="./log/launch_prefill_$(date +'%Y-%m-%d-%H:%M').log"
nohup python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp $tp_size \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host ${PIPs[$VC_TASK_INDEX]} --port 8000 \
--mem-fraction-static 0.73 \
--chunked-prefill-size -1 --max-prefill-tokens 68000 \
--max-running-requests 1 \
--moe-a2a-backend deepep --deepep-mode normal \
--quantization modelslim \
--disaggregation-transfer-backend ascend \
--disaggregation-mode prefill \
--disable-cuda-graph \
--nnodes $nnodes --node-rank $VC_TASK_INDEX \
--disaggregation-bootstrap-port 8995 \
--enable-nsa-prefill-context-parallel  --moe-dense-tp-size 1 \
--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
--dist-init-addr ${PIPs[0]}:10000 \
> $PREFILL_LOG_FILE 2>&1 &

pause

# launch router node
ROUTER_LOG_FILE="./log/launch_router_$(date +'%Y-%m-%d-%H:%M').log"
export SGLANG_DP_ROUND_ROBIN=1
nohup python -u -m sglang_router.launch_router \
    --pd-disaggregation \
    --host 127.0.0.1 \
    --port 6688 \
    --prefill http://${PIPs[0]}:8000 8995\
    --decode http://${DIPs[0]}:8001 \
> $ROUTER_LOG_FILE 2>&1 &
