pkill -9 sglang
pkill -9 python

export cann_path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/driver/bin/setenv.bash
source ${cann_path}/../set_env.sh
source ${cann_path}/../../nnal/atb/set_env.sh
source ${cann_path}/opp/vendors/customize/bin/set_env.bash
export ASCEND_HOME_PATH=${cann_path}


echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

export PYTHORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export HCCL_SOCKET_TFNAME=enp2133
export GLOO_SOCKET_IFNAME=enp2133

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_ALGO="level0:NA;levrl1:ring"
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export HCCL_BUFFSIZE=720
export SGLANG_DP_ROUND_ROBIN=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export TASK_QUEUE_ENABLE=0



# D节点
python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
--host 192.168.0.102 --port 8001 --trust-remote-code \
--nnodes 1 \
--tp-size 16 \
--dp-size 16 \
--mem-fraction-static 0.8 \
--max-running-requests 384 \
--quantization modelslim \
--moe-a2a-backend deepep \
--enable-dp-attention \
--deepep-mode low_latency \
--enable-dp-lm-head \
--cuda-graph-bs 8 10 12 14 16 18 20 22 24 \
--watchdog-timeout 9000 \
--context-length 8192 \
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--prefill-round-robin-balance \
--disable-shared-experts-fusion \
--dtype bfloat16 \
--tokenizer-worker-num 4 \
