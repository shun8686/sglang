echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
export ASCEND_LAUNCH_BLOCKING=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export CPU_AFFINITY_CONF=1

#export PYTHONPATH=/home/h00848570/code/sglang-main/python:$PYTHONPATH

#export PYTORCH_NO_NPU_MEMORY_CACHING=1
#export TASK_QUEUE_ENABLE=0

export ENABLE_PROFILING=0
export PROFILING_BS=162
export PROFILING_STAGE="decode"
export PROFILING_step=10

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8w8

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"


export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

#export TORCH_LOGS="graph_breaks,graph,recompiles"



python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7788 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 162 \
    --disable-radix-cache \
    --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path /home/weights/Qwen/Qwen3-a3B_eagle3 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size -1 --max-prefill-tokens 35000 \
    --tp-size 2 --mem-fraction-static 0.87 --cuda-graph-bs 1 5 15 40 70 100 120 130 140 146 150 154 156 158 160 162 --dtype bfloat16 --base-gpu-id 14

#python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7788 --max-concurrency 162 --random-input-len 3500 --random-output-len 1500 --num-prompts 624 --random-range-ratio 1 --dataset-path /data/l30081563/GSM8K-in3500-bs3000_qwen3-30b.jsonl

