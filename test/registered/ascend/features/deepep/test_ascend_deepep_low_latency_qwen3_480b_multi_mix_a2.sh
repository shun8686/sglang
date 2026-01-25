#pkill -9 python | pkill -9 sglang
#pkill -9 sglang

MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot
NIC_NAME=enp189s0f0
NODE_IP=('61.47.16.106' '61.47.16.107')
SERVER_PORT=6688
HEALTH_CHECK_URL="http://127.0.0.1:${SERVER_PORT}/health"
TEST_CASE_FILE=test_ascend_deepep_qwen3_480b_a2.py

set_proxy() {
    export http_proxy=61.251.170.143:30066
    export https_proxy=$http_proxy
    export no_proxy=127.0.0.1,localhost,local,.local
}

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

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

# Set Envs
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export HCCL_BUFFSIZE=2100
export HCCL_SOCKET_IFNAME=$NIC_NAME
export GLOO_SOCKET_IFNAME=$NIC_NAME
export HCCL_OP_EXPANSION_MODE=AIV

export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

wait_server_ready() {
    log "Begin to check server health: $HEALTH_CHECK_URL"
    CHECK_INTERVAL=30
    MAX_RETRY=50
    RETRY_COUNT=0
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_CHECK_URL")    
        if [ "$HTTP_CODE" -eq 200 ]; then
            log "Response code is 200. The server is ready."
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ "$RETRY_COUNT" -ge "$MAX_RETRY" ]; then
            log "Error: Reached($MAX_RETRY), and the last response code: $HTTP_CODE"
            exit 1
        fi
        log "Current response code: $HTTP_CODE, is not 200, retrying after ${CHECK_INTERVAL} second. $RETRY_COUNT/$MAX_RETRY "
        sleep $CHECK_INTERVAL
    done
}

launch_server() {
    log "Begin to launch server."
    NODE_RANK=$1
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --host 127.0.0.1 --port ${SERVER_PORT} --trust-remote-code \
        --nnodes 2 --node-rank $NODE_RANK \
        --dist-init-addr ${NODE_IP[0]}:5000 \
        --attention-backend ascend --device npu --quantization modelslim \
        --max-running-requests 96 \
        --context-length 8192 \
        --dtype bfloat16 \
        --chunked-prefill-size 1024 \
        --max-prefill-tokens 458880 \
        --disable-radix-cache \
        --moe-a2a-backend deepep --deepep-mode low_latency \
        --tp-size 16 --dp-size 4 \
        --enable-dp-attention  \
        --enable-dp-lm-head \
        --mem-fraction-static 0.7 \
        --cuda-graph-bs 16 20 24 \
        &
}

run_test_case() {
    set_proxy
    test_case_file=$1
    if ! [ -e "$test_case_file" ];then
        echo "The test case file is not exist: $test_case_file"
        exit 1
    fi
    echo "Begin to run test case: $test_case_file"
    python3 -u $test_case_file
}

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
log "The current node IP: ${LOCAL_HOST1} ${LOCAL_HOST2}"
for i in "${!NODE_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${NODE_IP[$i]}" || "$LOCAL_HOST2" == "${NODE_IP[$i]}" ]];then
        echo "Launch server on node ${NODE_IP[$i]}"

        launch_server $i
        wait_server_ready

        if [ $i -eq 0 ];then
            echo "Running tests on master node..."
            run_test_case $TEST_CASE_FILE
        fi
    fi
done


