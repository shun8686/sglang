#!/bin/bash

INTERVAL=10
LOGDIRPATH="./log"
LOGFILEPATH="monitor_$(date +"%y%m%d-%H:%M").log"

# function sglangMonitor() {
#     echo "======================sglangMonitor()==========================" >> "$LOGDIRPATH/$LOGFILEPATH"
#     sglangPid=$(ps -ef | grep "python3 -m sglang.launch_server" | grep -v grep | awk '{print $2}' | head -1)
#     if [[ $str =~ ^.*[^[:space:]].*$ ]]; then
#         sglangLsopOpenFile=$(lsof -p $sglangPid | wc -l)
#         sglangRES=$(top -bn1 -p ${sglangPid} | tail -n2 | grep ${sglangPid} | awk '{print $6}')
#         sglangMEM=$(top -bn1 -p ${sglangPid} | tail -n2 | grep ${sglangPid} | awk '{print $10}')
#         sglangCPU=$(top -bn1 -p ${sglangPid} | tail -n2 | grep ${sglangPid} | awk '{print $9}')
#         sglangZoom=$(ps -ef | grep defunc[t] | wc -l)
#         echo "$(date +"%y%m%d-%H:%M:%S") sglangPid:${sglangPid} sglangCPU:${sglangCPU}% sglangRES:${sglangRES} sglangMEM:${sglangMEM}% sglangLsopOpenFile:${sglangLsopOpenFile} sglangZoom:${sglangZoom}" >> "$LOGDIRPATH/$LOGFILEPATH"
#     fi
# }

function nodeMonitor() {
    echo "======================nodeMonitor()==========================" >> "$LOGDIRPATH/$LOGFILEPATH"
    nodeSYCPU=$(top -bn1 | grep Cpu | awk '{print $4}')
    nodeUSCPU=$(top -bn1 | grep Cpu | awk '{print $2}')
    nodeCPU=$(echo ${nodeSYCPU} + ${nodeUSCPU} | bc)
    nodemem_kb=$(vmstat -s | grep "used memory" | awk '{print $1}')
    nodemem=$(awk "BEGIN {print $nodemem_kb/1024/1024}")
    echo "$(date +"%y%m%d-%H:%M:%S") nodeSYCPU:${nodeSYCPU}% nodeUSCPU:${nodeUSCPU}% nodeCPU:${nodeCPU}% nodemem:${nodemem}g" >> "$LOGDIRPATH/$LOGFILEPATH"
}

function npuMonitor() {
    echo "======================npuMonitor()==========================" >> "$LOGDIRPATH/$LOGFILEPATH"
    # 定义列宽度常量
    TIMESTAMP_WIDTH=20
    NPU_ID_WIDTH=9
    CHIP_ID_WIDTH=10
    PHY_ID_WIDTH=10
    AICORE_WIDTH=11
    HBM_INFO_WIDTH=20

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    NPU_INFO=$(npu-smi info 2>/dev/null || echo "")

    OUTPUT=""

    while IFS= read -r line; do
        if [[ "$line" == *"Process id"* ]]; then
            break
        fi
        if [[ "$line" =~ ^\|[[:space:]]*([0-9]+)[[:space:]]+([0-9]+).*\|[[:space:]]*([0-9]+).*\|.*$ ]]; then
            chip_id="${BASH_REMATCH[1]}"
            phy_id="${BASH_REMATCH[2]}"
            aicore="${BASH_REMATCH[3]}"

            if [[ "$line" =~ ([0-9]+[[:space:]]*/[[:space:]]*[0-9]+)[[:space:]]*\|[[:space:]]*$ ]]; then
                hbm_info="${BASH_REMATCH[1]}"
            else
                hbm_info="N/A"
            fi

            npu_id=$((phy_id / 2))

            # 使用printf格式化输出，实现左对齐
            printf -v log_entry "%-${TIMESTAMP_WIDTH}s %-${NPU_ID_WIDTH}s %-${CHIP_ID_WIDTH}s %-${PHY_ID_WIDTH}s %-${AICORE_WIDTH}s %-${HBM_INFO_WIDTH}s" \
                   "${TIMESTAMP}" \
                   "NPU_ID:${npu_id}" \
                   "CHIP_ID:${chip_id}" \
                   "Phy_ID:${phy_id}" \
                   "AICORE:${aicore}%" \
                   "HBM_INFO:${hbm_info}"

            OUTPUT+="${log_entry}"$'\n'
        fi
    done <<< "$NPU_INFO"

    echo -n "$OUTPUT" >> "$LOGDIRPATH/$LOGFILEPATH"
}

[[ ! -d ${LOGDIRPATH} ]] && mkdir -p "${LOGDIRPATH}"
touch "$LOGDIRPATH/$LOGFILEPATH"

while true; do
    # sglangMonitor
    nodeMonitor
    npuMonitor

    sleep $INTERVAL
done
