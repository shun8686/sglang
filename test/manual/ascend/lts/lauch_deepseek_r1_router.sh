mkdir -p lts_test_log
LOG_FILE="./lts_test_log/launch_router_$(date +'%Y-%m-%d-%H:%M').log"

nohup \
python -u -m sglang_router.launch_router \
    --pd-disaggregation \
    --host 127.0.0.1 \
    --port 6688 \
    --prefill http://141.61.39.231:8000 8995\
    --decode http://141.61.29.201:8001 \
> $LOG_FILE 2>&1 &
