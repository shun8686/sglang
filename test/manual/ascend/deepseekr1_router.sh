python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://192.168.0.184:8000 8995\
    --decode http://192.168.0.60:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
