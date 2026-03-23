python -m sglang_router.launch_router \
    --pd-disaggregation
    --prefill http://127.0.0.1:8080 8998 \
    --decode http://127.0.0.1:8081 \
    --host 127.0.0.1 \
    --port 6699 \
    --policy cache_aware
