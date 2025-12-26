export logfile="./launch_router_$(date +'%Y-%m-%d-%H:%M').log"

nohup \
python -u -m sglang_router.launch_router \
    --pd-disaggregation \
    --host 127.0.0.1 \
    --port 6688 \
    --prefill http://141.61.39.231:8000 8995\
    --decode http://141.61.29.201:8001 \
> $logfile 2>&1 & \

# nohup \
# python -u -m sglang_router.launch_router \
#     --pd-disaggregation \
#     --host 127.0.0.1 \
#     --port 6688 \
#     --prefill http://192.168.0.184:8000 8995\
#     --decode http://192.168.0.60:8001 \
# > $logfile 2>&1 & \
