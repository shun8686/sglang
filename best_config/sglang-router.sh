

#nohup bash 1024-sglang-prefill.sh > log_p.txt 2>&1 &
#nohup bash 1024-sglang-decode.sh  > log_d.txt 2>&1 &

export PYTHONPATH=/home/q30063557/code/sglang-qdj/python:$PYTHONPATH

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

python -m sglang_router.launch_router \
--pd-disaggregation \
--prefill http://192.168.25.215:32125 8998 \
--prefill http://192.168.25.217:32125 8999 \
--decode http://192.168.25.216:33125 \
--host 192.168.25.215 \
--port 31125 \
--health-check-interval-secs 3600

# --prefill http://192.168.25.217:32125 8999 \
# --mini-lb \
