export ASCEND_MF_STORE_URL="tcp://127.0.0.1:8000"


python -m sglang.launch_server \
    --model-path /home/weights/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --disaggregation-mode prefill \
    --host 127.0.0.1 \
    --port 8080 \
    --trust-remote-code \
    --base-gpu-id 4 \
    --tp-size 2 \
    --enable-dp-attention --dp-size 2 \
    --load-balance-method auto \
    --mem-fraction-static 0.9 \
    --attention-backend ascend \
    --device npu \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --log-level debug \
    --log-level-http debug \
    --disaggregation-bootstrap-port 8998 \
    --dtype bfloat16 \
    --dist-init-addr 127.0.0.1:10001
