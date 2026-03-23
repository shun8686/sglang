export ASCEND_MF_STORE_URL="tcp://127.0.0.1:8000"


python -m sglang.launch_server \
    --model-path /home/weights/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --disaggregation-mode decode \
    --host 127.0.0.1 \
    --port 8081 \
    --trust-remote-code \
    --tp-size 2 \
    --enable-dp-attention --dp-size 2 \
    --base-gpu-id 6 \
    --mem-fraction-static 0.9 \
    --attention-backend ascend \
    --device npu \
    --log-level debug \
    --disaggregation-transfer-backend ascend \
    --num-reserved-decode-tokens 1024 \
    --attention-backend ascend \
