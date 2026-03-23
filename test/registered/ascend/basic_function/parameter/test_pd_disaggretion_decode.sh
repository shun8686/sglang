export ASCEND_MF_STORE_URL="tcp://127.0.0.1:8000"


python -m sglang.launch_server \
    --model-path /home/weights/Qwen3-VL-30B-A3B-Instruct \
    --disaggregation-mode decode \
    --host 141.61.29.204 \
    --port 8081 \
    --trust-remote-code \
    --tp-size 2 \
    --base-gpu-id 12 \
    --mem-fraction-static 0.9 \
    --attention-backend ascend \
    --device npu \
    --log-level debug \
    --disaggregation-transfer-backend ascend \
    --num-reserved-decode-tokens 1024 \
    --attention-backend ascend \
