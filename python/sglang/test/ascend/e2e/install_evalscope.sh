python -m venv evalscope
source evalscope/bin/activate

# 在线安装
# pip install evalscope

# 离线安装
cp -r /root/.cache/.cache/evalscope /tmp/
pip3 install --no-index --find-links=/tmp/evalscope/ evalscope

evalscope eval \
  --model /root/.cache/modelscope/hub/models/DeepSeek-R1-0528 \
  --api-url http://127.0.0.1:6677/v1 \
  --api-key EMPTY \
  --eval-type openai-api \
  --generation-config '{
    "do_sample": true,
    "max_tokens": 1024,
    "seed": 3407,
    "top_p": 0.8,
    "top_k": 20,
    "temperature": 0.7,
    "n": 1,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "timeout": 3600,
    "stream": true,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}' \
  --datasets gsm8k \
  --eval-batch-size 128 \
  --ignore-errors \
  --limit 200