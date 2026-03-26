pip install evalscope

model=$1
api_url=$2
# api_url=http://127.0.0.1:8080/v1

evalscope eval \
--model $model \
--api-url $api_url \
--api-key EMPTY \
--eval-type openai_api \
--datasets ceval \
--dataset-hub Local \
--dataset-args '{"ceval": {"local_path": "/root/.cache/modelscope/hub/datasets/EvalScope/ceval"}}' \
--eval-batch-size 8 \
--ignore-errors \
--generation_config '{"timeout": 60}' \
--limit 2
