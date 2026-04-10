#!/bin/bash
set -e

# Check if the correct number of arguments are provided
if [ $# -ne 8 ]; then
    echo -e "\033[31mUsage:\033[0m"
    echo "  $0 <IP> <PORT> <MODEL_NAME> <MODEL_PATH> <DATASET> <MAX_OUT_LEN> <BATCH_SIZE> <NUM_PROMPTS>"
    echo "  Example: $0 127.0.0.1 54321 Qwen2-7B-Instruct /models/qwen gsm8k_gen 1024 32 128"
    exit 1
fi

# Assign command-line arguments to variables
IP=$1
PORT=$2
MODEL=$3
PATH=$4
DATASET_PATH=$5
MAX_OUT_LEN=$6
BATCH_SIZE=$7
NUM_PROMPTS=$8

# Create a unique temporary config file (prevents conflicts in parallel execution)
TMP_CFG=$(touch "/tmp/vllm_api_${MODEL}.py")

# Generate dynamic VLLM API config with user-provided parameters
cat > "$TMP_CFG" << EOF
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="$PATH",
        model="$MODEL",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="$IP",
        host_port=$PORT,
        url="",
        max_out_len=$MAX_OUT_LEN,
        batch_size=$BATCH_SIZE,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True
        )
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
EOF


TMP_DATASET=$(touch "/tmp/mm_custom_gen_${MODEL}.py")
cat > "$TMP_DATASET" << EOF
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMCustomDataset, MMCustomEvaluator


mm_custom_reader_cfg = dict(
    input_columns=['question', 'image', 'video', 'audio'],
    output_column='answer'
)


mm_custom_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "file://{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mm_custom_eval_cfg = dict(
    evaluator=dict(type=MMCustomEvaluator)
)

mm_custom_datasets = [
    dict(
        abbr='mm_custom',
        type=MMCustomDataset,
        path=$DATASET_PATH,
        mm_type="path",
        num_frames=5,
        reader_cfg=mm_custom_reader_cfg,
        infer_cfg=mm_custom_infer_cfg,
        eval_cfg=mm_custom_eval_cfg,
        k=1,
        n=1,
    )
]
EOF


echo -e "\033[32m✅ Created isolated temp config:\033[0m $TMP_CFG"
echo "IP: $IP | Port: $PORT | Model: $MODEL | Path: $PATH"
echo "Batch: $BATCH_SIZE | Max Tokens: $MAX_OUT_LEN"

source test_env_aisbench/bin/activate
ais_bench \
  --models "$TMP_CFG" \
  --datasets "$TMP_DATASET" \
  --mode perf \
  --num-prompts "$NUM_PROMPTS"
