import subprocess

import logging
from evalscope import TaskConfig, run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

GENERATION_CONFIG_DEFAULT = {
    "do_sample": True,
    "max_tokens": 1024,
    "seed": 3407,
    "top_p": 0.8,
    "top_k": 20,
    "temperature": 0.7,
    "n": 1,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "time_out": 3600,
    "stream": True,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
}
DATASET_DEFAULT = ["gsm8k", "mmlu", "mmlu_pro", "aime24", "math_500", "gpqa_diamond"]
DATASET_ARGS_DEFAULT = {
    "gsm8k": {},
    "mmlu": {},
    "mmlu_pro": {},
    "aime24": {},
    "math_500": {},
    "gpqa_diamond": {},
}
WORK_DIR_DEFAULT = "/root/.cache/tests/output/accuracy/"


def run_evalscope_accuracy_test(
    model,
    eval_type="openai_api",
    api_url=None,
    api_key="EMPTY",
    datasets=None,
    dataset_args=None,
    eval_batch_size=128,
    generation_config=None,
    limit=None,
    work_dir=None,
):
    task_config = TaskConfig(model=model, eval_type=eval_type, api_url=api_url)
    task_config.api_key = "EMPTY" if generation_config is None else api_key
    task_config.datasets = DATASET_DEFAULT if datasets is None else datasets
    task_config.dataset_args = DATASET_ARGS_DEFAULT if dataset_args is None else dataset_args
    task_config.eval_batch_size = eval_batch_size
    task_config.generation_config = GENERATION_CONFIG_DEFAULT if generation_config is None else generation_config
    if limit is not None:
        task_config.limit = limit
    task_config.work_dir = WORK_DIR_DEFAULT if work_dir is None else work_dir

    result = run_task(task_config)
    print(result)

