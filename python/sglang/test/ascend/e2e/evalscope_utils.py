import argparse
import json
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
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
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
    api_key=None,
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
    task_config.dataset_args = (
        DATASET_ARGS_DEFAULT if dataset_args is None else dataset_args
    )
    task_config.eval_batch_size = eval_batch_size
    task_config.generation_config = (
        GENERATION_CONFIG_DEFAULT if generation_config is None else generation_config
    )
    if limit is not None:
        task_config.limit = limit
    task_config.work_dir = WORK_DIR_DEFAULT if work_dir is None else work_dir

    result = run_task(task_config)
    logger.info(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evalscope", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        required=False,
        default="openai_api",
        help="Eval type",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        required=True,
        help="API URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        help="API Key",
    )
    parser.add_argument(
        "--datasets",
        type=json.loads,
        required=False,
        default=None,
        help="datasets",
    )
    parser.add_argument(
        "--dataset-args",
        type=json.loads,
        required=False,
        default=None,
        help="dataset args",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        required=False,
        default=None,
        help="eval batch size",
    )
    parser.add_argument(
        "--generation-config",
        type=json.loads,
        required=False,
        default=None,
        help="generation config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=None,
        help="limit",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=False,
        default=None,
        help="work dir",
    )

    args = parser.parse_args()

    run_evalscope_accuracy_test(
        model=args.model,
        eval_type=args.eval_type,
        api_url=args.api_url,
        api_key=args.api_key,
        datasets=args.datasets,
        dataset_args=args.dataset_args,
        eval_batch_size=args.eval_batch_size,
        generation_config=args.generation_config,
        limit=args.limit,
        work_dir=args.work_dir,
    )
