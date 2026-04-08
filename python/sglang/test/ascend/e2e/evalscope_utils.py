import argparse
import json
import logging

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
    api_url,
    datasets,
    dataset_args,
    eval_batch_size,
    eval_type="openai_api",
    api_key="EMPTY",
    generation_config=None,
    limit=None,
    work_dir=None,
):
    import ssl
    import subprocess

    ssl._create_default_https_context = ssl._create_unverified_context

    cmd_args = [
        "evalscope/bin/evalscope",
        "eval",
        "--model",
        model,
        "--eval-type",
        eval_type,
        "--dataset-args",
        json.dumps(dataset_args),
        "--api-url",
        api_url,
        "--api-key",
        api_key,
        "--eval-batch-size",
        str(eval_batch_size),
    ]

    datasets_args = ["--datasets"]
    datasets_args.extend(datasets)
    cmd_args.extend(datasets_args)

    if generation_config is not None:
        cmd_args.extend(["--generation-config", json.dumps(generation_config)])
    else:
        cmd_args.extend(["--generation-config", json.dumps(GENERATION_CONFIG_DEFAULT)])

    if limit is not None:
        cmd_args.extend(["--limit", str(limit)])

    if work_dir is not None:
        cmd_args.extend(["--work-dir", work_dir])
    else:
        cmd_args.extend(["--work-dir", WORK_DIR_DEFAULT])

    logger.info(f"Executing command: {cmd_args}")

    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    try:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            logger.info(line)
            output_lines.append(line)

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with return code: {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd_args)

        logger.info("Command executed successfully")
        return "\n".join(output_lines)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("Process terminated")
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing it...")
            process.kill()
            logger.info("Process killed")
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        process.terminate()
        process.wait(timeout=5)
        raise


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
