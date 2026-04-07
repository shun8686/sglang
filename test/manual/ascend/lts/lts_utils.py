import logging
import ssl
import subprocess
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    E2E_TOLERANCE,
    OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
    TPOT_THRESHOLD,
    TPOT_TOLERANCE_HIGH,
    TPOT_TOLERANCE_LOW,
    TTFT_TOLERANCE,
    retry,
    run_bench_serving,
)
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import CustomTestCase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

LONG_SEQ_DEFAULT_CONFIGS = {
    "64k+1k": {
        "input_len": 65536,
        "output_len": 1024,
        "max_concurrency": 1,
        "num_prompts": 4,
        "ttft": 100000,
        "tpot": 350,
        "tps": 10,
    },
    "32k+1k": {
        "input_len": 32768,
        "output_len": 1024,
        "max_concurrency": 1,
        "num_prompts": 4,
        "ttft": 70000,
        "tpot": 250,
        "tps": 10,
    },
    "16k+1k": {
        "input_len": 16384,
        "output_len": 1024,
        "max_concurrency": 1,
        "num_prompts": 4,
        "ttft": 40000,
        "tpot": 200,
        "tps": 10,
    },
}


def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.info(f"command error: {e}")
        return None


class TestAscendLtsTestCaseBase(CustomTestCase):
    max_attempts = 5
    host = None
    port = None
    base_url = None
    model = None
    backend = "sglang"
    dataset_name = "random"
    dataset_path = "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
    output_file = None
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = None
    image_resolution = None
    image_count = None
    warmup_requests = None
    seed = None
    ttft = None
    tpot = None
    mean_e2e_latency = None
    output_token_throughput = None
    accuracy = {"gsm8k": 1, "mmlu": 1}

    def _assert_metrics(self, metrics):
        """Assert benchmark metrics against expected values.

        Args:
            metrics (dict): Benchmark metrics dictionary.
        """
        if not metrics:
            self.fail("No metrics obtained from benchmark")

        if self.tpot:
            if self.tpot < TPOT_THRESHOLD:
                self.assertLessEqual(
                    float(metrics["mean_tpot"]),
                    self.tpot + TPOT_TOLERANCE_LOW,
                )
            else:
                self.assertLessEqual(
                    float(metrics["mean_tpot"]),
                    self.tpot * TPOT_TOLERANCE_HIGH,
                )
        if self.output_token_throughput:
            self.assertGreaterEqual(
                float(metrics["total_tps"]),
                self.output_token_throughput * OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
            )
        if self.ttft:
            self.assertLessEqual(
                float(metrics["mean_ttft"]),
                self.ttft * TTFT_TOLERANCE,
            )
        if self.mean_e2e_latency:
            self.assertLessEqual(
                float(metrics["mean_e2e_latency"]),
                self.mean_e2e_latency * E2E_TOLERANCE,
            )

    @retry()
    def run_throughput(self):
        logger.info(f"---------- Start benchserving test ----------")
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        bench_params = {
            "host": host,
            "port": port,
            "model_path": self.model,
            "backend": self.backend,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "request_rate": self.request_rate,
            "max_concurrency": self.max_concurrency,
            "num_prompts": self.num_prompts,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "random_range_ratio": self.random_range_ratio,
            "image_resolution": self.image_resolution,
            "image_count": self.image_count,
            "warmup_requests": self.warmup_requests,
            "seed": self.seed,
            "output_file": self.output_file,
        }
        logger.info(f"Starting benchmark with parameters: {bench_params}")
        metrics = run_bench_serving(**bench_params)
        self._assert_metrics(metrics)
        logger.info(f"---------- Benchserving test finished ----------")

    def run_gsm8k(self):
        logger.info(f"---------- Start gsm8k accuracy test ----------")
        logger.info(f"host:{self.host}, port:{self.port}")
        args = SimpleNamespace(
            host=self.host,
            port=self.port,
            eval_name="gsm8k",
            num_shots=8,
            data_path="/tmp/test.jsonl",
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
        )
        metrics = run_eval(args)
        logger.info(f"{metrics}")
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy["gsm8k"],
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy["gsm8k"]}',
        )
        logger.info(f"---------- Gsm8k accuracy test finished ----------")

    def run_mmlu(self):
        logger.info(f"---------- Start mmlu accuracy test ----------")
        ssl._create_default_https_context = ssl._create_unverified_context
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        logger.info(f"{metrics}")
        self.assertGreater(
            metrics["score"],
            self.accuracy["mmlu"],
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy["mmlu"]}',
        )
        logger.info(f"---------- Mmlu accuracy test finished ----------")

    def run_long_seq_testcase(self, long_seq_configs=None):
        if long_seq_configs is None:
            long_seq_configs = LONG_SEQ_DEFAULT_CONFIGS
        for seq_type, seq_config in long_seq_configs.items():
            logger.info(f"---------- Start long seq test: {seq_type} ----------")
            metrics = run_bench_serving(
                host=self.host,
                port=self.port,
                model_path=self.model,
                input_len=seq_config["input_len"],
                output_len=seq_config["output_len"],
                random_range_ratio=1,
                dataset_name=self.dataset_name,
                dataset_path=self.dataset_path,
                request_rate="inf",
                max_concurrency=seq_config["max_concurrency"],
                num_prompts=seq_config["num_prompts"],
            )
            logger.info(f"metrics: {metrics}")

            if "tpot" in seq_config.keys():
                self.assertLessEqual(
                    float(metrics["mean_tpot"]),
                    seq_config["tpot"],
                )
            if "tps" in seq_config.keys():
                self.assertLessEqual(
                    float(metrics["total_tps"]),
                    seq_config["tps"],
                )
            if "ttft" in seq_config.keys():
                self.assertLessEqual(
                    float(metrics["mean_ttft"]),
                    seq_config["ttft"],
                )
            logger.info(f"---------- Finish long seq test: {seq_type} ----------")

    def run_evalscope(self):
        cmd_args = [
            "evalscope/bin/python",
            "-m",
            "sglang.test.ascend.e2e.evalscope_utils",
            "--model",
            self.model,
            "--api-url",
            str(self.base_url),
            "--eval-type",
            "openai_api",
            "--datasets",
            ["gsm8k"],
            "--dataset-args",
            {"gsm8k": {}},
            "--eval-batch-size",
            128,
            "--work-dir",
            "./",
        ]
        logger.info(f"Command: {' '.join(cmd_args)}")

        subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
