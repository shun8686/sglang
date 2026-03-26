import logging
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
    run_bench_serving,
)
from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
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
        "input": 65536,
        "output": 1024,
        "ttft": 100000,
        "tpot": 350,
        "tps": 10,
    },
    "32k+1k": {
        "input": 32768,
        "output": 1024,
        "ttft": 70000,
        "tpot": 250,
        "tps": 10,
    },
    "16k+1k": {
        "input": 16384,
        "output": 1024,
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


def run_long_seq_bench_serving(
    host=None,
    port=None,
    dataset_name="random",
    dataset_path=None,
    seq_config=None,
):
    if seq_config is None:
        logger.warning(f"seq_config is None")
        return

    metrics = run_bench_serving(
        host=host,
        port=port,
        input_len=seq_config["input_len"],
        output_len=seq_config["output_len"],
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        request_rate=1,
        max_concurrency=1,
        num_prompts=2,
        random_range_ratio=1,
    )
    logger.info(f"metrics: {metrics}")

    res_ttft = run_command("cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'")
    res_tpot = run_command("cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'")
    res_output_token_throughput = run_command(
        "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
    )
    res_ttft = res_ttft.strip() if res_ttft else "0"
    res_tpot = res_tpot.strip() if res_tpot else "0"

    logger.info("res_ttft is " + str(res_ttft))
    logger.info("res_tpot is " + str(res_tpot))
    logger.info("res_output_token_throughput is " + str(res_output_token_throughput))


class TestAscendLtsTestCaseBase(CustomTestCase):
    host = None
    port = None
    base_url = None
    model = None
    backend = "sglang"
    dataset_name = "random"
    dataset_path = "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
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

    def run_throughput(self, run_cycles=2):
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
        }
        logger.info(f"Starting benchmark with parameters: {bench_params}")

        metrics = None
        for i in range(run_cycles):
            logger.info(f"Running benchmark, {i + 1}/{run_cycles}")
            metrics = run_bench_serving(**bench_params)

        self._assert_metrics(metrics)

    def run_gsm8k(self):
        logger.info(f"---------- Start gsm8k accuracy test ----------")
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=self.host,
            port=self.port,
        )
        metrics = run_eval_gsm8k(args)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy["gsm8k"],
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy["gsm8k"]}',
        )
        logger.info(f"---------- Gsm8k accuracy test PASSED ----------")

    def test_mmlu(self):
        logger.info(f"---------- Start mmlu accuracy test ----------")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(
            metrics["score"],
            self.accuracy["mmlu"],
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy["mmlu"]}',
        )
        logger.info(f"---------- Mmlu accuracy test PASSED ----------")

    def run_all_long_seq_verify(self, long_seq_configs=None):
        if long_seq_configs is None:
            long_seq_configs = LONG_SEQ_DEFAULT_CONFIGS
        for seq_type, seq_config in long_seq_configs.items():
            logger.info(f"---------- Start long seq test: {seq_type} ----------")
            run_long_seq_bench_serving(
                host=self.host,
                port=self.port,
                dataset_name=self.dataset_name,
                dataset_path=self.dataset_path,
                seq_config=seq_config,
            )
            logger.info(f"---------- Finish long seq test: {seq_type} ----------")
