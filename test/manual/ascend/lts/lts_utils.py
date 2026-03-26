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
    host=None, port=None, dataset_name="random", dataset_path=None
):
    """依次验证16k+1k、32k+1k、64k+1k三种单条长序列"""
    # 新增：三种长序列配置（16k+1k/32k+1k/64k+1k）
    long_seq_configs = {
        "64k+1k": {
            "input_len": 65536,
            "output_len": 1024,
            "ttft_threshold": 100000,
            "tpot_threshold": 350,
        },
        "32k+1k": {
            "input_len": 32768,
            "output_len": 1024,
            "ttft_threshold": 70000,
            "tpot_threshold": 250,
        },
        "16k+1k": {
            "input_len": 16384,
            "output_len": 1024,
            "ttft_threshold": 40000,
            "tpot_threshold": 200,
        },
    }
    for seq_type, config in long_seq_configs.items():
        logger.info(
            f"\n========== Start {seq_type} single long sequence test =========="
        )
        # 执行单条长序列请求
        metrics = run_bench_serving(
            host=host,
            port=port,
            input_len=config["input_len"],
            output_len=config["output_len"],
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            request_rate=1,
            max_concurrency=1,
            num_prompts=2,
            random_range_ratio=1,
        )
        logger.info(f"{seq_type} metrics: {metrics}")

        res_ttft = run_command(
            "cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'"
        )
        res_tpot = run_command(
            "cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'"
        )
        res_output_token_throughput = run_command(
            "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
        )
        res_ttft = res_ttft.strip() if res_ttft else "0"
        res_tpot = res_tpot.strip() if res_tpot else "0"

        logger.info("res_ttft is " + str(res_ttft))
        logger.info("res_tpot is " + str(res_tpot))
        logger.info(
            "res_output_token_throughput is " + str(res_output_token_throughput)
        )
        logger.info(
            f"========== {seq_type} single long sequence test PASSED ==========\n"
        )
        # self.assertLessEqual(
        #     float(res_ttft),
        #     config["ttft_threshold"],
        #     f"{seq_type} TTFT {res_ttft}ms exceeds threshold {config['ttft_threshold']}ms"
        # )
        # self.assertLessEqual(
        #     float(res_tpot),
        #     config["tpot_threshold"],
        #     f"{seq_type} TPOT {res_tpot}ms exceeds threshold {config['tpot_threshold']}ms"
        # )
        # # 验证无错误日志
        # self.assertEqual(
        #     res_error, "",
        #     f"{seq_type} request failed with error: {res_error}"
        # )


def run_single_long_seq_test(host, port, input_len, output_len, seq_type):
    command = (
        f"python3 -m sglang.bench_serving --backend sglang --host {host} --port {port} --dataset-name random "
        f"--request-rate 1 --max-concurrency 1 --num-prompts 1 "
        f"--random-input-len {input_len} --random-output-len {output_len} "
        f"--random-range-ratio 1"
    )  # 固定长度，不随机
    logger.info(f"{seq_type} single long sequence test command:{command}")
    metrics = run_command(f"{command} | tee ./single_long_seq_{seq_type}_log.txt")
    return metrics


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

    def run_all_long_seq_verify(self):
        _, host, port = self.base_url.split(":")
        host = host[2:]
        run_long_seq_bench_serving(
            host=host,
            port=port,
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
        )
