import datetime
import sys
import unittest
from urllib.parse import urlparse

from lts_utils import (
    run_bench_serving,
    run_gsm8k,
)

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    E2E_TOLERANCE,
    OUTPUT_TOKEN_THROUGHPUT_TOLERANCE,
    TPOT_THRESHOLD,
    TPOT_TOLERANCE_HIGH,
    TPOT_TOLERANCE_LOW,
    TTFT_TOLERANCE,
)
from sglang.test.test_utils import CustomTestCase

MODEL_PATH = "/root/.cache/modelscope/hub/models/Eco-Tech/Kimi-K2.5-w4a8"


class TestLTSKimi(CustomTestCase):
    model = MODEL_PATH
    dataset_name = "random"
    dataset_path = "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
    backend = ("sglang",)
    request_rate = None
    max_concurrency = 96
    num_prompts = 96
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    warmup_requests = 16
    image_resolution = None
    image_count = None
    seed = None
    ttft = 15000
    tpot = 55
    output_token_throughput = 1350
    accuracy = 0.80
    host = "127.0.0.1"
    port = 8100

    def _assert_metrics(self, metrics):
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
        print(f"========== Start 3.5k/1.5k benchmark test ==========\n")
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
        print(f"Starting benchmark with parameters: {bench_params}")

        metrics = None
        for i in range(run_cycles):
            print(f"Running benchmark, {i + 1}/{run_cycles}")
            metrics = run_bench_serving(**bench_params)

        self._assert_metrics(metrics)
        print("res_ttft is " + str(metrics["mean_ttft"]))
        print("res_tpot is " + str(metrics["mean_tpot"]))
        print("res_output_token_throughput is " + str(metrics["total_tps"]))
        print(f"========== 3.5k/1.5k benchmark test PASSED ==========\n")

    def run_gsm8k(self):
        metrics = run_gsm8k(host=f"http://{self.host}", port=self.port)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )
        print(f"========== gsm8k test PASSED ==========\n")

    def test_lts_kimi_k25(self):
        i = 0
        while True:
            i = i + 1
            time_str_1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={time_str_1}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_file = "./log/lts_test_deepseek_r1_" + time_str + ".log"

    with open(log_file, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f

        try:
            unittest.main(verbosity=2)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Test log saved to {log_file}")
