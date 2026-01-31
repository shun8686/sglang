import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_bench_offline_throughput,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=500, suite="nightly-4-npu-a3", nightly=True)


class TestAscendGsm8kAndThroughput(CustomTestCase):
    """
    Testcase：Verify the accuracy and throughput of Qwen3-30B-A3B on gsm8k dataset when cuda graph mode is disabled and
    tp size is 4

    [Test Category] Parameter
    [Test Target] --disable-cuda-graph, --tp-size 4
    """
    TEST_MODEL_MATRIX = {}
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.models = cls.TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
        ]

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1800,
                    other_args=[
                        *self.common_args,
                        *self.extra_args,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        num_shots=5,
                        data_path=None,
                        num_questions=1319,
                        max_new_tokens=512,
                        parallel=128,
                        host=f"http://{self.url.hostname}",
                        port=int(self.url.port),
                    )

                    metrics = run_eval_few_shot_gsm8k(args)
                    self.assertGreaterEqual(
                        metrics["accuracy"],
                        self.TEST_MODEL_MATRIX[model]["accuracy"],
                    )
                finally:
                    kill_process_tree(process.pid)

    def test_b_throughput(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing throughput: {model} ===##")

                output_throughput = run_bench_offline_throughput(
                    model,
                    [
                        *self.common_args,
                        *self.extra_args,
                    ],
                )

                print(f"##=== {model} throughput: {output_throughput} ===##")

                self.assertGreater(
                    output_throughput,
                    self.TEST_MODEL_MATRIX[model]["output_throughput"],
                )


if __name__ == "__main__":
    unittest.main()
