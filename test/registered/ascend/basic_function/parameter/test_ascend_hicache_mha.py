import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import Qwen2_5_7B_Instruct_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    Qwen2_5_7B_Instruct_WEIGHTS_PATH: {
        "accuracy": 0.85,
        "latency": 150,
        "output_throughput": 30,
    },
}


class TestAscendMhaHicache(CustomTestCase):
    """
    Testcase：Verify the correctness and performance of the hierarchical cache in multi-head attention operator

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache, --hicache-ratio 1.2
    """

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            1.2,
        ]

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *self.common_args,
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
                        TEST_MODEL_MATRIX[model]["accuracy"],
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
                    ],
                )

                print(f"##=== {model} throughput: {output_throughput} ===##")

                if is_in_ci():
                    self.assertGreater(
                        output_throughput,
                        TEST_MODEL_MATRIX[model]["output_throughput"],
                    )


if __name__ == "__main__":
    unittest.main()
