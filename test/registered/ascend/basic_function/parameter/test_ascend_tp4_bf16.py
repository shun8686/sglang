import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=500, suite="nightly-4-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH: {
        "accuracy": 0.90,
        "latency": 180,
        "output_throughput": 20,
    },
}


class TestAscendTp4Bf16(CustomTestCase):
    """
    Testcase：Verify the accuracy and throughput of Qwen3-30B-A3B on gsm8k dataset when cuda graph mode is disabled and
    tp size is 4

    [Test Category] Parameter
    [Test Target] --disable-cuda-graph, --tp-size 4
    """

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.7,
            "--max-running-requests",
            32,
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--cuda-graph-max-bs",
            32,
            "--tp-size",
            4,
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
