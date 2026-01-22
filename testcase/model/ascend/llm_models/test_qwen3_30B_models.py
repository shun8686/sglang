import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

TEST_MODEL_MATRIX = {
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "accuracy": 0.90,
        "latency": 180,
        "output_throughput": 20,
    },
}


class TestAscendTp4Bf16(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.7,
            "--max-running-requests",
            32,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--cuda-graph-max-bs",
            32,
            "--tp-size",
            2,
        ]

        cls.process = popen_launch_server(
            cls.models,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{self.url.hostname}",
                port=int(self.url.port),
                )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
                metrics["accuracy"],
                TEST_MODEL_MATRIX[self.models]["accuracy"],
                )


if __name__ == "__main__":
    unittest.main()
