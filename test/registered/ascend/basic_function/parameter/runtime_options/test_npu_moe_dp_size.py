import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True, disabled="https://github.com/Ascend/sglang/issues/134")


class TestNpuMoeDpSize(CustomTestCase):
    """Test Case: Verify that the accuracy of the DeepSeek-V2-Lite model on the GSM8K dataset does not degrade after configuring the --moe-dp-size parameter.

    [Test Category] Parameter
    [Test Target] --moe-dp-size
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V2_LITE_WEIGHTS_PATH
        cls.accuracy = 0.34
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--tp-size",
            4,
            "--moe-dp-size",
            2,
            "--disable-radix-cache",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
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
            self.accuracy,
        )


if __name__ == "__main__":
    unittest.main()
