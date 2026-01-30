import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.ascend.test_ascend_utils import DeepSeek_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-16-npu-a3", nightly=True)

class TestDPAttentionMinimumTokenLoadBalance(CustomTestCase):
    """
    Testcase：Verify that the inference is successful when --load-balance-method is set to round_robin, queue and minimum_tokens

    [Test Category] Parameter
    [Test Target] --load-balance-method round_robin/queue/minimum_tokens
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DeepSeek_R1_0528_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.modes = ["round_robin", "queue", "minimum_tokens"]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        for mode in self.modes:
            other_args = [
                "--trust-remote-code",
                "--tp",
                "16",
                "--enable-dp-attention",
                "--dp",
                "1",
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                "2",
                "--load-balance-method",
                mode,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--quantization",
                "modelslim",
                "--mem-fraction-static",
                "0.75",
            ]

            process = popen_launch_server(
                self.model_path,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

            try:
                args = SimpleNamespace(
                    base_url=self.base_url,
                    model=self.model,
                    eval_name="mgsm_en",
                    num_examples=10,
                    num_threads=1024,
                )

                metrics = run_eval(args)
                print(f"{metrics=}")
                self.assertGreater(metrics["score"], 0.5)
            finally:
                kill_process_tree(process.pid)



if __name__ == "__main__":
    unittest.main()
