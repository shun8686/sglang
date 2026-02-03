import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH: {
        "accuracy": 0.90,
        "latency": 1000,
        "output_throughput": 6,
    },
}


class TestAscendDistTimeout(CustomTestCase):
    """Testcase: Enable MTP features
    configuring '--speculative-draft-attention-backend' and '--speculative-moe-runner-backend' did not degrade inference accuracy.

    [Test Category] Parameter
    [Test Target] --speculative-draft-attention-backend; --speculative-moe-runner-backend
    """

    os.environ["HCCL_BUFFSIZE"] = "2048"
    os.environ["SGLANG_ENABLE_OVERLAP_PLAN_SITEAM"] = "1"
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
    env = os.environ.copy()
    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            0.7,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            32768,
            "--tp-size",
            16,
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            1,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            2,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--max-running-requests",
            64,
            "--speculative-draft-attention-backend",
            "ascend",
            "--speculative-moe-runner-backend",
            "auto",
            ]

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")
                other_args =  self.common_args
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1500,
                    other_args=[
                         *other_args,
                     ],
                    env=self.env,
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
                    print(f"metrics['accuracy']=")
                    self.assertGreaterEqual(
                    metrics["accuracy"],
                    TEST_MODEL_MATRIX[model]["accuracy"],
                    )
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
