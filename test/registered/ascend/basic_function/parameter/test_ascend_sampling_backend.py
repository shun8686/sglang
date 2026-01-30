import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.ascend.test_ascend_utils import Qwen2_5_7B_Instruct_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendSamplingBackend(CustomTestCase):
    """
    Testcase：Verify the function availability and related performance of sampling inference
    when using ascend sampling backend, with radix cache disabled and cuda graph disabled

    [Test Category] Parameter
    [Test Target] --sampling-backend=ascend, --disable-radix-cache, --disable-cuda-graph
    """

    @classmethod
    def setUpClass(cls):
        cls.model = Qwen2_5_7B_Instruct_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--sampling-backend",
                "ascend",
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.85,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

    def test_greedy(self):

        first_text = None

        # ensure the answer is identical across single response
        for _ in range(5):
            response_single = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of Germany is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            ).json()
            text = response_single["text"]
            if first_text is None:
                first_text = text

            self.assertEqual(text, first_text)

        first_text = None

        response_batch = requests.post(
            self.base_url + "/generate",
            json={
                "text": ["The capital of Germany is"] * 10,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        ).json()

        # ensure the answer is identical among the batch
        for i in range(10):
            text = response_batch[i]["text"]
            if first_text is None:
                first_text = text
            self.assertEqual(text, first_text)


if __name__ == "__main__":
    unittest.main()
