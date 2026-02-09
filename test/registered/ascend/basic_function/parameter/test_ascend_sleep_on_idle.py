import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestSleepOnIdle(CustomTestCase):
    """Testcase: Test configuration --sleep-on-idle, send request, interence successful.

    [Test Category] Parameter
    [Test Target] --sleep-on-idle
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--sleep-on-idle",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_sleep_on_idle(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        # response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        # self.assertEqual(response.status_code, 200)
        # self.assertEqual(response.json()["sleep_on_idle"], True)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        # self.assertGreaterEqual(metrics["accuracy"], 0.86)


if __name__ == "__main__":
    unittest.main()
