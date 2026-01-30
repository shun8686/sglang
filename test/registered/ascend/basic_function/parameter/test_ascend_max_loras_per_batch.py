import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestLoraPaths(CustomTestCase):
    """Testcase：Verify the correctness of --max-loras-per-batch=1 and related APIs availability.

    [Test Category] Parameter
    [Test Target] --max-loras-per-batch
    """

    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--max-loras-per-batch",
                1,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_paths(self):
        """Core test case: Verify the availability of 3 core APIs and the correctness of --max-loras-per-batch parameter configuration"""
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

        # Verify max_loras_per_batch parameter is correctly set in server info
        response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["max_loras_per_batch"], 1)

if __name__ == "__main__":
    unittest.main()
