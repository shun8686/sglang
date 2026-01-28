import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLoraTargetModules(CustomTestCase):
    """Test class for Llama-3.2-1B with --lora-target-modules=all parameter.

    Tests functionality with LORA target modules set to 'all':
    - health-check: /health_generate API returns 200 OK
    - inference: Generate API returns valid result (200 OK + "Paris" in response)
    - server-info: get_server_info API confirms lora_target_modules is ["all"]
    """

    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--lora-target-modules",
                "all",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_target_modules(self):
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

        # Verify lora_target_modules parameter is correctly set in server info
        response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lora_target_modules"], ["all"])


if __name__ == "__main__":

    unittest.main()
