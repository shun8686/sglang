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


class TestEnableMixedChunk(CustomTestCase):
    """Testcase：Verify the correctness of --enable-mixed-chunk feature and related APIs (health/generate/server-info) availability.

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk
    """

    def test_enable_mixed_chunk(self):
    """Verify the availability of related APIs and the correctness of --enable-mixed-chunk parameter configuration"""
        other_args = (
            [
                "--enable-mixed-chunk",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        # Launch the service with mixed chunk feature enabled
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        # Verify that the /generate inference interface returns 200 OK and contains the expected result "Paris"
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

        # Verify enable_mixed_chunk parameter is correctly set
        response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["enable_mixed_chunk"], True)
        kill_process_tree(process.pid)


if __name__ == "__main__":

    unittest.main()
