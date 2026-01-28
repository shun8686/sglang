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


class TestEnableRequestTimeStatsLogging(CustomTestCase):
    """Testcase：Verify the correctness of request time stats logging feature and related API availability with --enable-request-time-stats-logging enabled

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """
    
    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-request-time-stats-logging",
            ]
        )

        # Launch the model server as a child process and save the process handle for subsequent termination
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

    def test_enable_request_time_stats_logging(self):
        """Core Test: Verify that the --enable-request-time-stats-logging parameter takes effect and the server functions normally

        Two-Step Verification Logic:
        1. Verify the /generate API works normally (correct inference result, 200 status code)
        2. Verify the feature is enabled in the server info API (configuration takes effect)
        """
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
        print(response.text)
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(response.json())
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertTrue(
            response.json()["enable_request_time_stats_logging"],
            "--enable-request-time-stats-logging is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
