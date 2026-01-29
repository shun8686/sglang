import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import Llama_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestTboTokenDistributionThresholdBase(CustomTestCase):
   """Testcase：Verify the correctness of --tbo-token-distribution-threshold (0.8) and related(/generate /get_server_info) API availability.

    [Test Category] Parameter
    [Test Target] --tbo-token-distribution-threshold;
    """

    tbo_token_distribution_threshold = 0.8

    @classmethod
    def setUpClass(cls):
        # Test class initialization: Launch the service with TBO token distribution threshold configured
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tbo-token-distribution-threshold",
                cls.tbo_token_distribution_threshold,
            ]
        )

        cls.process = popen_launch_server(
            Llama_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tbo_token_distribution_threshold(self):
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
        print(response.text)
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

        # Verify tbo_token_distribution_threshold parameter is correctly set
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(response.json())
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertEqual(
            response.json()["tbo_token_distribution_threshold"],
            self.tbo_token_distribution_threshold,
            f"--tbo-token-distribution-threshold is not equal to {self.tbo_token_distribution_threshold}."
        )


class TestDisableTboTokenDistributionThreshold(TestTboTokenDistributionThresholdBase):
    """Testcase：Verify the correctness of --tbo-token-distribution-threshold (0, disabled) and related API availability.

    [Test Category] Parameter
    [Test Target] --tbo-token-distribution-threshold
    """
    tbo_token_distribution_threshold = 0


if __name__ == "__main__":
    unittest.main()
