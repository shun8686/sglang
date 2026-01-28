import json
import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_pd_server,
)

os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDisaggregationDecodeDp(TestDisaggregationBase):
    """Testcase：Verify the correctness of --disaggregation-decode-dp=2 and Prefill/Decode disaggregated services availability on Ascend NPU backend.

    [Test Category] Parameter
    [Test Target] --disaggregation-decode-dp; --disaggregation-mode; --disaggregation-transfer-backend
    """

    @classmethod
    def setUpClass(cls):
         """Test class initialization: Launch Prefill/Decode disaggregated services and load balancer, then wait for services to be ready"""
        super().setUpClass()
        cls.model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
        )
        #os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        #env = os.environ.copy()

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        """Launch the Prefill service with disaggregation-decode-dp=2 configuration for Ascend NPU"""
        prefill_args = (
            [
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-decode-dp",
                "2",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
        )
        
        env = os.environ.copy()

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        """Launch the Decode service with specified configuration for Ascend NPU (disaggregated architecture)"""
        decode_args = (
            [
                "--disaggregation-mode",
                "decode",
                "--base-gpu-id",
                "2",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
            ]
        )

        env = os.environ.copy()

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_disaggregation_decode_dp(self):
         """Verify the availability of disaggregated services and the correctness of --disaggregation-decode-dp=2 configuration"""
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(self.prefill_url + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["disaggregation_decode_dp"], 2)

    @classmethod
    def tearDownClass(cls):
        """Test class cleanup: Remove the Ascend MF store environment variable and call parent class cleanup to terminate all processes"""
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
