import os
import unittest
import requests
from types import SimpleNamespace

# from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH, get_device_ids
from sglang.test.run_eval import run_eval
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNumReservedDecodeTokens(TestDisaggregationBase):
    """Testcase: Verify that in the PD disaggregation scenario, the model accuracy remains
    uncompromised when the Decode service is launched with the parameters --num-reserved-decode-tokens 128
    and --disaggregation-decode-polling-interval 2 configured.

    [Test Category] Parameter
    [Test Target] --num-reserved-decode-tokens; --disaggregation-decode-polling-interval
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = (
            [
                "--disaggregation-mode",
                "prefill",
                # "--disaggregation-decode-tp",
                "--tp-size",
                "2",
                "--enable-dp-attention",
                "--dp",
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

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = (
            [
                "--disaggregation-mode",
                "decode",
                "--base-gpu-id",
                8,
                "--enable-dp-attention",
                "--dp",
                "2",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.8,
                "--num-reserved-decode-tokens",
                128,
                "--disaggregation-decode-polling-interval",
                2,
            ]
        )
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_inference(self, max_new_tokens=32):
        """Send a basic inference request to test inference function."""
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        return response.text

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
