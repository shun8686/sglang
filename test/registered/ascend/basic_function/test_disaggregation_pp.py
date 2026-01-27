import os
import time
import unittest
import requests
from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestDisaggregationPPAccuracy(TestDisaggregationBase):
    """Test class for accuracy verification of disaggregated Prefill-Decode (PP) architecture.

    Core Purpose:
    - Launch disaggregated Prefill/Decode services with TP/PP parallelism on Ascend backend
    - Verify GSM8K mathematical reasoning accuracy meets threshold (>0.24)
    - Ensure stable operation of disaggregated architecture with RDMA/Ascend transfer backend
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        env = os.environ.copy()
        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )


    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp",
            "2",
            "--base-gpu-id",
            "4",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.24)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
