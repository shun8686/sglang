import json
import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.test.run_eval import run_eval
# from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_pd_server,
)


class TestNumReservedDecodeTokens(TestDisaggregationBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
        )    
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
        prefill_args = (
            [
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-decode-tp",
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

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.2)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
