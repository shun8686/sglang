import json
import os
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.test.run_eval import run_eval
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_pd_server,
)

QWEN3_32B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B-Eagle3"


class TestNumReservedDecodeTokens(TestDisaggregationBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
        cls.accuracy = 0.81
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
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
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--quantization",
                "modelslim",
                "--disable-radix-cache",
                "--speculative-draft-model-quantization",
                "unquant",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                QWEN3_32B_EAGLE_MODEL_PATH,
                "--speculative-num-steps",
                "4",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "5",
                "--speculative-attention-mode",
                "decode",
                "--tp-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--disable-cuda-graph",
                "--dtype",
                "bfloat16",
            ]
        )
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
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
                "4",
                "--disaggregation-transfer-backend",
                "ascend",
                "--num-reserved-decode-tokens",
                128,
                "--disaggregation-decode-polling-interval",
                2,
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--quantization",
                "modelslim",
                "--disable-radix-cache",
                "--speculative-draft-model-quantization",
                "unquant",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                QWEN3_32B_EAGLE_MODEL_PATH,
                "--speculative-num-steps",
                "4",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "5",
                "--speculative-attention-mode",
                "prefill",
                "--tp-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--disable-cuda-graph",
                "--dtype",
                "bfloat16",
            ]
        )
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_a_gsm8k(self):
        print(f"##=== Testing accuracy: {cls.model} ===##")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            cls.accuracy,
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
