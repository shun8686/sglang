import os
import tempfile
import unittest

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Base class for disaggregation with HiCache tests"""
    out_file = open("./out_log.txt", "w+", encoding="utf-8")
    err_file = open("./err_log.txt", "w+", encoding="utf-8")

    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = QWEN3_32B_WEIGHTS_PATH

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.decode_url + "/health")

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--base-gpu-id",
            "0",
            "--mem-fraction-static",
            "0.8",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-decode-enable-fake-auto",
            "--load-balance-method",
            "follow_bootstrap_room",
        ]
        env = {
            **os.environ,
        }
        cls.process_decode = popen_launch_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    def test_disaggregation_decode_enable_fake_auto(self):
        response = requests.get(f"{self.decode_url}/health_generate")
        self.assertEqual(response.status_code, 200)

        text1 = "The capital of France is"

        response = requests.post(
            f"{self.decode_url}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print(response.json())


if __name__ == "__main__":
    unittest.main()
