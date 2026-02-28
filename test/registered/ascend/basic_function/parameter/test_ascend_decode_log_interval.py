import math
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH, run_command
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDecodeLogInterval(CustomTestCase):
    """Testcase: Test configuration --decode-log-interval is set to 10, generating 52 decode batches.

    [Test Category] Parameter
    [Test Target] --decode-log-interval
    """

    decode_numbers = 10

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--decode-log-interval",
            cls.decode_numbers,
        ]
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_decode_log_interval(self):
        max_tokens = 512
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        result = run_command("cat ./cache_err_log.txt | grep 'Decode batch' | wc -l")
        decod_batch_result = math.floor((max_tokens + 9) / self.decode_numbers)
        self.assertEqual(decod_batch_result, int(result.strip()))
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")


class TestDecodeLogIntervalOther(TestDecodeLogInterval):
    decode_numbers = 30


if __name__ == "__main__":
    unittest.main()
