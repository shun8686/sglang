import math
import os
import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None


class TestDecodeLogInterval(CustomTestCase):
    decode_numbers = 10
    @classmethod
    def setUpClass(cls):

        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
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
        response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        # decode_number = response.json()["decode_log_interval"]
        decod_batch_result = math.ceil(max_tokens / self.decode_numbers)
        print(f"******result={result}")
        print(f"******result={decod_batch_result}")
        self.assertIn(decod_batch_result, int(result.strip()))
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

class TestDecodeLogIntervalOther(TestDecodeLogInterval):
    decode_numbers = 30


if __name__ == "__main__":

    unittest.main()
