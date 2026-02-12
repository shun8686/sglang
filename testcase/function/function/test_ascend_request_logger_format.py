import subprocess
import unittest
import tempfile
from abc import ABC

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


class TestrequestABC(ABC):
    log_requests_format = None

    @classmethod
    def setUpClass(cls):
        cls.text_message = "Receive: obj=GenerateReqInput"
        cls.text_message1 = "Finish: obj=GenerateReqInput"
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--log-requests",
            "--log-requests-level",
            "2",
            "--log-requests-format",
            cls.log_requests_format,
            "--skip-server-warmup",
            "--log-requests-target",
            "stdout",
            cls.temp_dir,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestRequestLoggerFormatText(TestrequestABC, CustomTestCase):
    log_requests_format = "text"

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
        log_path = run_command(f"ls -l {self.temp_dir}").split(" ")[-1]
        self.assertNotEqual(log_path, None)
        result = run_command(f"cat {log_path} | grep '{self.text_message}'")
        result1 = run_command(f"cat {log_path} | grep '{self.text_message1}'")
        self.assertNotEqual(result, None)
        self.assertNotEqual(result1, None)


class TestRequestLoggerFormatJson(TestrequestABC, CustomTestCase):
    log_requests_format = "json"

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
        log_path = run_command(f"ls -l {self.temp_dir}").split(" ")[-1]
        self.assertNotEqual(log_path, None)
        result = run_command(f"cat {log_path} | grep 'timestamp'")
        result1 = run_command(f"cat {log_path} | grep 'rid'")
        result2 = run_command(f"cat {log_path} | grep 'obj'")
        result3 = run_command(f"cat {log_path} | grep 'out'")
        self.assertNotEqual(result, None)
        self.assertNotEqual(result1, None)
        self.assertNotEqual(result2, None)
        self.assertNotEqual(result3, None)


if __name__ == "__main__":
    unittest.main()
