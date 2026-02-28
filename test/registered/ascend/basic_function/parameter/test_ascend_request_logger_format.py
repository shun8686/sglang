import unittest
import tempfile
from abc import ABC

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_WEIGHTS_PATH,
    run_command
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestrequestABC(ABC):
    """Testcase: Test configure the --log-requests-format to "text", and the output log format will be "text".
    And configure the --log-requests-format to "json", and the output log format will be "json".

    [Test Category] Parameter
    [Test Target] --log-requests-format
    """

    log_requests_format = None

    @classmethod
    def setUpClass(cls):
        cls.text_message = "Receive: obj=GenerateReqInput"
        cls.text_message1 = "Finish: obj=GenerateReqInput"
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.model = LLAMA_3_2_1B_WEIGHTS_PATH
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
    """Configure the format to "test"."""

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
        result = run_command(f"grep '{self.text_message}' {self.temp_dir}/{log_path}")
        result1 = run_command(f"grep '{self.text_message1}' {self.temp_dir}/{log_path}")
        self.assertNotEqual(result, None)
        self.assertNotEqual(result1, None)


class TestRequestLoggerFormatJson(TestrequestABC, CustomTestCase):
    """Configure the format to "json"."""

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
        result = run_command(f"grep 'timestamp' {self.temp_dir}/{log_path}")
        result1 = run_command(f"grep 'rid' {self.temp_dir}/{log_path}")
        result2 = run_command(f"grep 'obj' {self.temp_dir}/{log_path}")
        result3 = run_command(f"grep 'out' {self.temp_dir}/{log_path}")
        self.assertNotEqual(result, None)
        self.assertNotEqual(result1, None)
        self.assertNotEqual(result2, None)
        self.assertNotEqual(result3, None)


if __name__ == "__main__":
    unittest.main()
