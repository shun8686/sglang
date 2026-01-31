import io
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)


class BaseTestSoftWatchdog:
    """
    Testcase：Verify that the correctness and stability of --soft-watchdog mechanism under long-running generation
    requests on GPU backend

    [Test Category] Parameter
    [Test Target] --soft-watchdog-timeout
    """

    env_override = None
    expected_message = None

    @classmethod
    def setUpClass(cls):
        cls.model_path = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        with cls.env_override():
            cls.process = popen_launch_server(
                cls.model_path,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--soft-watchdog-timeout",
                    "20",
                    "--skip-server-warmup",
                ],
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_triggers(self):
        print("Start call /generate API", flush=True)
        try:
            requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Hello, please repeat this sentence for 1000 times.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                },
                timeout=30,
            )
        except requests.exceptions.ReadTimeout as e:
            print(f"requests.post timeout (but expected): {e}")
        print("End call /generate API", flush=True)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(self.expected_message, combined_output)


class TestSoftWatchdog(BaseTestSoftWatchdog, CustomTestCase):
    expected_message = "Watchdog timeout"

if __name__ == "__main__":
    unittest.main()
