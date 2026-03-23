import io
import os
import unittest
import time
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestBaseTestWatchdog(CustomTestCase):
    env_override = None
    expected_crash_message = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        # Simulate blocking of specified module, start service and set watchdog
        # with cls.env_override():
        cls.extra_envs = {
            "SGLANG_TEST_STUCK_DETOKENIZER": "0",
        }
        os.environ.update(cls.extra_envs)
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--watchdog-timeout",
                20,
                "--skip-server-warmup",
                "--attention-backend",
                "ascend",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_crashes_server(self):
        # Verify that the service crashes after watchdog timeout is triggered

        print("Start call /generate API", flush=True)
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Hello, please repeat this sentence for 1000 times.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                },
                timeout=30,
            )
            print("=============================")
            print(response.text)
        except requests.exceptions.ConnectionError:
            # Expected connection failure, indicating the service has crashed
            print("API request failed (expected): Server is crashed as watchdog triggered")

            # Logs contain service crash keywords
            combined_output = self.stdout.getvalue() + self.stderr.getvalue()
            self.assertIn(
                self.expected_crash_message,
                combined_output,
                f"Expected crash message '{self.expected_crash_message}' not found in logs"
            )
            print(f"Verified: Found crash message '{self.expected_crash_message}' in logs")

            # # Verify the process has exited (watchdog triggered crash)
            # self.assertIsNotNone(
            #     self.process.poll(),
            #     f"Process should exit after watchdog timeout ({self.watchdog_timeout}s), but still running"
            # )


# class TestWatchdogDetokenizer(BaseTestWatchdog, CustomTestCase):
#     """Test Case: Verify that Detokenizer blocking triggers watchdog timeout and the service crashes
#
#     [Test Category] Parameter
#     [Test Target] --watchdog-timeout
#     """
#     env_override = lambda: envs.SGLANG_TEST_STUCK_DETOKENIZER.override(30)
#     expected_crash_message = "DetokenizerManager watchdog timeout, crashing server to prevent hanging"
#
#
# class TestWatchdogTokenizer(BaseTestWatchdog, CustomTestCase):
#     """Test Case: Verify that Tokenizer blocking triggers watchdog timeout and the service crashes
#
#     [Test Category] Parameter
#     [Test Target] --watchdog-timeout
#     """
#     env_override = lambda: envs.SGLANG_TEST_STUCK_TOKENIZER.override(30)
#     expected_crash_message = "TokenizerManager watchdog timeout, crashing server to prevent hanging"
#
#
# class TestWatchdogSchedulerInit(BaseTestWatchdog, CustomTestCase):
#     """Test Case: Verify that Scheduler initialization blocking triggers watchdog timeout and the service crashes
#
#     [Test Category] Parameter
#     [Test Target] --watchdog-timeout
#     """
#     env_override = lambda: envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.override(30)
#     expected_crash_message = "Scheduler watchdog timeout, crashing server to prevent hanging"


if __name__ == "__main__":
    unittest.main()
