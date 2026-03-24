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

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--watchdog-timeout",
                10,
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
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello, please repeat this sentence for 1000 times.",
                "sampling_params": {"max_new_tokens": 100, "temperature": 0},
            },
            timeout=30,
        )

        # Logs contain service crash keywords
        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        expected_timeout_message = "Scheduler watchdog timeout (self.watchdog_timeout=1.0, self.soft=False)"
        expected_crash_message = "SIGQUIT received."
        self.assertIn(
            expected_timeout_message,
            combined_output,
            f"Expected timeout message '{expected_timeout_message}' not found in logs"
        )
        self.assertIn(
            expected_crash_message,
            combined_output,
            f"Expected crash message '{expected_crash_message}' not found in logs"
        )

        # Verify the process has exited (watchdog triggered crash)
        self.assertIsNotNone(
            self.process.poll(),
            f"Process should exit after watchdog timeout, but still running"
        )

class TestWatchdogSchedulerInit(BaseTestWatchdog, CustomTestCase):
    """Test Case: Verify that Scheduler initialization blocking triggers watchdog timeout and the service crashes

    [Test Category] Parameter
    [Test Target] --watchdog-timeout
    """
    env_override = lambda: envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.override(30)
    expected_crash_message = "Scheduler watchdog timeout, crashing server to prevent hanging"


if __name__ == "__main__":
    unittest.main()
