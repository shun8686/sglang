import os
import unittest

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


class TestWatchdogTimeout(CustomTestCase):
    """Testcase:

    [Test Category] Parameter
    [Test Target] --watchdog-timeout
    """

    @classmethod
    def setUpClass(cls):
        cls.expected_timeout_message = "Scheduler watchdog timeout (self.watchdog_timeout=1.0, self.soft=False)"
        cls.expected_crash_message = "SIGQUIT received."
        cls.out_log_file = open("./out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./err_log.txt", "w+", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./out_log.txt")
        os.remove("./err_log.txt")

    def test_watchdog_timeout(self):

        try:
            self.process = popen_launch_server(
                QWEN3_0_6B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--watchdog-timeout",
                    1,
                    "--skip-server-warmup",
                    "--attention-backend",
                    "ascend",
                ],
                return_stdout_stderr=(self.out_log_file, self.err_log_file),
            )
        except Exception as e:
            print(f"Server launch failed as expects:{e}")
        finally:
            self.err_log_file.seek(0)
            content = self.err_log_file.read()
            self.assertIn(
                self.expected_timeout_message,
                content,
                f"Expected timeout message '{self.expected_timeout_message}' not found in logs"
            )
            self.assertIn(
                self.expected_crash_message,
                content,
                f"Expected crash message '{self.expected_crash_message}' not found in logs"
            )



if __name__ == "__main__":
    unittest.main()
