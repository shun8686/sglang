import subprocess
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


def run_command(cmd, shell=True):
    """Execute system command and return stdout

    parameter:
        cmd: command to execute
        shell:
        True, Execute command in shell
        False, Commands are invoked directly without shell parsing
    return:
        The result of executing the command
    """
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None


class TestSleepOnIdle(CustomTestCase):
    """Testcase: Test configuration --sleep-on-idle, send request, interence successful.

    [Test Category] Parameter
    [Test Target] --sleep-on-idle
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            [
                "--sleep-on-idle",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        ]
        cls.cpu_values = []
        cls.process = []

    def test_sleep_on_idle(self):
        for i, common_arg in enumerate(self.other_args):
            process = popen_launch_server(
                LLAMA_3_2_1B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=common_arg,
            )

            time.sleep(5)

            pid = run_command(
                f"ps -ef | awk -v ppid = {self.process.pid} '/sglang::scheduler_TP0/ && $3 == ppid' | tr -s '' | cut -d'' -f2")
            if not pid:
                self.fail("Failed to get child process PID")
            cpu_usage = run_command(f"ps -p {pid} -o %cpu --no-headers | xargs")
            if not cpu_usage:
                self.fail("Failed to get CPU usage")

            cpu_float = float(cpu_usage)
            self.cpu_values.append(cpu_float)
            print(f"***********{self.cpu_values[1]=}")
            print(f"***********{self.cpu_values[0]=}")

            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)

            # save process
            self.process.append(process)

    @classmethod
    def tearDownClass(cls):
        for process in cls.process:
            kill_process_tree(process.pid)

    def test_cpu_reducation(self):
        self.assertGreater(self.cpu_values[0], self.cpu_values[1], f"CPU usage shoule drop with --sleep-on-idle")


if __name__ == "__main__":
    unittest.main()
