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

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


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


cpu_float = 0
cpu_float_sleep = 0


class TestAscendSleepOnIdle(CustomTestCase):
    """Testcase: Test configuration --sleep-on-idle, send request, interence successful.

    [Test Category] Parameter
    [Test Target] --sleep-on-idle
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ],

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        time.sleep(5)
        # pid = run_command(
        #     f"ps -ef | awk -v ppid = {cls.process.pid} '/sglang::scheduler_TP0/ && $3 == ppid' | head -1 | awk '{{print $2}}'")
        cls.cpu_float = run_command(f"ps -p {cls.process.pid} -o %cpu --no-headers | xargs")
        print(f"***********{cls.cpu_float=}")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_Ascend_sleep_on_idle(self):
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


class TestSleepOnIdle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--sleep-on-idle",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ],

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )
        time.sleep(5)

        # pid_sleep_on = run_command(
        #     f"ps -ef | awk -v ppid = {cls.process.pid} '/sglang::scheduler_TP0/ && $3 == ppid' | head -1 | awk '{{print $2}}'")
        # if not pid_sleep_on:
        #     cls.fail("Failed to get child process PID")
        cpu_usage_sleep_on = run_command(f"ps -p {cls.process.pid} -o %cpu --no-headers | xargs")
        # if not cpu_usage_sleep_on:
        #     cls.fail("Failed to get CPU usage")
        cls.cpu_float_sleep_on = cpu_usage_sleep_on
        print(f"***********{cls.cpu_float_sleep_on=}")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_sleep_on_idle(self):
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

    def _test_cpu_reducation(self):
        self.assertGreater(cpu_float, cpu_float_sleep, f"CPU usage shoule drop with --sleep-on-idle")


if __name__ == "__main__":
    unittest.main()
