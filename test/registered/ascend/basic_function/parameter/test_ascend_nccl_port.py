import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import run_command_and_capture_output
from sglang.test.ascend.test_ascend_utils import LLAMA_2_7B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


# def run_command(cmd, shell=True):
#     try:
#         result = subprocess.run(
#             cmd, shell=shell, capture_output=True, text=True, check=True
#         )
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         print(f"execute command error: {e}")
#         return None


class TestNcclPort(CustomTestCase):
    """Testcase: Test the basic functions of nccl-port
                 Test nccl-port configured, the inference request successful.

    [Test Category] Parameter
    [Test Target] --nccl-port
    """

    model = LLAMA_2_7B_WEIGHTS_PATH

    def test_nccl_port(self):
        """Test the --nccl-port argument."""
        other_args = (
            [
                "--nccl-port",
                "8111",
                "--tp-size",
                "2",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
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
            print(response.text)
            result = run_command_and_capture_output("lsof -i:8111")
            self.assertIn("*:8111 (LISTEN)", result)

        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
