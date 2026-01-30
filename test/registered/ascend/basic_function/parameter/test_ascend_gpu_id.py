import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None


class TestGpuId(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_id = 0
        cls.step = 2
        other_args = (
            [
                "--base-gpu-id",
                cls.device_id,
                "--tp-size",
                "2",
                "--gpu-id-step",
                cls.step,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )

        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gpu_id(self):
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
        result = run_command("npu-smi info")
        print(result)
        result1 = run_command(
           "npu-smi info | grep '/ 65536' | awk -F '|' '{print $4}' | awk '{print $5}' | awk -F '/' '{print $1}'"
        )
        result2 = result1.split("\n")
        print(result2)
        for i in range(len(result2)-1):
            if i in [self.device_id, self.device_id+self.step]:
                self.assertGreater(int(result2[i]), 10000)
            else:
                self.assertLess(int(result2[i]), 5000)


if __name__ == "__main__":
    unittest.main()
