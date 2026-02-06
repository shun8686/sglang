import requests
import unittest
import subprocess
import time
from urllib.parse import urlparse
from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendGrpcMode(CustomTestCase):
    """
    Testcase：

    [Test Category] Parameter
    [Test Target] --grpc-mode
    """

    @classmethod
    def setUpClass(cls):
        # cls.model = QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
        cls.model = "Qwen/Qwen2-0.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.grpc_base_url = f"grpc://127.0.0.1:20000"
        cls.grpc_url = urlparse(cls.grpc_base_url)

        worker_args = [
            "--grpc-mode", "--port", "20000",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=worker_args,
        )



        router_command = [
            "python3",
            "-m", "sglang_router.launch_router",
            "--worker-urls", cls.grpc_base_url,
            "--model-path", cls.model,
            "--reasoning-parser", "deepseek-r1",
            "--tool-call-parser", "json",
            # "--host", "0.0.0.0", "--port", "8080",
            "--host", cls.url.hostname, "--port", str(cls.url.port),
        ]
        cls.router_process = subprocess.Popen(router_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        cls.wait_server_ready(
            cls.base_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.worker_process.pid)
        kill_process_tree(cls.router_process.pid)

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(1)

    def test_grpc_mode(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertEqual(
            response.json()['grpc_mode'], True, "The Grpc mode is not started."
                                                "The fastapi root path is not correct."
        )

        response = requests.post(
            f"{self.grpc_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")


if __name__ == "__main__":
    unittest.main()
