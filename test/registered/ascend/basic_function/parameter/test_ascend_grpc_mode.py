import os
from time import sleep

import requests
import unittest
import subprocess
import time
from urllib.parse import urlparse
from sglang.srt.utils import kill_process_tree
#from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server, popen_with_error_check,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendGrpcModePDMixed(CustomTestCase):
    """
    Testcase：Verify that gRPC requests are correctly received and process when gRPC mode is enabled.

    [Test Category] Parameter
    [Test Target] --grpc-mode
    """

    @classmethod
    def setUpClass(cls):
        # cls.model = QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen2-0.5B-Instruct"
        cls.grpc_base_url = f"grpc://127.0.0.1:30111"
        cls.grpc_url = urlparse(cls.grpc_base_url)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)

        # worker_args = [
        #     "--grpc-mode"
        # ]
        # cls.worker_process = popen_launch_server(
        #     cls.model,
        #     cls.base_url,
        #     timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        #     other_args=worker_args,
        # )
        cls.out_log_file_name = "./grpc_mode_out_log.txt"
        cls.out_log_file = open("./grpc_mode_out_log.txt", "w+", encoding="utf-8")
        worker_command = [
            "python3",
            "-m", "sglang.launch_server",
            "--model-path", cls.model,
            "--grpc-mode",
            "--host", cls.grpc_url.hostname, "--port", str(cls.grpc_url.port),
        ]
        cls.worker_process = subprocess.Popen(worker_command, stdout=cls.out_log_file, stderr=None)
        cls.wait_grpc_server_ready(cls.out_log_file)
        # cls.wait_server_ready(cls.grpc_base_url + "/health")
        # print("=========server is ready==========")
        # sleep(100)

        router_command = [
            "python3",
            "-m", "sglang_router.launch_router",
            "--worker-urls", cls.grpc_base_url,
            "--host", cls.url.hostname, "--port", str(cls.url.port),
            "--model-path", cls.model,
        ]

        # cls.router_process = subprocess.Popen(router_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        cls.router_process = popen_with_error_check(router_command)

        cls.wait_server_ready(cls.base_url + "/health")
        print("=========server is ready==========")

    @classmethod
    def tearDownClass(cls):

        kill_process_tree(cls.router_process.pid)
        kill_process_tree(cls.worker_process.pid)
        cls.out_log_file.close()
        os.remove(cls.out_log_file_name)

    @classmethod
    def wait_grpc_server_ready(cls, out_log_file, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                out_log_file.seek(0)
                content = out_log_file.read()
                if len(content) > 0 and "The server is fired up and ready to roll" in content:
                    return
            except Exception:
                pass
            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server failed to start in {timeout}s")

            time.sleep(1)


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
            # f"{self.base_url}/generate",
            f"http://127.0.0.1:21000/generate",
            json={
                "text": "The capital of France is",
                "model": self.model,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

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

        print("============base==============")
        print(f"{response.status_code=}")
        print(f"{response.text=}")

        # self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        # self.assertIn("Paris", response.text, "The inference result does not include Paris.")


if __name__ == "__main__":
    unittest.main()
