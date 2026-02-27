import os
from time import sleep
import requests
import unittest
import subprocess
import time
from urllib.parse import urlparse
from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
# from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_with_error_check, popen_launch_pd_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendGrpcModePDDisaggregation(CustomTestCase):
    """
    Testcase：Verify that gRPC requests are correctly received and process when gRPC mode is enabled.

    [Test Category] Parameter
    [Test Target] --grpc-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
        cls.prefill_url = f"grpc://127.0.0.1:20100"
        cls.prefill_port = "20100"
        cls.decode_url = f"grpc://127.0.0.1:20200"
        cls.decode_port = "20200"
        cls.base_url =  DEFAULT_URL_FOR_TEST
        cls.lb_url = cls.base_url
        cls.url = urlparse(cls.lb_url)
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        cls.process_lb, cls.process_decode, cls.process_prefill = None, None, None

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # TODO
        # Block until both
        # cls.wait_server_ready(cls.prefill_url + "/health")
        # cls.wait_server_ready(cls.decode_url + "/health")


        cls.launch_lb()
        sleep(200)

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # wait for 5 seconds
        time.sleep(5)

    @classmethod
    def start_prefill(cls):
        prefill_args = (
            [
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.4",
                "--tp-size",
                "1",
                "--base-gpu-id",
                12,
                "--grpc-mode",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disaggregation-mode",
                "prefill",
                "--port",
                cls.prefill_port,
            ]
        )

        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = (
            [
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--disable-cuda-graph",
                "--prefill-round-robin-balance",
                "--mem-fraction-static",
                "0.4",
                "--tp-size",
                "1",
                "--base-gpu-id",
                14,
                "--grpc-mode",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disaggregation-mode",
                "decode",
                "--port",
                cls.decode_port,
            ]
        )
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.url.hostname,
            "--port",
            str(cls.url.port),
        ]
        cls.process_lb = popen_with_error_check(lb_command)
    #     cls.wait_server_ready(cls.lb_url + "/health")
    #
    # @classmethod
    # def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
    #     start_time = time.perf_counter()
    #     while True:
    #         try:
    #             response = requests.get(url)
    #             if response.status_code == 200:
    #                 print(f"Server {url} is ready")
    #                 return
    #         except Exception:
    #             pass
    #
    #         if time.perf_counter() - start_time > timeout:
    #             raise RuntimeError(f"Server {url} failed to start in {timeout}s")
    #
    #         time.sleep(1)

    def test_grpc_mode(self):
        print(f"====================Testing grpc mode=======================")
        response = requests.post(
            # f"http://127.0.0.1:4567/generate",
            f"{self.base_url}/generate",
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


if __name__ == "__main__":
    unittest.main()
