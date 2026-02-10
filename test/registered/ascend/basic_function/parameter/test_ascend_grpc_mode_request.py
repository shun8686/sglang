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
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen2-0.5B-Instruct"
        pass


    @classmethod
    def tearDownClass(cls):
        pass


    def test_grpc_mode(self):
        # response = requests.post(
        #     f"http://127.0.0.1:20000/generate",
        #     json={
        #         "text": "The capital of France is",
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_new_tokens": 32,
        #         },
        #     },
        # )
        #
        # print("============http://127.0.0.1:20000==============")
        # print(f"{response.status_code=}")
        # print(f"{response.text=}")

        response = requests.post(
            f"http://127.0.0.1:21000/generate",
            json={
                "model": self.model,
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        print("============http://127.0.0.1:21000==============")
        print(f"{response.status_code=}")
        print(f"{response.text=}")


        # response = requests.post(
        #     f"grpc://127.0.0.1:20000/generate",
        #     json={
        #         "text": "The capital of France is",
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_new_tokens": 32,
        #         },
        #     },
        # )

        # print("============grpc://127.0.0.1:20000==============")
        # print(f"{response.status_code=}")
        # print(f"{response.text=}")
        #
        #
        # response = requests.post(
        #     f"grpc://127.0.0.1:21000/generate",
        #     json={
        #         "text": "The capital of France is",
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_new_tokens": 32,
        #         },
        #     },
        # )
        #
        # print("============grpc://127.0.0.1:21000==============")
        # print(f"{response.status_code=}")
        # print(f"{response.text=}")

        # self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        # self.assertIn("Paris", response.text, "The inference result does not include Paris.")


if __name__ == "__main__":
    unittest.main()

