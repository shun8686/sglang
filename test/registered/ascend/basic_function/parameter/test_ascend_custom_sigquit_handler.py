import os
import shutil
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendDeleteCkptAfterLoading(CustomTestCase):
    """
    Testcase：

    [Test Category] Parameter
    [Test Target] --custom-sigquit-handler
    """

    @classmethod
    def setUpClass(cls):
        # cls.model = QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen2-0.5B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--custom-sigquit-handler",
            "my_sigquit_handler"
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        print("****************************teardown_class**************************")
        kill_process_tree(cls.process.pid)


    def test_delete_ckpt_after_loading(self):
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
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )




if __name__ == "__main__":
    unittest.main()


def my_sigquit_handler(self):
    print("******************************my_sigquit_handler***************************")
