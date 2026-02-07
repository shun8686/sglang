import os
import requests
import unittest
from urllib.parse import urlparse
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendFastapiRootPath(CustomTestCase):
    """
    Testcase：Verify that the system correctly processes the root path prefix when configuring the root path prefix and
    correctly performs the route redirection behavior.

    [Test Category] Parameter
    [Test Target] --fastapi-root-path
    """

    fastapi_root_path = "test_fastapi_root_path"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static", 0.8,
            "--attention-backend", "ascend",
            "--fastapi-root-path", cls.fastapi_root_path,
        ]

        cls.out_log_file = open("./warmup_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./warmup_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./warmup_out_log.txt")
        os.remove("./warmup_err_log.txt")

    def test_delete_fastapi_root_path(self):
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
        # response url is the same as request url which doesn't contain fastapi root path
        print(f"{response.url=}")
        self.assertNotIn(
            self.fastapi_root_path, response.url,
            "The root path should not in response url."
        )
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        self.out_log_file.seek(0)
        content = self.out_log_file.read()
        self.assertTrue(len(content) > 0)
        # request should be redirected to fastapi_root_path.
        self.assertIn(f"POST {self.fastapi_root_path}/generate HTTP/1.1", content)


# class TestAscendFastapiRootPathMultiLevel(TestAscendFastapiRootPath):
#     fastapi_root_path = "/test/fastapi/root/path"
#
#
# class TestAscendFastapiRootPathErrorPath(CustomTestCase):
#     # TODO 确认错误模式
#     fastapi_root_path = "test_fastapi_root_path"


if __name__ == "__main__":
    unittest.main()
