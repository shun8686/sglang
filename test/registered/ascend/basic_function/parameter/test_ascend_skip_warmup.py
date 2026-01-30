import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)


class TestSkipServerWarmup(CustomTestCase):
    """
    Testcase：Verify that inference requests can be successfully processed after the --skip-server-warmup parameter is set.

    [Test Category] Parameter
    [Test Target] --skip-server-warmup
    """

    def test_skip_server_warmup(self):
        self.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        self.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--skip-server-warmup",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        out_log_file = open("./warmup_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./warmup_err_log.txt", "w+", encoding="utf-8")
        process = popen_launch_server(
            self.model_path,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            response = requests.get(f"{self.base_url}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)
            out_log_file.seek(0)
            content = out_log_file.read()
            self.assertTrue(len(content) > 0)
            self.assertNotIn("GET /get_model_info HTTP/1.1", content)
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove("./warmup_out_log.txt")
            os.remove("./warmup_err_log.txt")


if __name__ == "__main__":
    unittest.main()
