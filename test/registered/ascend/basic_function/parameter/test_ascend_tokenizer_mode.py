import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)


class TestEnableTokenizerMode(CustomTestCase):
    """
    Testcase：Verify that the inference is successful when the tokenizer mode is set to slow or auto

    [Test Category] Parameter
    [Test Target] --tokenizer-mode slow/auto
    """

    def test_tokenzier_mode(self):
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        self.base_url = DEFAULT_URL_FOR_TEST
        for tokenizer_mode in ["slow", "auto"]:
            other_args = [
                "--tokenizer-mode",
                tokenizer_mode,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tokenizer-path",
                model_path,
                "--tokenizer-worker-num",
                4,
            ]

            process = popen_launch_server(
                model_path,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

            try:
                response = requests.get(f"{self.base_url}/health_generate")
                self.assertEqual(response.status_code, 200)

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
                self.assertEqual(response.status_code, 200)
                self.assertIn("Paris", response.text)

                response = requests.get(self.base_url + "/get_server_info")
                self.assertEqual(response.status_code, 200)
                print(response.json())
                self.assertEqual(response.json()["tokenizer_path"], model_path)
                self.assertEqual(response.json()["tokenizer_mode"], tokenizer_mode)
            finally:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
