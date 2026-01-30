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
    Testcase：Verify that inference requests can be successfully processed after the --served-model-name parameter is set.

    [Test Category] Parameter
    [Test Target] --served-model-name module_name
    """

    def test_tokenzier_mode(self):
        self.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        self.base_url = DEFAULT_URL_FOR_TEST
        served_model_name = "Llama3.2"
        other_args = (
            [
                "--served-model-name",
                served_model_name,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else ["--served-model-name", served_model_name]
        )
        process = popen_launch_server(
            self.model_path,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
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
        self.assertEqual(response.json()["served_model_name"], served_model_name)
        kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
