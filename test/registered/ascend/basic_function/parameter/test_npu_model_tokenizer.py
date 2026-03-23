import unittest
import os
import shutil

import requests

import tempfile

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestNpuModelTokenizer(CustomTestCase):
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    model_path = tempfile.mkdtemp(prefix="model_path")
    tokenizer_path = tempfile.mkdtemp(prefix="tokenizer_path")
    tokenizer_worker_num = 4
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--model-path",
        model_path,
        "--tokenizer-path",
        tokenizer_path,
        "--tokenizer-worker-num",
        tokenizer_worker_num,
        "--tokenizer-mode",
        "auto",
        "--skip-tokenizer-init",
        "--load-format",
        "safetensors",
    ]
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.base_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        if os.path.exists(cls.model_path):
            shutil.rmtree(cls.model_path)
        if os.path.exists(cls.tokenizer_path):
            shutil.rmtree(cls.tokenizer_path)


    def test_model_tokenizer_sending_request(self):
        """Send a request to the server and count the number of generated tensor dump directories."""
        text1 = "The capital of France is"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text1,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text", response.json())

if __name__ == "__main__":
    unittest.main()


