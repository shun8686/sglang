import os
import unittest
import logging

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=150, suite="nightly-1-npu-a3", nightly=True)


class TestNPUKVCacheDtype(CustomTestCase):
    """Testcase：Verify set --kv_cache_dtype is auto, bf16 or bfloat16, request inference successful.

    [Test Category] Parameter
    [Test Target] --kv_cache_dtype
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    kv_cache_dtype = "auto"
    using_kv_cache_dtype = "torch.bfloat16"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--dtype",
            "auto",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--kv-cache-dtype",
            cls.kv_cache_dtype,
        ]

        cls.old_stdout = os.dup(1)
        cls.old_stderr = os.dup(2)
        cls.pipe_out, cls.pipe_in = os.pipe()
        cls.pipe_err_out, cls.pipe_err_in = os.pipe()
        os.dup2(cls.pipe_in, 1)
        os.dup2(cls.pipe_err_in, 2)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_dtype_options(self):
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

        response = requests.get(
            f"{self.base_url}/server_info",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'"kv_cache_dtype":"{self.kv_cache_dtype}"', response.text)

        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)
        os.close(self.pipe_in)
        os.close(self.pipe_err_in)

        output = os.read(self.pipe_out, 1024 * 1024).decode("utf-8")
        error = os.read(self.pipe_err_out, 1024 * 1024).decode("utf-8")
        os.close(self.pipe_out)
        os.close(self.pipe_err_out)
        logger = logging.getLogger()
        logger.info(output)
        logger.info(error)
        self.assertIn(f"Using KV cache dtype: {self.using_kv_cache_dtype}", error)


class TestNPUKVCacheDtypeBf16(TestNPUKVCacheDtype):
    kv_cache_dtype = "bf16"


class TestNPUKVCacheDtypeBfloat16(TestNPUKVCacheDtype):
    kv_cache_dtype = "bfloat16"


if __name__ == "__main__":
    unittest.main()
