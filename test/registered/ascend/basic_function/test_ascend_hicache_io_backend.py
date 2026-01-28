import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestHicacheIoBackend(CustomTestCase):
    """Testcase: Tests core functionality with --hicache-io-backend configuration

    [Test Category] --hicache-io-backend
    [Test Target] The IO backend for KV cache transfer between CPU and GPU
    """

    def test_hicache_io_backend(self):
        other_args = (
            [
                "--hicache-io-backend",
                "direct",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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
        response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["hicache_io_backend"], "direct")
        kill_process_tree(process.pid)


if __name__ == "__main__":

    unittest.main()
