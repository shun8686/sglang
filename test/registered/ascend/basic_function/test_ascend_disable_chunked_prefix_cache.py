import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDisableChunkCache(CustomTestCase):
    def test_chunk_prefix_cache(self):
        other_args = (
            [
                "--disable-chunked-prefix-cache",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.9",
                "--quantization",
                "modelslim",
                "--tp-size",
                "16",
            ]
        )
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=3600,
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
        self.assertEqual(response.json()["disable_chunked_prefix_cache"], True)
        kill_process_tree(process.pid)


if __name__ == "__main__":

    unittest.main()
