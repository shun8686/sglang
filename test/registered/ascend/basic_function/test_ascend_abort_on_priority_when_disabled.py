import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestDisableCudaGraphPadding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--abort-on-priority-when-disabled",
            ]
        )

        cls.process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_disable_cuda_graph_padding(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "priority": 2,
                },
            },
        )
        print(response.text)
        self.assertEqual(
            response.status_code, 500, "The request status code is not 500."
        )
      
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(response.json())
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertTrue(
            response.json()["abort_on_priority_when_disabled"],
            "abort_on_priority_when_disabled is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
