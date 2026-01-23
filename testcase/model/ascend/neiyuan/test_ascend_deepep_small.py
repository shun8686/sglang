import unittest
import os
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTBO(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=60000,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "16",
                # "--enable-dp-attention",
                # "--dp",
                # "4",
                "--moe-dense-tp-size",
                "1",
                "--moe-a2a-backend",
                "deepep",
                # "--enable-two-batch-overlap",
                # "--cuda-graph-max-bs",
                # "128",
                "--max-running-requests",
                "512",
                "--attention-backend",
                "ascend",
                # "--mem-fraction-static",
                # 0.7,
                "--quantization",
                "modelslim",
                "--disable-cuda-graph",
            ],
            env={
                    "HCCL_BUFFSIZE": "1000",
                    **os.environ,
                },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
