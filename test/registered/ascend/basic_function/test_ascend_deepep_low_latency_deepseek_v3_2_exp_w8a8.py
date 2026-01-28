import time
import os
import unittest
from types import SimpleNamespace

from utils.test_ascend_deepep_mode_config import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDeepEpDeepseek(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=6000,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--mem-fraction-static",
                0.82,
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--context-length", 40960,
                "--max-prefill-tokens", 128,
                "--max-total-tokens", 40960,
            ],
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "HCCL_SOCKET_IFNAME": NIC_NAME,
                "GLOO_SOCKET_IFNAME": NIC_NAME,
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
                "HCCL_BUFFSIZE": "2048",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "TASK_QUEUE_ENABLE": "0",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        expect_score = 0.565
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=32,
        )
        print("Starting mmlu test...")
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)

    def test_gsm8k(self):
        expect_accuracy = 0.565
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            timeout=60000,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        print("Starting gsm8k test...")
        metrics = run_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
