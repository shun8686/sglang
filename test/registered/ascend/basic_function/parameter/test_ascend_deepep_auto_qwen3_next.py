import os
import unittest
from types import SimpleNamespace

from utils.test_ascend_deepep_mode_config import QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH, NIC_NAME
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestQwen3Next(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend", "ascend",
                "--device", "npu",
                "--tp-size", 8,
                "--mem-fraction-static", 0.8,
                "--max-running-requests", 80,
                "--watchdog-timeout", 9000,
                "--disable-radix-cache",
                # "--cuda-graph-bs", 80,
                "--disable-cuda-graph",
                "--max-prefill-tokens", 28672,
                "--max-total-tokens", 450560,
                "--moe-a2a-backend", "deepep",
                "--deepep-mode", "auto",
                "--quantization", "modelslim",
                "--chunked-prefill-size", -1,
            ],
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "HCCL_SOCKET_IFNAME": NIC_NAME,
                "GLOO_SOCKET_IFNAME": NIC_NAME,
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "HCCL_ALGO": "level0:NA;level1:ring",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "20",
                "HCCL_BUFFSIZE": "2000",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        # 0.625
        expect_score = 0.56
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
        # 0.945
        expect_accuracy = 0.9
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
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
