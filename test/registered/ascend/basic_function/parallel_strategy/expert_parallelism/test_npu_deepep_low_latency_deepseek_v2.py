import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.run_eval import run_eval
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestDeepEpDeepseek(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
        cls.model = "/home/weights/DeepSeek-V2-Lite-w8a8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--quantization",
                "modelslim",
                "--tp-size",
                "8",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--disable-cuda-graph",
                "--dp-size",
                8,
                "--enable-dp-attention",
                "--chunked-prefill-size",
                1024,
                "--mem-fraction-static",
                0.7,
            ],
            env={
                "SGLANG_SET_CPU_AFFINITY": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "HCCL_BUFFSIZE": "4096",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        expect_score = 0.5
        # expect_score = 0.58
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=512,
            num_threads=32,
            num_shots=5,
            api="completion",
        )
        print("Starting mmlu test...")
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)

    def test_gsm8k(self):
        expect_accuracy = 0.34
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            data_path=None,
            num_examples=200,
            num_threads=128,
            num_shots=5,
            max_new_tokens=512,
            api="completion"
        )
        print("Starting gsm8k test...")
        metrics = run_eval(args)
        # Assertion: The GSM8K accuracy is not lower than the preset threshold (0.96)
        self.assertGreaterEqual(
            metrics["score"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["score"])}, is lower than {expect_accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
