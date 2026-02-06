import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAscendW4A8Quantization(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--tp-size", 16,
                "--trust-remote-code",
                "--attention-backend", "ascend",
                "--quantization", "modelslim",
                "--watchdog-timeout", "9000",
                "--cuda-graph-bs", 8, 16, 24, 28, 32, 36,
                "--mem-fraction-static", 0.71,
                "--max-running-requests", 144,
                "--context-length", 8188,
                "--disable-radix-cache",
                "--disable-shared-experts-fusion",
                "--chunked-prefill-size", -1,
                "--max-prefill-tokens", 9000,
                "--moe-a2a-backend", "ascend_fuseep",
                # "--moe-a2a-backend", "deepep",
                # "--deepep-mode", "auto",
                "--enable-dp-attention",
                "--dp-size", 4,
                "--enable-dp-lm-head",
                "--dtype", "bfloat16",
            ],
            env={
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "2",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
                "HCCL_BUFFSIZE": "1600",
                "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
                "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
                # "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
                "SGLANG_NPU_USE_MLAPO": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_USE_FIA_NZ": "1",
                "ENABLE_MOE_NZ": "1",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.86)

    def test_gsm8k(self):
        expect_accuracy = 0.94
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_gsm8k(args)
        achieved_accuracy = metrics["accuracy"]
        self.assertGreaterEqual(
            achieved_accuracy,
            expect_accuracy,
            f"Accuracy of {self.model} is {str(achieved_accuracy)}, is lower than {expect_accuracy}",
        )
        print(f"Model {self.model} achieved accuracy: {str(achieved_accuracy)}")


if __name__ == "__main__":
    unittest.main()
