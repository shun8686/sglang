import os
import unittest
from time import sleep
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
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
        cls.model = "/home/weights/DeepSeek-V2-Lite-W8A8"
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
                "--max-running-requests",
                128,
                "--disable-cuda-graph",
                "--dp-size",
                8,
                "--enable-dp-attention",
                "--chunked-prefill-size",
                1024,
                "--mem-fraction-static",
                0.68,
                "--base-gpu-id",
                6,
                # "--log-requests",
            ],
            env={
                "SGLANG_SET_CPU_AFFINITY": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "HCCL_BUFFSIZE": "4096",
                # "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                # "HCCL_BUFFSIZE": "2048",
                # "MOE_ENABLE_TOPK_NEG_ONE": "1",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        # expect_score = 0.58
        expect_score = 0.5
        # sleep(1200)

        print("=" * 20 + " 5 num shot START" + "=" * 20)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=3000,
            num_threads=32,
            num_shots=5,
            api="completion",
        )
        print("Starting mmlu test...")
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)
        print("=" * 20 + " 5 num shot END" + "=" * 20)

        # print("=" * 20 + " 8 num shot START" + "=" * 20)
        # args = SimpleNamespace(
        #     base_url=self.base_url,
        #     model=self.model,
        #     eval_name="mmlu",
        #     num_examples=128,
        #     num_threads=32,
        #     num_shots=8,
        #     api="completion"
        # )
        # print("Starting mmlu test...")
        # metrics = run_eval(args)
        # self.assertGreater(metrics["score"], expect_score)
        # print("=" * 20 + " 8 num shot END" + "=" * 20)

    # def test_gsm8k(self):
    #     expect_accuracy = 0.34
    #     args = SimpleNamespace(
    #         num_shots=8,
    #         data_path=None,
    #         num_questions=200,
    #         max_new_tokens=512,
    #         parallel=128,
    #         host="http://127.0.0.1",
    #         port=int(self.base_url.split(":")[-1]),
    #         api="completion",
    #     )
    #     print("Starting gsm8k test...")
    #     metrics = run_gsm8k(args)
    #     self.assertGreaterEqual(
    #         metrics["accuracy"],
    #         expect_accuracy,
    #         f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
    #     )


if __name__ == "__main__":
    unittest.main()
