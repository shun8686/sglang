import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_bench_offline_throughput,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)


class TestAscendDeepseekV31(CustomTestCase):
    """
    Testcase：Verify the correctness and performance when kernels for attention layers are chosen

    [Test Category] Parameter
    [Test Target] --attention-backend
    """

    TEST_MODEL_MATRIX = {
        "/home/weights/DeepSeek-R1-0528-w4a8-per-channel": {
            "accuracy": 0.9,
            "latency": 150,
            "output_throughput": 30,
        },
    }
    extra_args = ["--mem-fraction-static", 0.77, ]
    envs = {
        "SGLANG_SCHEDULER_DECREASE_PREFILE_IDLE": "1",
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",

        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "80",
        "HCCL_BUFFSIZE": "1600",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",


    }
    num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.models = cls.TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--tp", 16,
            "--device", "npu",
            "--quantization",
            "modelslim",
            "--watchdog-timeout", 9000,
            "--cuda-graph-bs",
            4,
            8,
            16,
            20,
            "--max-running-requests",
            320,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            -1,
            "--max-prefill-tokens",
            1500,
            "--moe-a2a-backend", "deepep", "--deepep-mode", "auto",
            "--enable-dp-attention",
            "--dp",
            "16",
            "--enable-dp-lm-head",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            3,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            4,
            "--dtype",
            "bfloat16",
        ]

        # basic testcase, reserved for setting environment
        for env in cls.envs.keys():
            os.environ[env] = cls.envs[env]

    @classmethod
    def tearDownClass(cls):
        pass

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1800,
                    other_args=[
                        *self.common_args,
                        *self.extra_args,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        num_shots=self.num_shots,
                        data_path=None,
                        num_questions=1319,
                        max_new_tokens=512,
                        parallel=128,
                        host=f"http://{self.url.hostname}",
                        port=int(self.url.port),
                    )

                    metrics = run_eval_few_shot_gsm8k(args)
                    self.assertGreaterEqual(
                        metrics["accuracy"],
                        self.TEST_MODEL_MATRIX[model]["accuracy"],
                    )
                finally:
                    kill_process_tree(process.pid)

if __name__ == "__main__":
    unittest.main()
