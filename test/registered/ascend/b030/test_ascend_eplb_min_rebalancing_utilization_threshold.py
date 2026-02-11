import os
import unittest

from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server, CustomTestCase,
)


class TestQwen330B(CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-30B-A3B-Instruct-2507 model on the GSM8K dataset is no less than 0.90.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-30B-A3B-Instruct-2507
    """

    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.86
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "8",
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "1024",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        16,
        "--eplb-rebalance-num-iterations",
        50,
        "--expert-distribution-recorder-buffer-size",
        50,
        "--eplb-min-rebalancing-utilization-threshold",
        0.05,
        "--base-gpu-id",
        "8"
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env={
                "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                "HCCL_BUFFSIZE": "2048",
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
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
