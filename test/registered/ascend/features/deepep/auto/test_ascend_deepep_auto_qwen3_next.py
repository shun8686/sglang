import os
import unittest
from types import SimpleNamespace

from registered.ascend.features.deepep.test_ascend_deepep_mode_config import QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPureTP(CustomTestCase):
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
                "--tp-size",
                "2",
                "--quantization",
                "modelslim",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--disable-cuda-graph",
            ],
            env={
                "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                "HCCL_BUFFSIZE": "2048",
                "MOE_ENABLE_TOPK_NEG_ONE": "1",
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
            num_examples=8,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)



if __name__ == "__main__":
    unittest.main()
