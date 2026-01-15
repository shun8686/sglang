import os
import unittest
from types import SimpleNamespace

from utils.test_ascend_deepep_mode_config import QWEN3_235B_A22B_W8A8_MODEL_PATH, NIC_NAME
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPureTP(CustomTestCase):
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
    ]

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_235B_A22B_W8A8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
                "HCCL_BUFFSIZE": "2100",
                "HCCL_SOCKET_IFNAME": NIC_NAME,
                "GLOO_SOCKET_IFNAME": NIC_NAME,
                "HCCL_OP_EXPANSION_MODE": "AIV",
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
