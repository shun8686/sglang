import os
import unittest
from abc import ABC

from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server, CustomTestCase,
)

SKIP_OUT_LOG = "./skip_out_log.txt"
SKIP_ERR_LOG = "./skip_err_log.txt"
REBALANCE_OUT_LOG = "./rebalance_out_log.txt"
REBALANCE_ERR_LOG = "./rebalance_err_log.txt"


class TestEplbMinRebalancingUtilizationThresholdBase(ABC):
    model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
    eplb_min_rebalancing_utilization_threshold = 0.05
    accuracy = 0.86
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--chunked-prefill-size",
        "1024",
        "--tp-size",
        "8",
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        16,
        "--eplb-rebalance-num-iterations",
        50,
        "--expert-distribution-recorder-buffer-size",
        50,
        "--enable-expert-distribution-metrics",
        "--eplb-min-rebalancing-utilization-threshold",
        eplb_min_rebalancing_utilization_threshold,
    ]
    out_file = None
    err_file = None
    log_info = ""

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
                "HCCL_BUFFSIZE": "1024",
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                **os.environ,
            },
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if hasattr(cls, 'out_file') and cls.out_file:
            cls.out_file.close()
        if hasattr(cls, 'err_file') and cls.err_file:
            cls.err_file.close()

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

    def test_eplb_min_rebalancing_utilization_threshold(self):
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertIn(self.log_info, content)


class TestEplbMinRebalancingUtilizationThreshold005(TestEplbMinRebalancingUtilizationThresholdBase, CustomTestCase):
    log_info = "Skipped ep rebalancing: current GPU utilization"
    out_file = open(SKIP_OUT_LOG, "w+", encoding="utf-8")
    err_file = open(SKIP_ERR_LOG, "w+", encoding="utf-8")


class TestEplbMinRebalancingUtilizationThreshold095(TestEplbMinRebalancingUtilizationThresholdBase, CustomTestCase):
    log_info = "rebalance end"
    out_file = open(REBALANCE_OUT_LOG, "w+", encoding="utf-8")
    err_file = open(REBALANCE_ERR_LOG, "w+", encoding="utf-8")
    eplb_min_rebalancing_utilization_threshold = 0.95


if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(TestEplbMinRebalancingUtilizationThreshold005("test_gsm8k"))
    # suite.addTest(TestEplbMinRebalancingUtilizationThreshold005("test_eplb_min_rebalancing_utilization_threshold"))
    suite.addTest(TestEplbMinRebalancingUtilizationThreshold095("test_gsm8k"))
    suite.addTest(TestEplbMinRebalancingUtilizationThreshold095("test_eplb_min_rebalancing_utilization_threshold"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
