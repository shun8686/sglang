import unittest

from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=500, suite="nightly-4-npu-a3", nightly=True)


class TestAscendTp4Bf16(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the accuracy on gsm8k dataset and throughput of Qwen3-30B-A3B when cuda graph mode is disabled and
    tp size is 4

    [Test Category] Parameter
    [Test Target] --disable-cuda-graph, --tp-size 4
    """

    TEST_MODEL_MATRIX = {
        QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH: {
            "accuracy": 0.90,
            "latency": 180,
            "output_throughput": 20,
        },
    }

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-cuda-graph",
        "--tp-size", 4,
    ]


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendTp4Bf16))
    runner = unittest.TextTestRunner()
    runner.run(suite)
