import unittest

from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=150, suite="nightly-1-npu-a3", nightly=True)

class TestAscendTp1Bf16(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance when kernels for attention layers are chosen and cuda graph mode is disabled

    [Test Category] Parameter
    [Test Target] --disable-cuda-graph
    """

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-cuda-graph",
    ]

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendTp1Bf16))
    runner = unittest.TextTestRunner()
    runner.run(suite)
