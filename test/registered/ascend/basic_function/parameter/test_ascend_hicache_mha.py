import unittest

from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)


class TestAscendHicacheMha(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance of the hierarchical cache in multi-head attention operator

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache, --hicache-ratio
    """

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--enable-hierarchical-cache",
        "--hicache-ratio", 1.2,
    ]


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendHicacheMha))
    runner = unittest.TextTestRunner()
    runner.run(suite)
