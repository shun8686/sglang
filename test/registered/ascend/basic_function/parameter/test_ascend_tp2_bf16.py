import unittest

from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=250, suite="nightly-2-npu-a3", nightly=True)


class TestAscendTp2Bf16(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance when kernels for attention layers are chosen, cuda graph mode is
    disabled, and tp size is 2, and defaults to auto-detection the device to use if not specified

    [Test Category] Parameter
    [Test Target] --device (auto-detection if not specified), --attention-backend ascend (set in TestAscendGraphTp1Bf16),
    --disable-cuda-graph, --tp-size 2
    """

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-cuda-graph",
        "--tp-size", 2,
    ]


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendTp2Bf16))
    runner = unittest.TextTestRunner()
    runner.run(suite)
