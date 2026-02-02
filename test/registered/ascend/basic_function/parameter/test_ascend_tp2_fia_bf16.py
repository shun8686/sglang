import unittest

from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
register_npu_ci(est_time=250, suite="nightly-2-npu-a3", nightly=True)


class TestAscendTp2FIABf16(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance when kernels for attention layers are chosen, cuda graph mode is
    disabled, tp size is 2 and FIA acceleration is used， and defaults to auto-detection the device to use if not specified

    [Test Category] Parameter
    [Test Target] --attention-backend ascend (set in TestAscendGraphTp1Bf16), --disable-cuda-graph, --tp-size 2,
    ENV ASCEND_USE_FIA=true
    """

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--tp-size", 2,
    ]

    envs = {
        "ASCEND_USE_FIA": "true",
    }


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendTp2FIABf16))
    runner = unittest.TextTestRunner()
    runner.run(suite)
