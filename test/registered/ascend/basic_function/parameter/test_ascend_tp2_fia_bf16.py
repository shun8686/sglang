import unittest

from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16

from sglang.test.ci.ci_register import register_npu_ci
register_npu_ci(est_time=250, suite="nightly-2-npu-a3", nightly=True)


class TestAscendTp2FIABf16(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance when kernels for attention layers are chosen, cuda graph mode is
    disabled, radix cache is disabled, tp size is 2 and FIA acceleration is used.

    [Test Category] Parameter
    [Test Target] --attention-backend, --disable-cuda-graph, --tp-size, --disable-radix-cache
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
