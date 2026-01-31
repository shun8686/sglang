import unittest

import TestAscendGraphTp1Bf16
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=150, suite="nightly-1-npu-a3", nightly=True)

class TestAscendTp1Bf16(TestAscendGsm8kAndThroughput):
    """
    Testcase：Verify the accuracy and throughput of Qwen2.5-7B on gsm8k dataset when cuda graph mode is disabled and
    tp size is 1

    [Test Category] Parameter
    [Test Target] --disable-cuda-graph, --tp-size 1 (default setting)
    """

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-cuda-graph",
    ]

if __name__ == "__main__":
    unittest.main()
