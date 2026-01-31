import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ascend.test_ascend_gsm8k_and_throughput import TestAscendGsm8kAndThroughput
from sglang.test.ascend.test_ascend_utils import QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH: {
        "accuracy": 0.85,
        "latency": 150,
        "output_throughput": 30,
    },
}


class TestAscendGraphTp1Bf16(TestAscendGsm8kAndThroughput):
    """
    Testcase：Verify the accuracy and throughput of Qwen2.5-7B on gsm8k dataset when cuda graph mode is enabled and
    tp size is 1

    [Test Category] Parameter
    [Test Target] enable cuda graph mode (default setting), --tp-size 1 (default setting)
    """
    extra_args = ["--mem-fraction-static", 0.8, ]


if __name__ == "__main__":
    unittest.main()
