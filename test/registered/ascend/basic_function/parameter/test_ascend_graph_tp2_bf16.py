import unittest

from sglang.test.ascend.base_test_ascend_gsm8k_and_throughput import BaseTestAscendGsm8kAndThroughput
from sglang.test.ascend.test_ascend_utils import QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-2-npu-a3", nightly=True)


class TestAscendGraphTp2Bf16(BaseTestAscendGsm8kAndThroughput):
    """
    Testcase：Verify the accuracy on gsm8k dataset and throughput of Qwen2.5-7B when cuda graph mode is enabled and
    tp size is 2

    [Test Category] Parameter
    [Test Target] enable cuda graph mode (default setting), --tp-size 2
    """

    TEST_MODEL_MATRIX = {
        QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH: {
            "accuracy": 0.85,
            "latency": 180,
            "output_throughput": 20,
        },
    }

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--tp-size", 2,
    ]


if __name__ == "__main__":
    unittest.main()
