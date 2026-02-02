import unittest

from sglang.test.ascend.base_test_ascend_gsm8k_and_throughput import TestAscendGsm8kAndThroughput
from sglang.test.ascend.test_ascend_utils import QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=250, suite="nightly-2-npu-a3", nightly=True)


class TestAscendTp2Bf16(TestAscendGsm8kAndThroughput):
    """
    Testcase：Verify the accuracy on gsm8k dataset and throughput of Qwen2.5-7B when cuda graph mode is disabled and
    tp size is 2

    [Test Category] Parameter
    [Test Target] --disable-cuda-graph, --tp-size 2
    """

    TEST_MODEL_MATRIX = {
        QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH: {
            "accuracy": 0.85,
            "latency": 150,
            "output_throughput": 30,
        },
    }

    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-cuda-graph",
        "--tp-size", 2,
    ]


if __name__ == "__main__":
    unittest.main()
