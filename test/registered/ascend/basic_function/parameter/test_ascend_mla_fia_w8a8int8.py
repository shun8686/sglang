import unittest
from urllib.parse import urlparse

from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestAscendMlaFiaW8A8Int8(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance when quantization model is modelslim and FIA acceleration is used。

    [Test Category] Parameter
    [Test Target] --quantization
    """

    TEST_MODEL_MATRIX = {
        DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH: {
            "accuracy": 0.34,
            "latency": 1000,
            "output_throughput": 6,
        },
    }
    extra_args = [
        "--mem-fraction-static", 0.8,
        "--disable-cuda-graph",
        "--quantization", "modelslim",
        "--tp-size", 2,
        "--disable-radix-cache",
    ]
    envs = {"ASCEND_USE_FIA": "true"}


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendMlaFiaW8A8Int8))
    runner = unittest.TextTestRunner()
    runner.run(suite)
