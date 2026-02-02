import unittest
from types import SimpleNamespace
from urllib.parse import urlparse
import requests
from sglang.srt.utils import kill_process_tree
from test_ascend_graph_tp1_bf16 import TestAscendGraphTp1Bf16
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestAscendMlaW8A8Int8(TestAscendGraphTp1Bf16):
    """
    Testcase：Verify the correctness and performance when quantization model is modelslim and cuda graph mode is
    disabled, and verify that a memory error occurs when --mem-fraction-static is too large or too small.

    [Test Category] Parameter
    [Test Target] --quantization modelslim, --disable-cuda-graph, --mem-fraction-static 0.8(normal)/0.1(too small)/0.9(too large)
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
        "--chunked-prefill-size", 32768,
    ]

    @classmethod
    def tearDownClass(cls):
        pass

    def test_c_mem_fraction_static_too_small(self):
        mem_fraction_static_values = [0.1, 0.9]
        for mem_fraction_static_value in mem_fraction_static_values:
            with self.subTest(mem_fraction_static_value=mem_fraction_static_value):
                print(f"##=== Testing --mem-fraction-static: {mem_fraction_static_value} ===##")
                self.common_args = [
                    "--trust-remote-code",
                    "--disable-cuda-graph",
                    "--mem-fraction-static",
                    mem_fraction_static_value,
                    "--attention-backend",
                    "ascend",
                    "--quantization",
                    "modelslim",
                    "--tp-size",
                    2,
                    "--disable-radix-cache",
                    "--chunked-prefill-size",
                    32768,
                ]

                excepted_message = "Server process exited with code -9. Check server logs for errors."
                exception_message = None

                try:
                    process = popen_launch_server(
                        model,
                        self.base_url,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                        other_args=[
                            *self.common_args,
                        ],
                    )
                    self.fail("##=== Service should have crashed due to OOM===##")
                except Exception as e:
                    print("##=== Service have correctly crashed due to OOM===##")
                    exception_message = str(e)
                finally:
                    self.assertEqual(exception_message, excepted_message)
                    if exception_message is None:
                        kill_process_tree(process.pid)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendMlaW8A8Int8))
    runner = unittest.TextTestRunner()
    runner.run(suite)
