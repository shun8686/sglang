import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

class TestSkipServerWarmup(CustomTestCase):
    """Testcase: Test the L2 cache service will fail to start if L1 cache is not enabled

    [Test Category] --disable-radix-cache; --enable-hierarchical-cache
    [Test Target] The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive and cannot be used at the same time
    """

    def test_L2_cache_04(self):
        error_message="The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive and cannot be used at the same time. Please use only one of them."
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.8,
                "--tp-size",
                2,
                "--enable-hierarchical-cache",
                "--disable-radix-cache",
                ]
        )
        out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        try:
            process = popen_launch_server(
                (
                    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
                ),
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
             )
        except Exception as e:
            print(f"Server launch failed as expectes:{e}")
        finally:
            err_log_file.seek(0)
            content = err_log_file.read()
            self.assertIn(error_message, content)
            out_log_file.close()
            err_log_file.close()
            os.remove("./cache_out_log.txt")
            os.remove("./cache_err_log.txt")


if __name__ == "__main__":
    unittest.main()
