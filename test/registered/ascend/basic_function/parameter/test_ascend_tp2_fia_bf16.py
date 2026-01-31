import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=250, suite="nightly-2-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH: {
        "accuracy": 0.85,
        "latency": 180,
        "output_throughput": 20,
    },
}



class TestAscendTp2Bf16(CustomTestCase):
    """
    Testcase：Verify the accuracy on gsm8k dataset and throughput of Qwen2.5-7B when graph mode is disabled,
    tp size is 2 and FIA acceleration is used.

    [Test Category] Parameter
    [Test Target] --disable-radix-cache, --tp-size 2, ENV ASCEND_USE_FIA=true
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

    @classmethod
    def setUpClass(cls):
        cls.models = cls.TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
        ]
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--tp-size",
            2,
            "--disable-radix-cache",
        ]
        os.environ["ASCEND_USE_FIA"] = "true"



if __name__ == "__main__":
    unittest.main()
