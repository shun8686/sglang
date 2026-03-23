import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GPT_OSS_120B_bf16_DRAFT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True, disabled="https://github.com/sgl-project/sglang/pull/18032")


class TestNpuSwa(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the SWA model maintains accurate precision on the GSM8K dataset when hybrid KVCache is enabled.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    """

    model = GPT_OSS_120B_bf16_DRAFT_WEIGHTS_PATH
    accuracy = 0.852
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
        "--swa-full-tokens-ratio",
        "0.8",
    ]

class TestNpuDisableHybridSwaMemory(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the SWA model maintains accurate precision on the GSM8K dataset when hybrid KVCache is disabled.

    [Test Category] Parameter
    [Test Target] --disable-hybrid-swa-memory
    """

    model = GPT_OSS_120B_bf16_DRAFT_WEIGHTS_PATH
    accuracy = 0.852
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
        "--swa-full-tokens-ratio",
        "0.8",
        "--disable-hybrid-swa-memory",
    ]



if __name__ == "__main__":
    unittest.main()
