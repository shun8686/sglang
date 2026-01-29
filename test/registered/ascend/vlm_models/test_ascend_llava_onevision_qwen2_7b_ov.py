import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLAVA_ONEVISION_QWEN2_7B_OV_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestLlavaOneVision(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the lmms-lab/llava-onevision-qwen2-7b-ov model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] lmms-lab/llava-onevision-qwen2-7b-ov
    """

    model = LLAVA_ONEVISION_QWEN2_7B_OV_WEIGHTS_PATH
    accuracy = 0.73
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        2,
        "--max-running-requests",
        2048,
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mm-per-request-timeout",
        60,
        "--enable-multimodal",
    ]


if __name__ == "__main__":
    unittest.main()
