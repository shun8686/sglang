import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLAVA_NEXT_72B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestLlavaNext72B(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the lmms-lab/llava-next-72b model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] lmms-lab/llava-next-72b
    """

    model = LLAVA_NEXT_72B_WEIGHTS_PATH
    accuracy = 0.79
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        16,
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--enable-multimodal",
    ]


if __name__ == "__main__":
    unittest.main()
