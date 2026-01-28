import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDeepSeek(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the DeepSeek-V3.2-Exp-W8A8 model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] DeepSeek-V3.2-Exp-W8A8
    """

    model = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"
    accuracy = 0.51
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
