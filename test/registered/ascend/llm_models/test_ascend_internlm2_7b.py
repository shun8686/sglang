import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestInternlm2(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the Shanghai_AI_Laboratory/internlm2-7b model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] Shanghai_AI_Laboratory/internlm2-7b
    """

    model = "/root/.cache/modelscope/hub/models/Shanghai_AI_Laboratory/internlm2-7b"
    accuracy = 0.585


if __name__ == "__main__":
    unittest.main()
