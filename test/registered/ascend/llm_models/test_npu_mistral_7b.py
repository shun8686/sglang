import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the mistralai/Mistral-7B-Instruct-v0.2 model on the GSM8K dataset is no less than 0.36.

    [Test Category] Model
    [Test Target] mistralai/Mistral-7B-Instruct-v0.2
    """

    model = MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH
    accuracy = 0.36


class TestMistral7BChatTemplate(TestMistral7B):
    other_args = GSM8KAscendMixin.other_args + ["--chat-template", "mistral"]


if __name__ == "__main__":
    unittest.main()
