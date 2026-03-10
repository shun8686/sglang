import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import STARCODER2_7B_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the bigcode/starcoder2-7b model on the GSM8K dataset is no less than 0.3.

    [Test Category] Model
    [Test Target] bigcode/starcoder2-7b
    """

    model = STARCODER2_7B_WEIGHTS_PATH
    accuracy = 0.3


if __name__ == "__main__":
    unittest.main()
