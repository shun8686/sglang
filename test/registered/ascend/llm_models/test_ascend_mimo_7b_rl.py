import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMiMo7BRL(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Accuracy of the XiaomiMiMo/MiMo-7B-RL model was tested using the GSM8K dataset.

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-7B-RL
    """

    model = "/root/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-7B-RL"
    accuracy = 0.75


if __name__ == "__main__":
    unittest.main()
