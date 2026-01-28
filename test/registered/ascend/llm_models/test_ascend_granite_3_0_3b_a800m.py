import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestGranite(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the ibm-granite/granite-3.0-3b-a800m-instruct model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] ibm-granite/granite-3.0-3b-a800m-instruct
    """

    model = (
        "/root/.cache/modelscope/hub/models/ibm-granite/granite-3.0-3b-a800m-instruct"
    )
    accuracy = 0.38


if __name__ == "__main__":
    unittest.main()
