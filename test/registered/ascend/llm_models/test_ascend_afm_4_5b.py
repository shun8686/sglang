import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Accuracy of the arcee-ai/AFM-4.5B-Base model was tested using the GSM8K dataset.

    [Test Category] Model
    [Test Target] arcee-ai/AFM-4.5B-Base
    """

    model = "/root/.cache/modelscope/hub/models/arcee-ai/AFM-4.5B-Base"
    accuracy = 0.375


if __name__ == "__main__":
    unittest.main()
