import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestQWQ32BW8A8(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the vllm-ascend/QWQ-32B-W8A8 model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] vllm-ascend/QWQ-32B-W8A8
    """

    model = "/root/.cache/modelscope/hub/models/vllm-ascend/QWQ-32B-W8A8"
    accuracy = 0.59
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
        "--quantization",
        "modelslim",
    ]


if __name__ == "__main__":
    unittest.main()
