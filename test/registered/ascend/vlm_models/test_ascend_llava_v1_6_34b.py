import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestLlava(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Accuracy of the AI-ModelScope/llava-v1.6-34b model was tested using the GSM8K dataset.

    [Test Category] Model
    [Test Target] AI-ModelScope/llava-v1.6-34b
    """

    model = "/root/.cache/modelscope/hub/models/AI-ModelScope/llava-v1.6-34b"
    accuracy = 0.63
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        4,
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
        "--tokenizer-path",
        "/root/.cache/modelscope/hub/models/AI-ModelScope/llava-v1.6-34b/llava-1.6v-34b-tokenizer"
    ]


if __name__ == "__main__":
    unittest.main()
