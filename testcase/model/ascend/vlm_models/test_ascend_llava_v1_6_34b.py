import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin

#from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

#register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
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
            "/data/c30044170/code/newHDK/llava-1.6v-34b-tokenizer"
    ]


if __name__ == "__main__":
    unittest.main()
