import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/lmms-lab/llava-onevision-qwen2-7b-ov"
    accuracy = 0.73
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        2,
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
    ]


if __name__ == "__main__":
    unittest.main()
