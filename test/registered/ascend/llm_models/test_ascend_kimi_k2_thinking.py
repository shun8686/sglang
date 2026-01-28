import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestKimiK2Thinking(GSM8KAscendMixin, CustomTestCase):
    """Testcase:Test the accuracy of the Kimi/Kimi-K2-Thinking model using the GSM8K dataset.

    [Test Category] Model
    [Test Target] Kimi/Kimi-K2-Thinking
    """

    model = "/root/.cache/modelscope/hub/models/Kimi/Kimi-K2-Thinking"
    accuracy = 0.95
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        16,
        "--max-total-tokens",
        "66000",
        "--context-length",
        8192,
        "--chunked-prefill-size",
        8192,
        "--max-prefill-tokens",
        8000,
        "--max-running-requests",
        16,
    ]


if __name__ == "__main__":
    unittest.main()
