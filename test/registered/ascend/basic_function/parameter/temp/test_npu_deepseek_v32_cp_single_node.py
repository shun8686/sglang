import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH

register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
    disabled="https://github.com/Ascend/sglang/issues/94"
)

ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--tp=8",
    "--attn-cp-size=4",
    "--enable-nsa-prefill-context-parallel",
]

# Accuracy thresholds
GSM8K_BASELINE = 0.935


class TestDeepseekV32CPSingleNode(unittest.TestCase):
    """Test Case: Test DeepSeek V3.2 model with NSA context parallelism,testing context parallel (CP) modes
    combined with DP (data parallel) + MTP (speculative decoding)

    [Test Category] Parameter
    [Test Target] --attn-cp-size
    """

    def test_deepseek_v32_cp_variants(self):
        """Run accuracy tests for DeepSeek V3.2 CP variants."""
        self.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        variants = [
            ModelLaunchSettings(
                self.model,
                tp_size=8,
                extra_args=ARGS,
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
                variant="CP-round-robin-split",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2-Exp CP Single Node",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=GSM8K_BASELINE
            ),
            performance_params=None,
        )


if __name__ == "__main__":
    unittest.main()
