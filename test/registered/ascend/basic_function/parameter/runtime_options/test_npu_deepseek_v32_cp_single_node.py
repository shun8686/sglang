import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings
from test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

DEEPSEEK_V32_EXP_MODEL_PATH = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH

BASE_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 64}',
]

DP_ARGS = [
    "--tp=8",
    "--dp=2",
    "--attn-cp-size=4",
    "--enable-dp-attention",
]

MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--mem-frac=0.7",
    "--cuda-graph-max-bs=32",
    "--max-running-requests=32",
]

# Accuracy thresholds
GSM8K_BASELINE = 0.935

# CP mode arguments
CP_IN_SEQ_SPLIT_ARGS = [
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode=in-seq-split",
]

CP_ROUND_ROBIN_ARGS = [
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode=round-robin-split",
    "--attn-cp-size=8",
]


class TestDeepseekV32CPSingleNode(unittest.TestCase):
    """Test Case: 针对搭载NSA上下文并行技术的DeepSeek V3.2模型，测试结合DP（数据并行）+MTP（推测执行）的上下文并行（CP）模式：
    - in-seq-split：序列内拆分式上下文并行模式
    - round-robin-split：轮询拆分式上下文并行模式

    [Test Category] Parameter
    [Test Target] --attn-cp-size
    """

    def test_deepseek_v32_cp_variants(self):
        """Run accuracy tests for DeepSeek V3.2 CP variants."""
        variants = [
            # Variant: in-seq-split CP mode with DP+MTP
            ModelLaunchSettings(
                DEEPSEEK_V32_EXP_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + MTP_ARGS + CP_IN_SEQ_SPLIT_ARGS,
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
                variant="CP-in-seq-split",
            ),
            # Variant: round-robin-split CP mode (TP only, no DP)
            ModelLaunchSettings(
                DEEPSEEK_V32_EXP_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + MTP_ARGS + CP_ROUND_ROBIN_ARGS,
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
