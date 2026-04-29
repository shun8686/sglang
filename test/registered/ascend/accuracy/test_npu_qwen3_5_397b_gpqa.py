import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_397B_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
)

QWEN3_5_397B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "3000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "6",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

QWEN3_5_397B_OTHER_ARGS = [
    "--skip-server-warmup",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    17500,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    448,
    "--mem-fraction-static",
    0.8,
    "--max-total-tokens",
    320000,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    36,
    48,
    52,
    54,
    56,
    "--quantization",
    "modelslim",
    "--enable-multimodal",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestNPUQwen3_5_397B_GPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.5-397B-w4a8 on GPQA"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_5_397B_W4A8_MODEL_PATH
    other_args = QWEN3_5_397B_OTHER_ARGS
    envs = QWEN3_5_397B_ENVS
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    batch_size = 128
    max_out_len = 1024

    def test_npu_qwen3_5_397b_gpqa(self):
        """Run NPU accuracy test for Qwen3.5-397B on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
