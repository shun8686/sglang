import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_5_27B_W8A8_HOME_MODEL_PATH,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
)

QWEN3_5_27B_3K5_1K5_LOW_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
}

QWEN3_5_27B_3K5_1K5_LOW_OTHER_ARGS = [
    "--model-path",
    QWEN3_5_27B_W8A8_HOME_MODEL_PATH,
    "--tp-size",
    4,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    186000,
    "--enable-prefill-delayer",
    "--prefill-delayer-max-delay-passes",
    200,
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.94,
    "--max-total-tokens",
    700000,
    "--max-running-requests",
    38,
    "--max-mamba-cache-size",
    200,
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    12,
    18,
    24,
    32,
    34,
    36,
    38,
    "--trust-remote-code",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUQwen3_5_27B_2P_In3k5_Out1k5_Low_AIME2025(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.5-27B-W8A8 2p in3k5 out1k5 low latency on AIME2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_5_27B_W8A8_HOME_MODEL_PATH
    other_args = QWEN3_5_27B_3K5_1K5_LOW_OTHER_ARGS
    envs = QWEN3_5_27B_3K5_1K5_LOW_ENVS
    accuracy = 0.1
    dataset_type = "aime2025"
    dataset_name = "aime2025_gen_0_shot_cot"
    output_len = 8192
    max_concurrency = 1
    num_prompts = 100000

    def test_npu_qwen3_5_27b_2p_in3k5_out1k5_low_aime2025(self):
        """Run NPU accuracy test for Qwen3.5-27B-W8A8 in3k5 out1k5 low on AIME2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()