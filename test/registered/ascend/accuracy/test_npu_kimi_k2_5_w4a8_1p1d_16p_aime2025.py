import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import KIMI_K2_5_W4A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-pd-sep-2-node",
    nightly=True,
)

PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1800",
}

DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "800",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--mem-fraction-static",
    0.75,
    "--max-running-requests",
    16,
    "--chunked-prefill-size",
    32768,
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--dtype",
    "bfloat16",
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--mem-fraction-static",
    0.76,
    "--max-running-requests",
    16,
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--cuda-graph-bs",
    4,
    8,
    "--dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
]

MODEL_CONFIG = {
    "model_path": KIMI_K2_5_W4A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUKimiK2_5_W4A8_1P1D_32P_AIME2025(
    TestAscendAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for Kimi-K2.5-w4a8 1p1d_32p on AIME 2025"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    accuracy = 0.8
    dataset_type = "aime2025"
    dataset_name = "aime2025_gen"
    max_concurrency = 64
    output_len = 220000

    def test_npu_kimi_k2_5_w4a8_1p1d_32p_aime2025(self):
        """Run NPU accuracy test for Kimi-K2.5-w4a8 1p1d_32p on AIME 2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
