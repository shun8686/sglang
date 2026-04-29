import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    AIME2025_DATASET,
    QWEN3_5_397B_W4A8_MODEL_PATH,
    TestAscendAccuracyPdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-pd-sep-2-node",
    nightly=True,
)

QWEN3_5_397B_W4A8_1P1D_16P_PREFILL_ENVS = {
    "ASCEND_USE_FIA": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "3000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}

QWEN3_5_397B_W4A8_1P1D_16P_DECODE_ENVS = {
    "ASCEND_USE_FIA": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "2400",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}

QWEN3_5_397B_W4A8_1P1D_16P_PREFILL_ARGS = [
    "--disaggregation-mode", "prefill",
    "--nnodes", 1,
    "--tp-size", 16,
    "--mem-fraction-static", 0.62,
    "--attention-backend", "ascend",
    "--device", "npu",
    "--enable-multimodal",
    "--quantization", "modelslim",
    "--disaggregation-transfer-backend", "ascend",
    "--max-running-requests", 40,
    "--chunked-prefill-size", -1,
    "--max-prefill-tokens", 128000,
    "--moe-a2a-backend", "deepep",
    "--deepep-mode", "normal",
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", 3,
    "--speculative-eagle-topk", 1,
    "--speculative-num-draft-tokens", 4,
    "--speculative-draft-model-quantization", "unquant",
    "--dp-size", 2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--dtype", "bfloat16",
    "--mamba-ssm-dtype", "bfloat16",
    "--disable-cuda-graph",
    "--disable-overlap-schedule",
    "--tokenizer-worker-num", 4,
    "--mamba-scheduler-strategy", "extra_buffer",
    "--mm-enable-dp-encoder",
    "--max-mamba-cache-size", 160,
]

QWEN3_5_397B_W4A8_1P1D_16P_DECODE_ARGS = [
    "--disaggregation-mode", "decode",
    "--nnodes", 1,
    "--tp-size", 16,
    "--ep-size", 16,
    "--mem-fraction-static", 0.75,
    "--max-running-requests", 192,
    "--attention-backend", "ascend",
    "--device", "npu",
    "--enable-multimodal",
    "--quantization", "modelslim",
    "--moe-a2a-backend", "deepep",
    "--enable-dp-attention",
    "--deepep-mode", "low_latency",
    "--enable-dp-lm-head",
    "--dp-size", 4,
    "--cuda-graph-bs", 8, 16, 24, 32, 40, 48,
    "--disaggregation-transfer-backend", "ascend",
    "--watchdog-timeout", 9000,
    "--speculative-algorithm", "NEXTN",
    "--speculative-draft-model-quantization", "unquant",
    "--speculative-num-steps", 3,
    "--speculative-eagle-topk", 1,
    "--speculative-num-draft-tokens", 4,
    "--tokenizer-worker-num", 4,
    "--dtype", "bfloat16",
    "--mamba-ssm-dtype", "bfloat16",
    "--load-balance-method", "round_robin",
    "--disable-radix-cache",
]

QWEN3_5_397B_W4A8_1P1D_16P_MODEL_CONFIG = {
    "model_path": QWEN3_5_397B_W4A8_MODEL_PATH,
    "prefill_args": QWEN3_5_397B_W4A8_1P1D_16P_PREFILL_ARGS,
    "decode_args": QWEN3_5_397B_W4A8_1P1D_16P_DECODE_ARGS,
    "prefill_envs": QWEN3_5_397B_W4A8_1P1D_16P_PREFILL_ENVS,
    "decode_envs": QWEN3_5_397B_W4A8_1P1D_16P_DECODE_ENVS,
    "router_args": [],
    "router_envs": {},
}


class TestNPUQwen3_5_397B_W4A8_1P1D_16P_AIME2025(TestAscendAccuracyPdSepTestCaseBase):
    """Test NPU accuracy for Qwen3.5-397B-w4a8 1p1d_16p PD separation AIME2025"""

    model_config = QWEN3_5_397B_W4A8_1P1D_16P_MODEL_CONFIG
    dataset = AIME2025_DATASET
    max_concurrency = 32
    num_prompts = 30
    max_tokens = 8192

    def test_npu_qwen3_5_397b_w4a8_1p1d_16p_aime2025(self):
        """Run NPU accuracy test for Qwen3.5-397B-w4a8 1p1d_16p AIME2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()