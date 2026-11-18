import unittest

from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-perf-4-npu-a5",
    nightly=True,
    disabled="TEMP: disabled during overlap-schedule validation",
)

CASE_ID = "test_npu_deepseek_v4_flash_fp8_1p1d_16p_in16k_out1k_a5_overlap"

_PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "2",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "MF_HYBM_USE_VMM_SEGMENT": "1",
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "ASCEND_MF_LOG_LEVEL": "2",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "TRANSFORMERS_VERBOSITY": "error",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
}

_DECODE_ENVS = {
    **_PREFILL_ENVS,
    "TASK_QUEUE_ENABLE": "0",
    "DEEPEP_HCCL_BUFFSIZE": "1024",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--tp-size",
    8,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-cuda-graph",
    "--quantization",
    "fp8",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--mem-fraction-static",
    0.81,
    "--max-running-requests",
    32,
    "--chunked-prefill-size",
    24576,
    "--max-prefill-tokens",
    24576,
    "--tokenizer-worker-num",
    16,
    "--watchdog-timeout",
    9000,
    "--enable-metrics",
]

_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--tp-size",
    8,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--fuseep-mode",
    1,
    "--quantization",
    "fp8",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--mem-fraction-static",
    0.85,
    "--max-running-requests",
    64,
    "--max-prefill-tokens",
    2048000,
    "--chunked-prefill-size",
    16384,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
    10,
    12,
    14,
    16,
    "--tokenizer-worker-num",
    8,
    "--watchdog-timeout",
    9000,
    "--enable-metrics",
]

MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    "prefill_args": _PREFILL_ARGS,
    "decode_args": _DECODE_ARGS,
    "prefill_envs": _PREFILL_ENVS,
    "decode_envs": _DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}


class TestNPUDeepSeekV4FlashFP8_1P1D_16PIn16kOut1kA5Overlap(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    input_len = 16000
    output_len = 1000
    num_prompts = 256
    max_concurrency = 256
    random_range_ratio = 1
    request_rate = float("inf")
    seed = 1
    # tpot/output_token_throughput baselines commented out; the +/-2% comparison
    # against pass 1 is the gate.

    def test_npu_deepseek_v4_flash_fp8_1p1d_4p_in16k_out1k_a5_overlap(self):
        """Run pass-2 throughput and assert no regression vs pass 1."""
        self.run_throughput()
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
