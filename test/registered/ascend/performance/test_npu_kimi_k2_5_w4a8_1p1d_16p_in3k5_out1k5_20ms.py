import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_5_W4A8_MODEL_PATH,
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-pd-sep-2-node",
    nightly=True,
)

KIMI_K2_5_W4A8_PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1800",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

KIMI_K2_5_W4A8_DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1800",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

KIMI_K2_5_W4A8_ROUTER_ENVS = {}

KIMI_K2_5_W4A8_PREFILL_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    160,
    "--disable-radix-cache",
    "--model-path",
    KIMI_K2_5_W4A8_MODEL_PATH,
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--host",
    "0.0.0.0",
    "--port",
    8100,
    "--tp-size",
    16,
    "--base-gpu-id",
    0,
    "--mem-fraction-static",
    0.765,
    "--chunked-prefill-size",
    49152,
    "--context-length",
    8192,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    6,
    8,
    10,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
    "--speculative-draft-model-quantization",
    "unquant",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
]

KIMI_K2_5_W4A8_DECODE_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    160,
    "--disable-radix-cache",
    "--model-path",
    KIMI_K2_5_W4A8_MODEL_PATH,
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--host",
    "0.0.0.0",
    "--port",
    8100,
    "--tp-size",
    16,
    "--base-gpu-id",
    0,
    "--mem-fraction-static",
    0.765,
    "--chunked-prefill-size",
    49152,
    "--context-length",
    8192,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    6,
    8,
    10,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
    "--speculative-draft-model-quantization",
    "unquant",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
]

KIMI_K2_5_W4A8_ROUTER_ARGS = [
    "--policy",
    "cache_aware",
]


class TestNPUKimiK2_5_W4A8_1P1D_16P_In3k5_Out1k5_20ms(TestAscendPerfMultiNodePdSepTestCaseBase):
    """Test NPU performance for Kimi-K2.5-w4a8 1P+1D 16p: input_len=3500, output_len=1500, TPOT=20ms"""

    model_config = {
        "model_path": KIMI_K2_5_W4A8_MODEL_PATH,
        "prefill_args": KIMI_K2_5_W4A8_PREFILL_ARGS,
        "decode_args": KIMI_K2_5_W4A8_DECODE_ARGS,
        "prefill_envs": KIMI_K2_5_W4A8_PREFILL_ENVS,
        "decode_envs": KIMI_K2_5_W4A8_DECODE_ENVS,
        "router_envs": KIMI_K2_5_W4A8_ROUTER_ENVS,
        "router_args": KIMI_K2_5_W4A8_ROUTER_ARGS,
    }

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 160
    num_prompts = 160
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 20

    def test_npu_kimi_k2_5_w4a8_1p1d_16p_in3k5_out1k5_20ms(self):
        """Run NPU performance test for 1P+1D 16p with 3.5k input, 1.5k output, TPOT=20ms"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()