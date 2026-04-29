import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM5_1_W4A8_MODEL_PATH,
    TestAscendPerformancePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=2400,
    suite="nightly-pd-sep-6-node",
    nightly=True,
)

GLM5_1_W4A8_2P1D_PREFILL_ENVS = {
    "ASCEND_MF_STORE_URL": "tcp://61.47.19.68:24707",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "HCCL_SOCKET_IFNAME": "enp196s0f0",
    "GLOO_SOCKET_IFNAME": "enp196s0f0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}

GLM5_1_W4A8_2P1D_DECODE_ENVS = {
    "ASCEND_MF_STORE_URL": "tcp://61.47.19.68:24707",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_SOCKET_IFNAME": "enp196s0f0",
    "GLOO_SOCKET_IFNAME": "enp196s0f0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}

GLM5_1_W4A8_2P1D_PREFILL_ARGS = [
    "--disaggregation-mode", "prefill",
    "--nnodes", 2,
    "--tp-size", 32,
    "--mem-fraction-static", 0.72,
    "--attention-backend", "ascend",
    "--device", "npu",
    "--quantization", "modelslim",
    "--disaggregation-transfer-backend", "ascend",
    "--max-running-requests", 16,
    "--served-model-name", "glm-5",
    "--chunked-prefill-size", 16384,
    "--max-prefill-tokens", 180000,
    "--moe-a2a-backend", "deepep",
    "--deepep-mode", "normal",
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--dtype", "bfloat16",
    "--speculative-draft-model-quantization", "unquant",
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", 1,
    "--speculative-eagle-topk", 1,
    "--speculative-num-draft-tokens", 2,
    "--dp-size", 1,
    "--enable-dp-attention",
    "--load-balance-method", "round_robin",
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode", "in-seq-split",
    "--attn-cp-size", 32,
    "--enable-dp-lm-head",
    "--moe-dense-tp", 1,
    "--trust-remote-code",
]

GLM5_1_W4A8_2P1D_DECODE_ARGS = [
    "--disaggregation-mode", "decode",
    "--nnodes", 2,
    "--tp-size", 32,
    "--dp-size", 32,
    "--enable-dp-attention",
    "--ep-size", 32,
    "--mem-fraction-static", 0.85,
    "--max-running-requests", 96,
    "--attention-backend", "ascend",
    "--device", "npu",
    "--quantization", "modelslim",
    "--served-model-name", "glm-5",
    "--moe-a2a-backend", "deepep",
    "--deepep-mode", "low_latency",
    "--cuda-graph-bs", 1, 2, 3,
    "--disaggregation-transfer-backend", "ascend",
    "--watchdog-timeout", 9000,
    "--context-length", 180000,
    "--tokenizer-worker-num", 4,
    "--prefill-round-robin-balance",
    "--disable-shared-experts-fusion",
    "--dtype", "bfloat16",
    "--load-balance-method", "round_robin",
    "--speculative-draft-model-quantization", "unquant",
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", 3,
    "--speculative-eagle-topk", 1,
    "--speculative-num-draft-tokens", 4,
    "--trust-remote-code",
]

GLM5_1_W4A8_2P1D_MODEL_CONFIG = {
    "model_path": GLM5_1_W4A8_MODEL_PATH,
    "prefill_args": GLM5_1_W4A8_2P1D_PREFILL_ARGS,
    "decode_args": GLM5_1_W4A8_2P1D_DECODE_ARGS,
    "prefill_envs": GLM5_1_W4A8_2P1D_PREFILL_ENVS,
    "decode_envs": GLM5_1_W4A8_2P1D_DECODE_ENVS,
    "router_args": [],
    "router_envs": {},
}


class TestNPUGLM5_1_W4A8_2P1D_96P_In3k5_Out1k5_50ms(TestAscendPerformancePdSepTestCaseBase):
    """Test NPU performance for GLM-5.1-w4a8 2p1d_96p PD separation in3k5 out1k5"""

    model_config = GLM5_1_W4A8_2P1D_MODEL_CONFIG
    max_concurrency = 440
    num_prompts = 1760
    random_input_len = 3500
    random_output_len = 1500
    tpot_threshold_ms = 50.0

    def test_npu_glm5_1_w4a8_2p1d_96p_in3k5_out1k5_50ms(self):
        """Run NPU performance test for GLM-5.1-w4a8 2p1d_96p"""
        self.run_performance()


if __name__ == "__main__":
    unittest.main()