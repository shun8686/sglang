import unittest

from utils.test_ascend_deepep_mode_config import DEEPSEEK_V32_W8A8_MODEL_PATH, NIC_NAME
from utils.test_ascend_pd_separation_utils import TestAscendPdSepTestCaseBase, launch_server

MODEL_CONFIG = {
    "model_path": DEEPSEEK_V32_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
        "SGLANG_NPU_USE_MLAPO": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_BUFFSIZE": "400",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",
    },
    "router_envs": {
        "SGLANG_DP_ROUND_ROBIN": "1",
    },
    "prefill_args": [
        "--nnodes", "2",
        "--disaggregation-mode", "prefill",
        "--tp", 32,
        "--watchdog-timeout", 9000,
        "--mem-fraction-static", 0.73,
        "--disable-radix-cache",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 68000,
        "--max-running-requests", 32,
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "normal",
        "--quantization", "modelslim",
        "--disable-cuda-graph",
        "--enable-nsa-prefill-context-parallel",
        "--moe-dense-tp-size", 1,
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 1,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 2,
    ],
    "decode_args": [
        "--nnodes", "2",
        "--disaggregation-mode", "decode",
        "--tp", 32,
        "--dp", 8,
        "--ep", 32,
        "--moe-dense-tp-size", 1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout", 9000,
        "--mem-fraction-static", 0.79,
        "--disable-radix-cache",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 68000,
        "--max-running-requests", 32,
        "--cuda-graph-max-bs", 30,
        "--moe-a2a-backend", "ascend_fuseep",
        "--quantization", "modelslim",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--prefill-round-robin-balance",
        "--load-balance-method", "round_robin",
    ],
    "router_args": [
        "--mini-lb",
    ],
}


class TestDeepSeekV32(TestAscendPdSepTestCaseBase):
    model_config = MODEL_CONFIG
    # 0.875
    expect_score = 0.8
    # 0.98
    expect_accuracy = 0.9

    def test_deepseek_v3_2(self):
        launch_server(self.role, self.model_config)
        self.run_test_mmlu()
        self.run_test_gsm8k()


if __name__ == "__main__":
    unittest.main()
