import unittest

from utils.test_ascend_deepep_mode_config import DEEPSEEK_R1_W8A8_MODEL_PATH, NIC_NAME
from utils.test_ascend_pd_separation_utils import TestAscendPdSepTestCaseBase, launch_server


MODEL_CONFIG = {
    "model_path": DEEPSEEK_R1_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "HCCL_BUFFSIZE": "1536",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "650",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "78",
        "TASK_QUEUE_ENABLE": "1",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "router_envs": {
        "SGLANG_DP_ROUND_ROBIN": "1",
    },
    "prefill_args": [
        "--nnodes", "1",
        "--node-rank", "0",
        "--disaggregation-mode", "prefill",
        "--tp-size", 16,
        "--mem-fraction-static", 0.81,
        "--quantization", "modelslim",
        "--max-running-requests", 8,
        "--context-length", 8192,
        "--disable-radix-cache",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 28680,
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "normal",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 1,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 2,
        "--dp-size", 2,
        "--enable-dp-attention",
        "--disable-shared-experts-fusion",
        "--dtype", "bfloat16",
        "--enable-attn-tp-input-scattered",
    ],
    "decode_args": [
        "--nnodes", "2",
        "--disaggregation-mode", "decode",
        "--tp-size", 32,
        "--dp-size", 32,
        "--mem-fraction-static", 0.815,
        "--max-running-requests", 832,
        "--quantization", "modelslim",
        "--moe-a2a-backend", "ascend_fuseep",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp", "1",
        "--cuda-graph-bs", 12, 14, 16, 18, 20, 22, 24, 26,
        "--watchdog-timeout", 9000,
        "--context-length", 8192,
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 2,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 3,
        "--tokenizer-worker-num", 4,
        "--prefill-round-robin-balance",
        "--disable-shared-experts-fusion",
        "--dtype", "bfloat16",
        "--load-balance-method", "round_robin",
    ],
    "router_args": [
        "--mini-lb",
    ],
}


class TestDeepSeekV32(TestAscendPdSepTestCaseBase):
    model_config = MODEL_CONFIG
    # 0.625
    expect_score = 0.56
    # 0.985
    expect_accuracy = 0.9

    def test_deepseek_r1(self):
        launch_server(self.role, self.model_config)
        self.run_test_mmlu()
        self.run_test_gsm8k()


if __name__ == "__main__":
    unittest.main()
