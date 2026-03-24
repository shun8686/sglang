import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V32_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

MODEL_CONFIG = {
    "model_path": DEEPSEEK_V32_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
        # "SGLANG_NPU_USE_MLAPO": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        # "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        # "SGLANG_NPU_USE_MULTI_STREAM": "1",
        # "SGLANG_NPU_USE_MLAPO": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_BUFFSIZE": "400",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    },
    "router_envs": {
        "SGLANG_DP_ROUND_ROBIN": "1",
    },
    "prefill_args": [
        "--nnodes",
        2,
        "--tp",
        32,
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.73,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        65536,
        "--max-prefill-tokens",
        65536,
        "--max-running-requests",
        16,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-mode",
        "prefill",
        "--disable-cuda-graph",
        # "--enable-nsa-prefill-context-parallel",
        # "--nsa-prefill-cp-mode",
        # "in-seq-split",
        "--moe-dense-tp-size",
        1,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
    ],
    "decode_args": [
        "--nnodes",
        2,
        "--tp",
        32,
        "--dp",
        8,
        "--ep",
        32,
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.79,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        65536,
        "--max-running-requests",
        16,
        "--cuda-graph-max-bs",
        16,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--quantization",
        "modelslim",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        3,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        4,
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-mode",
        "decode",
        "--prefill-round-robin-balance",
        "--load-balance-method",
        ROUND_ROBIN,
    ],
    "router_args": [
        "--mini-lb",
    ],
}


class TestDeepSeekV32(TestAscendPerfMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 8
    num_prompts = 8
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    tpot = 26
    output_token_throughput = 132.48

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
