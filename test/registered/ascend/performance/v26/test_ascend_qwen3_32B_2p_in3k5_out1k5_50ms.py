import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_4_6_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdMixTestCaseBase,
)
# QWEN3_32B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B")


MODEL_CONFIG = {
    # "model_path": GLM_4_6_W8A8_MODEL_PATH,
    "model_path": "Qwen/Qwen3-32B",
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        # "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "1800",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        # "ASCEND_RT_VISIBLE_DEVICES":[10,11,12,13],
    },
    "other_args": [
        "--trust-remote-code",
        "--nnodes",
        1,
        "--node-rank",
        0,
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--max-running-requests",
        101,
        "--disable-radix-cache",
        "--speculative-draft-model-quantization",
        "unquant",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        35000,
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "/home/weights/Eagle3-Qwen3-32B-zh",
        "--speculative-num-steps",
        3,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        4,
        "--tp-size",
        4,
        "--mem-fraction-static",
        0.845,
        "--cuda-graph-bs",
        16,
        32,
        64,
        72,
        88,
        90,
        92,
        94,
        96,
        97,
        98,
        99,
        100,
        101,
        "--dtype",
        "bfloat16",
    ],
}


class TestQwen332B(TestAscendPerfMultiNodePdMixTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 256
    num_prompts = int(max_concurrency) * 256
    input_len = 3200
    output_len = 1000
    random_range_ratio = 1
    tpot = 50
    # T: None   800I: xxxxx.     dev：3192/16@51.19ms
    output_token_throughput = 3192

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
