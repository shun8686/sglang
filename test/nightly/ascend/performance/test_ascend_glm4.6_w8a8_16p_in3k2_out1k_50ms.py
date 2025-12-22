import unittest

from test_ascend_multi_mix_utils import TestMultiMixUtils
from test_ascend_single_mix_utils import NIC_NAME

MODEL_PATH = "/root/.cache/modelscope/hub/models/GLM-4.6-w8a8_WITH_MTP"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
        "HCCL_BUFFSIZE": "1536",
        "SGLANG_USE_FIA_NZ": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_USE_NZ_MATMUL": "1",
    },
    "other_args": [
        "--trust-remote-code",
        "--nnodes",
        "2",
        "--tp-size",
        "32",
        "--dp-size",
        "32",
        "--mem-fraction-static",
        "0.82",
        "--max-running-requests",
        "384",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--cuda-graph-bs",
        6,
        8,
        10,
        12,
        "--enable-dp-lm-head",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "7168",
        "--disaggregation-transfer-backend",
        "ascend",
        "--watchdog-timeout",
        9000,
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
    ]
}


class TestQwen3_480B(TestMultiMixUtils):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = int(max_concurrency) * 4
    input_len = 3200
    output_len = 1000
    random_range_ratio = 1
    ttft = 10000
    tpot = 50
    # T: None   800I: xxxxx.     dev：3192/16@51.19ms
    output_token_throughput = 3192

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
