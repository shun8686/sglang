import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_A3B_EAGLE_MODEL_PATH,
    TestAscendPerformanceTestCaseBase, QWEN3_5_27B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    # "--quantization",
    # "modelslim",
    "--max-running-requests",
    32,
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-total-tokens",
    800000,
    "--max-prefill-tokens",
    100000,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-draft-model-path",
    QWEN3_A3B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--tp-size",
    8,
    "--mem-fraction-static",
    0.75,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    10,
    16,
    24,
    28,
    32,
    48,
    56,
    64,
    96,
    112,
    "--mm-attention-backend",
    "ascend_attn",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--dtype",
    "bfloat16",
]


class TestQwen3527B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_5_27B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 20

    def test_qwen3_5_27b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
