import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_NEXT_80B_A3B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "HCCL_BUFFSIZE": "2000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "30",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
}

QWEN3_NEXT_80B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--page-size",
    128,
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.7,
    "--watchdog-timeout",
    9000,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    26384,
    "--context-length",
    26384,
    "--max-running-requests",
    256,
    "--cuda-graph-bs",
    4,
    16,
    32,
    64,
    128,
    140,
    160,
    180,
    200,
    216,
    "--mamba-ssm-dtype",
    "bfloat16",
]


class TestQwen3Next80BA3B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = 320
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    # T: 1976@50ms       800I A3: None      Dev-800I: 1405.17/2 @49.91ms
    output_token_throughput = 1410

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
