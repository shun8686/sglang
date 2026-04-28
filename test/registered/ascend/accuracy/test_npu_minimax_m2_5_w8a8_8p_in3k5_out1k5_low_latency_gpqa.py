import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
)

MINIMAX_M2_5_LOW_LATENCY_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "800",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "204800",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
}

MINIMAX_M2_5_LOW_LATENCY_OTHER_ARGS = [
    "--model-path",
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    "--host",
    "127.0.0.1",
    "--port",
    32000,
    "--tp-size",
    16,
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--ep-size",
    16,
    "--mem-fraction-static",
    0.75,
    "--max-running-requests",
    512,
    "--disable-radix-cache",
    "--prefill-delayer-max-delay-passes",
    "--enable-prefill-delayer",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    4096,
    "--cuda-graph-bs",
    1,
    7,
    8,
    9,
    10,
    16,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dtype",
    "bfloat16",
    "--external-model-package",
    "custom_eagle3",
]


class TestNPUMiniMaxM2_5_W8A8_8P_GPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for MiniMax-M2.5-w8a8 8p single node low latency on GPQA"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_LOW_LATENCY_OTHER_ARGS
    envs = MINIMAX_M2_5_LOW_LATENCY_ENVS
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    batch_size = 128
    max_out_len = 1024

    def test_npu_minimax_m2_5_w8a8_8p_gpqa(self):
        """Run NPU accuracy test for MiniMax-M2.5-w8a8 8p single node low latency on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()