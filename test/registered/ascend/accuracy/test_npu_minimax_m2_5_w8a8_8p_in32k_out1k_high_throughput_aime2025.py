import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-16-npu-a3",
    nightly=True,
)

MINIMAX_M2_5_32K_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1500",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
}

MINIMAX_M2_5_32K_OTHER_ARGS = [
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
    8,
    "--ep-size",
    16,
    "--mem-fraction-static",
    0.6,
    "--prefill-delayer-max-delay-passes",
    200,
    "--enable-prefill-delayer",
    "--max-running-requests",
    24,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    32768,
    "--cuda-graph-bs",
    1,
    2,
    3,
    4,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
]


class TestNPUMiniMaxM2_5_W8A8_8P_AIME2025(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for MiniMax-M2.5-w8a8 8p single node high throughput in32k on AIME 2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_32K_OTHER_ARGS
    envs = MINIMAX_M2_5_32K_ENVS
    accuracy = 0.8
    dataset_type = "aime2025"
    dataset_name = "aime2025_gen"
    batch_size = 64
    max_out_len = 8192

    def test_npu_minimax_m2_5_w8a8_8p_aime2025(self):
        """Run NPU accuracy test for MiniMax-M2.5-w8a8 8p single node high throughput in32k on AIME 2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()