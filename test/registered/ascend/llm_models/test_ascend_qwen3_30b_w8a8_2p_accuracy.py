import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    # QWEN3_30B_A3B_W8A8_MODEL_PATH,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_30B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "SGLANG_NPU_USE_DEEPGEMM": "1",
}

QWEN3_30B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
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
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.845,
    "--cuda-graph-bs",
    16,
    32,
    64,
    72,
    "--dtype",
    "bfloat16",
    "--base-gpu-id",
    12,
]


class TestQwen30B(TestAscendAccuracyTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    # model = QWEN3_30B_A3B_W8A8_MODEL_PATH
    model = "/home/weights/Qwen/Qwen3-30B-A3B-W8A8"
    other_args = QWEN3_30B_OTHER_ARGS
    envs = QWEN3_30B_ENVS
    accuracy = 0.1
    dataset_name = "demo_gsm8k_gen_4_shot_cot_chat_prompt"
    def test_qwen3_30b(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
