import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import TestAscendAccuracyTestCaseBase
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="full-16-npu-a3", nightly=True)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "HCCL_BUFFSIZE": "3000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "0",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--tp",
    "16",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--watchdog-timeout",
    "9000",
    "--cuda-graph-bs",
    "8",
    "16",
    "24",
    "28",
    "32",
    "36",
    "--mem-fraction-static",
    "0.6",
    "--max-running-requests",
    "144",
    "--context-length",
    "8188",
    "--disable-radix-cache",
    "--chunked-prefill-size",
    "512",
    "--max-prefill-tokens",
    "4096",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--enable-dp-attention",
    "--dp-size",
    "4",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--dtype",
    "bfloat16",
]


class TestDeepEpDeepseek(TestAscendAccuracyTestCaseBase):
    """Testcase: Verify the accuracy of DeepSeek-R1 model on MMLU and GSM8K tasks with --deepep-mode low_latency on Ascend backend.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend; --deepep-mode
    """
    model = DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
    envs = ENVS
    other_args = OTHER_ARGS

    def test_accuracy(self):
        # test accuracy on gsm8k dataset
        self.dataset_type = "gsm8k"
        self.dataset_name = "demo_gsm8k_gen_4_shot_cot_chat_prompt"
        self.accuracy = 0.96
        self.num_prompts = 200
        self.run_accuracy()

        # test accuracy on mmlu dataset
        self.dataset_type = "mmlu"
        self.dataset_name = "mmlu_gen_5_shot_str"
        self.accuracy = 0.86
        self.num_prompts = 200
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()

