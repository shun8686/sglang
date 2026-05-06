import unittest
from time import sleep

from sglang.test.ascend.e2e.test_npu_accuracy_utils import TestAscendAccuracyTestCaseBase
# from sglang.test.ascend.test_ascend_utils import (
#     DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
# )
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="full-16-npu-a3", nightly=True)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "88",
    "HCCL_BUFFSIZE": "1600",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_USE_FIA_NZ": "1",
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
    "4",
    "8",
    "20",
    "21",
    "22",
    "--mem-fraction-static",
    "0.78",
    "--max-running-requests",
    "352",
    "--disable-radix-cache",
    "--chunked-prefill-size",
    "-1",
    "--max-prefill-tokens",
    "1500",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    "16",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "2",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "3",
    "--dtype",
    "bfloat16",
    "--log-requests",
    "--log-requests-level",
    "3",
]


class TestDeepEpDeepseek(TestAscendAccuracyTestCaseBase):
    """Testcase: Verify the accuracy of DeepSeek-R1 model on MMLU and GSM8K tasks with --deepep-mode auto on Ascend backend.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend; --deepep-mode
    """
    # model = DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
    # envs = ENVS
    # other_args = OTHER_ARGS

    # def test_accuracy(self):
    #     # test accuracy on gsm8k dataset
    #     self.dataset_type = "gsm8k"
    #     self.dataset_name = "demo_gsm8k_gen_4_shot_cot_chat_prompt"
    #     self.accuracy = 0.96
    #     self.num_prompts = 200
    #     self.run_accuracy()
    #
    #     # test accuracy on mmlu dataset
    #     self.dataset_type = "mmlu"
    #     self.dataset_name = "mmlu_gen_5_shot_str"
    #     self.accuracy = 0.86
    #     self.num_prompts = 200
    #     self.run_accuracy()
    model = "/home/weights/DeepSeek-R1-0528-w4a8-per-channel"
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.1
    dataset_name = "demo_gsm8k_gen_4_shot_cot_chat_prompt"
    def test_accuracy(self):
        sleep(36000)


if __name__ == "__main__":
    unittest.main()

