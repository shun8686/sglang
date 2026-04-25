import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import TestAscendAccuracyTestCaseBase, BENCHMARK_TOOL_DEFAULT
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "HCCL_BUFFSIZE": "4096",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--quantization",
    "modelslim",
    "--tp-size",
    "8",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--max-running-requests",
    128,
    "--disable-cuda-graph",
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--chunked-prefill-size",
    1024,
    "--mem-fraction-static",
    0.68,
    "--base-gpu-id",
    8,
    "--log-requests",
]

class TestDeepEpDeepseek(TestAscendAccuracyTestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http:127.0.0.1:6666"
        cls.dataset_type = "mmlu"

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    # base_url = DEFAULT_URL_FOR_TEST
    # model = QWEN3_30B_A3B_W8A8_MODEL_PATH
    model = "/home/weights/DeepSeek-V2-Lite-W8A8"
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.1
    # dataset_name = "demo_gsm8k_gen_4_shot_cot_chat_prompt"
    dataset_name = "mmlu_gen_5_shot_str"
    def test_accuracy(self):
        self.run_accuracy()




if __name__ == "__main__":
    unittest.main()
