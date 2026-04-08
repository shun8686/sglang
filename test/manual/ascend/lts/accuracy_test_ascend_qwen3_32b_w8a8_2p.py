import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_32B_EAGLE_MODEL_PATH,
    QWEN3_32B_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import popen_launch_server

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_32B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_USE_DEEPGEMM": "1",
}

QWEN3_32B_OTHER_ARGS = [
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
    16,
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    16384,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_32B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.843,
    "--cuda-graph-bs",
    1,
    4,
    10,
    15,
    16,
    "--dtype",
    "bfloat16",
    "--base-gpu-id",
    12,
]


class TestQwen32B(TestAscendLtsTestCaseBase):
    max_attempts = 5
    model = QWEN3_32B_W8A8_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = 16
    input_len = 6144
    output_len = 1500
    random_range_ratio = 1
    tpot = 17.9
    output_token_throughput = 590
    evalscope_datasets = (
        ["aime24", "math_500", "gpqa_diamaond", "gsm8k", "ceval", "mmlu", "mmlu_pro"],
    )
    evalscope_dataset_args = (
        {
            "aime24": {},
            "math_500": {},
            "gpqa_diamaond": {},
            "gsm8k": {},
            "ceval": {},
            "mmlu": {},
            "mmlu_pro": {},
        },
    )
    evalscope_eval_batch_size = 16

    @classmethod
    def setUpClass(cls):
        cls.host = "0.0.0.0"
        cls.port = 30077
        cls.base_url = f"http://{cls.host}:{cls.port}"
        env = os.environ.copy()
        env.update(cls.envs)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_qwen3_32b(self):
        self.run_evalscope()


if __name__ == "__main__":
    unittest.main()
