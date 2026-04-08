import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_14B_EAGLE_MODEL_PATH,
    QWEN3_14B_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import popen_launch_server

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_14B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_USE_FIA": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

QWEN3_14B_OTHER_ARGS = [
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
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.8,
    "--tp-size",
    2,
    "--dp-size",
    1,
    "--sampling-backend",
    "ascend",
    "--max-running-requests",
    16,
    "--served-model-name",
    "Qwen3-14B",
    "--chunked-prefill-size",
    -1,
    "--cuda-graph-bs",
    16,
    "--dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_14B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--schedule-conservativeness",
    0.01,
    "--base-gpu-id",
    12,
]


class TestQwen14B(TestAscendLtsTestCaseBase):
    max_attempts = 5
    model = QWEN3_14B_W8A8_MODEL_PATH
    other_args = QWEN3_14B_OTHER_ARGS
    envs = QWEN3_14B_ENVS
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = 16
    input_len = 6144
    output_len = 1500
    random_range_ratio = 1
    tpot = 14.92
    output_token_throughput = 723
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

    def test_qwen3_14b(self):
        self.run_evalscope()


if __name__ == "__main__":
    unittest.main()
