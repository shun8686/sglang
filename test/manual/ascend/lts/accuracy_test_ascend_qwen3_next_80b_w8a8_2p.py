import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import popen_launch_server

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
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "400",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "HCCL_BUFFSIZE": "2000",
    "ZBCCL_LOCAL_MEM_SIZE": "60416",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ZBCCL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "ZBCCL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBCCL_ENABLE_GRAPH": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",  # 不加这个跑精度压测会报错
}

QWEN3_NEXT_80B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--page-size",
    128,
    "--tp-size",
    4,
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.85,
    "--disable-radix-cache",
    "--max-prefill-tokens",
    14336,
    "--context-length",
    26384,
    "--max-total-tokens",
    122304,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    180,
    "--cuda-graph-bs",
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    90,
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-path",
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    "--base-gpu-id",
    12,
]


class TestQwen3Next80BA3B(TestAscendLtsTestCaseBase):
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_ENVS
    evalscope_config = {
        "datasets": [
            "mmlu",
            "mmlu_pro",
            "aime25",
            "math_500",
            "gpqa_diamond",
            "gsm8k",
            "ceval",
        ],
        "dataset_args": {
            "aime25": {"few_shot_num": 0},
            "math_500": {
                "few_shot_num": 0,
                "subset_list": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
            },
            "gpqa_diamond": {"subset_list": ["default"]},
            "gsm8k": {"few_shot_num": 8},
            "ceval": {"few_shot_num": 5},
            "mmlu": {"few_shot_num": 5},
            "mmlu_pro": {},
        },
        "eval_batch_size": 16,
        "generation_config": {
            "aime25": {
                "max_tokens": 25000,
                "temperature": 0.6,
                "n": 1,
            },
            "math_500": {
                "max_tokens": 25000,
                "temperature": 0.6,
            },
            "gpqa_diamond": {
                "max_tokens": 25000,
                "temperature": 0.6,
            },
            "gsm8k": {
                "max_tokens": 2048,
                "temperature": 0.0,
            },
            "ceval": {
                "max_tokens": 512,
                "temperature": 0.0,
            },
            "mmlu": {
                "max_tokens": 512,
                "temperature": 0.0,
            },
            "mmlu_pro": {
                "max_tokens": 512,
                "temperature": 0.0,
            },
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.host = "0.0.0.0"
        cls.port = 30077
        cls.base_url = f"http://{cls.host}:{cls.port}"
        env = os.environ.copy()
        for key, value in env.items():
            print(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                print(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

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

    def test_qwen3_next_80b_a3b(self):
        self.run_evalscope()


if __name__ == "__main__":
    unittest.main()
