import datetime
import os
import unittest

from lts_utils import TestAscendLtsTestCaseBase

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.test_utils import (
    popen_launch_server,
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"
OTHER_ARGS = [
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
    480,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    32768,
    "--max-prefill-tokens",
    68000,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-draft-model-path",
    EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--tp",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    0.78,
    "--cuda-graph-bs",
    6,
    8,
    10,
    12,
    15,
    18,
    28,
    30,
]

QWEN3_235B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
}


class TestLTSQwen3235B(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = QWEN3_235B_ENVS
    request_rate = 5.5
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 50
    output_token_throughput = 8314
    accuracy = {"gsm8k": 0.80, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "0.0.0.0"
        cls.port = 30050
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

    def testLtsQwen235b(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()
            self.run_all_long_seq_verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
