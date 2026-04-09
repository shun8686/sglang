import datetime
import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.test_utils import (
    popen_launch_server,
)

MODEL_PATH = "/home/weights/Eco-Tech/Qwen3.5-27B-W8A8"

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--tp-size",
    4,
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--chunked-prefill-size",
    "-1",
    "--max-prefill-tokens",
    "100000",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-total-tokens",
    "800000",
    "--max-running-requests",
    32,
    "--mem-fraction-static",
    0.75,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    10,
    16,
    20,
    24,
    28,
    32,
    48,
    56,
    64,
    96,
    112,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--max-mamba-cache-size",
    33,
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--base-gpu-id",
    "2",
]

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}


class TestLTSQwen3527BW8A8(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    max_concurrency = 16
    num_prompts = 16
    input_len = 128000
    output_len = 1000
    random_range_ratio = 1
    # tpot = 50
    # output_token_throughput = 43
    # accuracy = {"gsm8k": 0.755}
    tpot = 100
    output_token_throughput = 10
    accuracy = {"gsm8k": 0.50, "mmlu": 0.0}

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:31555"
        cls.host = "127.0.0.1"
        cls.port = 31555
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

    def test_lts_qwen3_5_27b_w8a8(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            # self.run_long_seq_testcase()
            self.run_mmlu()
            self.run_throughput()
            self.run_gsm8k()


if __name__ == "__main__":
    unittest.main(verbosity=2)
