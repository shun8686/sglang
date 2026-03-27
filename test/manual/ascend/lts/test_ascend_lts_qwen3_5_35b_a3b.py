import datetime
import os
import sys
import unittest

from lts_utils import TestAscendLtsTestCaseBase

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.test_utils import (
    popen_launch_server,
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B"

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--context-length",
    32768,
    "--mem-fraction-static",
    0.65,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    24,
    "--enable-multimodal",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--mm-attention-backend",
    "ascend_attn",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--disable-radix-cache",
    "--base-gpu-id",
    0,
]


class TestLtsQwen35(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    output_file = "./log/bench_results.jsonl"
    max_concurrency = 16
    num_prompts = 16
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 0
    accuracy = {"gsm8k": 0.80, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "127.0.0.1"
        cls.port = 21001
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

    def testLtsQwen35(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs("log", exist_ok=True)
    log_file = (
        f"./log/lts_{os.path.splitext(os.path.basename(__file__))[0]}_{time_str}.log"
    )

    with open(log_file, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f

        try:
            unittest.main(verbosity=2)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
