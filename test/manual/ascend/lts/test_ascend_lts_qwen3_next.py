import datetime
import os
import unittest

from lts_utils import TestAscendLtsTestCaseBase

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.test_utils import (
    popen_launch_server,
)

# MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Next-80B-A3B-Instruct-W8A8"

OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.685,
    "--max-running-requests",
    80,
    "--watchdog-timeout",
    9000,
    "--disable-radix-cache",
    "--cuda-graph-bs",
    80,
    "--max-prefill-tokens",
    28672,
    "--max-total-tokens",
    450560,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    -1,
    "--base-gpu-id",
    8,
]

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "20",
    "HCCL_BUFFSIZE": "2000",
}


class TestLTSQwen332B(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    max_concurrency = 80
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 50
    output_token_throughput = 500
    accuracy = {"gsm8k": 0.90, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "0.0.0.0"
        cls.port = 30010
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

    def test_lts_qwen3_next(self):
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
    unittest.main(verbosity=2)
