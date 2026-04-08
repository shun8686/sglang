import datetime
import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.test_utils import (
    popen_launch_server,
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
# MODEL_PATH = "/home/weights/Qwen3-32B-Int8"

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
    78,
    "--context-length",
    8192,
    "--enable-hierarchical-cache",
    "--hicache-write-policy",
    "write_through",
    "--hicache-ratio",
    3,
    "--chunked-prefill-size",
    43008,
    "--max-prefill-tokens",
    52500,
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.68,
    "--cuda-graph-bs",
    78,
    "--dtype",
    "bfloat16",
    "--enable-metrics",
    "--base-gpu-id",
    12,
]

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}


class TestLTSQwen332B(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    request_rate = 5.5
    max_concurrency = 16
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 50
    output_token_throughput = 350
    accuracy = {"gsm8k": 0.80, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "0.0.0.0"
        cls.port = 30020
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

    def testLtsQwen32b(self):
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
