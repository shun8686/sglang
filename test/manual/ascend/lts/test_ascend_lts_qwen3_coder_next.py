import datetime
import os
import unittest

from lts_utils import TestAscendLtsTestCaseBase

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
)
from sglang.test.test_utils import (
    popen_launch_server,
)

# MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-Next_W8A8"

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "5120",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "HCCL_BUFFSIZE": "2000",
    "ZBCCL_LOCAL_MEM_SIZE": "60416",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ZBCCL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBCCL_ENABLE_GRAPH": "1",
}

OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.85,
    "--disable-radix-cache",
    # 如开启radix cache , 需增加
    "--max-prefill-tokens",
    16384,
    # "--context-length",
    # 26384,
    "--max-total-tokens",
    1700000,
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
    100,
    "--cuda-graph-bs",
    2,
    4,
    8,
    16,
    20,
    36,
    48,
    50,
    "--mamba-ssm-dtype",
    "bfloat16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-draft-model-path",
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    "--quantization",
    "modelslim",
    "--base-gpu-id",
    6,
]


class TestLTSQwen3CoderNext(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    max_concurrency = 80
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 100
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

    def testLtsQwen3CoderNext(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            long_seq_configs = {
                "64k+1k": {
                    "input_len": 65536,
                    "output_len": 1024,
                    "max_concurrency": 8,
                    "num_prompts": 8,
                    "ttft": 100000,
                    "tpot": 1000,
                    "tps": 1,
                }
            }
            self.run_long_seq_testcase(long_seq_configs=long_seq_configs)
            self.run_throughput()
            self.run_gsm8k()


if __name__ == "__main__":
    unittest.main()
