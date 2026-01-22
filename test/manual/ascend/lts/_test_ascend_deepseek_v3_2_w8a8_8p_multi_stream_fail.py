import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from lts_utils import NIC_NAME

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server, CustomTestCase, \
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"

ENVS = {
    # "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_USE_MLAPO": "0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "TASK_QUEUE_ENABLE": "1",

    # "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
    # "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    # "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    # "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
    # "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    # "SGLANG_ENABLE_SPEC_V2": "1",
}

OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--tp-size", 16,
        "--quantization", "modelslim",
        "--mem-fraction-static", 0.81,
        "--chunked-prefill-size", -1,
        "--context-length", 8192,
        "--max-prefill-tokens", 20480,
        "--max-running-requests", 64,
        "--cuda-graph-bs", 16, 32, 64,
        "--cuda-graph-max-bs", 64,
        "--watchdog-timeout", 600,
        "--disable-radix-cache",
    ]
)

class TestDeepSeekV32(CustomTestCase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = 320
    input_len = 512
    output_len = 512
    random_range_ratio = 1
    tpot = 500
    output_token_throughput = 50
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        if cls.envs:
            for key, value in cls.envs.items():
                print(f"ENV_VAR_CASE {key}:{value}")
        env = os.environ.copy()
        for key, value in cls.envs.items():
            print(f"ENV_VAR_OTHER {key}:{value}")
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

    def test_deepseek_v3_2_by_gsm8k(self):
        colon_index = self.base_url.rfind(":")

        host = self.base_url[:colon_index]
        print("host:", host)
        port = int(self.base_url[colon_index+1:])
        print("port:", port)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1000,
            max_new_tokens=256,
            parallel=80,
            host=host,
            port=port,
        )
        for i in range(10):
            metrics = run_gsm8k(args)
            print(f"{metrics=}")
            print(f"{metrics['accuracy']=}")


if __name__ == "__main__":
    unittest.main()
