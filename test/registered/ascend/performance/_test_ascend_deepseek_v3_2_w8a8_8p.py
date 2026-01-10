import unittest
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval
from test_ascend_single_mix_utils import TestSingleNodeTestCaseBase, NIC_NAME

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"

ENVS = {
    # "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1024",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "SGLANG_NPU_USE_MLAPO": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
    "TASK_QUEUE_ENABLE": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",

}

OTHER_ARGS = (
    [
        "--tp-size", "16",
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--mem-fraction-static", 0.9,
        "--chunked-prefill-size", "8192",
        "--context-length", "40970",
        "--max-prefill-tokens", "40970",
        "--max-total-tokens", "40970",
        "--watchdog-timeout", "9000",
        # "--disable-radix-cache",
        # "--max-running-requests", 128,
        "--disable-cuda-graph",
    ]
)

# OTHER_ARGS = (
#     [
#         "--trust-remote-code",
#         "--mem-fraction-static", 0.9,
#         "--attention-backend", "ascend",
#         "--device", "npu",
#         "--disable-cuda-graph",
#         "--tp-size", "16",
#         "--quantization", "modelslim",
#     ]
# )

class TestDeepSeekV32(TestSingleNodeTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 128
    num_prompts = 160
    input_len = 512
    output_len = 512
    random_range_ratio = 1
    tpot = 500
    output_token_throughput = 50

    def test_deepseek_v3_2(self):
        self.run_throughput()

    # def test_deepseek_v3_2_by_gsm8k(self):
    #     colon_index = self.base_url.rfind(":")
    #
    #     host = self.base_url[:colon_index]
    #     print("host:", host)
    #     port = int(self.base_url[colon_index+1:])
    #     print("port:", port)
    #     args = SimpleNamespace(
    #         num_shots=5,
    #         data_path=None,
    #         num_questions=200,
    #         max_new_tokens=512,
    #         parallel=128,
    #         host=host,
    #         port=port,
    #     )
    #     for i in range(10):
    #         metrics = run_eval(args)
    #         print(f"{metrics=}")
    #         print(f"{metrics['accuracy']=}")


if __name__ == "__main__":
    unittest.main()
