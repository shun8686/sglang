import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    NIC_NAME
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"

MODEL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
#    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "36",
    "HCCL_BUFFSIZE": "1600",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "ENABLE_MOE_NZ": "1",
}
MODEL_OTHER_ARGS = (
    [
        "--tp",
        "16",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--watchdog-timeout",
        "9000",
        "--cuda-graph-bs",
        "8",
        "16",
        "24",
        "28",
        "32",
        "36",
        "--mem-fraction-static",
        "0.71",
        "--max-running-requests",
        "144",
        "--context-length",
        "128000",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "128000",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--enable-dp-attention",
        "--dp-size",
        "4",
        "--enable-dp-lm-head",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--dtype",
        "bfloat16",
    ]
)


class Test_Ascend_DeepSeek_R1_W4A8_In3500_Out3500(TestSingleMixUtils):
    model = MODEL_PATH
    other_args = MODEL_OTHER_ARGS
    envs = MODEL_ENVS
    dataset_name = "gsm8k"
    dataset_path = "/root/.cache/modelscope/hub/datasets/DeepSeek-R1-0528-w4a8/GSM8K-in3500-mix-jsonl"
    max_concurrency = 128
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    ttft = 10000
    tpot = 50
    output_token_throughput = 1000

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
