import unittest

from sglang.srt.utils import is_npu
from test_ascend_single_mix_utils import TestSingleMixUtils, NIC_NAME

QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"

QWEN3_235B_A22B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"

QWEN3_235B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PATH": "/usr/local/Ascend/8.5.0/compiler/bisheng/bin:$PATH",
    "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DISABLE_EAGLE_QUANT": "1",
}

QWEN3_235B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes",
        "2",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--max-running-requests",
        "1152",
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
        "--chunked-prefill-size",
        "32768",
        "--max-prefill-tokens",
        "458880",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--disable-radix-cache",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--tp",
        "32",
        "--dp-size",
        "32",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        "6",
        "8",
        "10",
        "12",
        "18",
        "24",
        "32",
        "36",
    ]
    if is_npu()
    else []
)

class TestQwen3_235B(TestSingleMixUtils):
    model = QWEN3_235B_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 480
    num_prompts = 480
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    ttft = 10000
    tpot = 50
    output_token_throughput = 8314

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
