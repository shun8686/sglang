from sglang.srt.utils import is_npu
from test_ascend_single_mix_utils import NIC_NAME

QWEN3_30B_A3B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"
QWEN3_A3B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3"

QWEN3_30B_A3B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
#    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DISABLE_EAGLE3_QUANT": "1",
}

QWEN3_30B_A3B_OTHER_ARGS = (
    [
        "--tp",
        "2",
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
        "48",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "65536",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        "156",
        "--dtype",
        "bfloat16",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_A3B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-draft-model-quantization",
        "unquant",
    ]
)



