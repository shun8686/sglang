from sglang.srt.utils import is_npu
from test_ascend_single_mix_utils import NIC_NAME

QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"

QWEN3_32B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_32B_OTHER_ARGS = (
    [
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
        "16",
        "--context-length",
        "8192",
        "--enable-hierarchical-cache",
        "--hicache-write-policy",
        "write_through",
        "--hicache-ratio",
        "3",
        "--chunked-prefill-size",
        "43008",
        "--max-prefill-tokens",
        "52500",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.68",
        "--cuda-graph-bs",
        "78",
        "--dtype",
        "bfloat16"
    ]
    if is_npu()
    else []
)

