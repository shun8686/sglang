from test_ascend_single_mix_utils import NIC_NAME

QWEN3_8B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-8B-W8A8"

QWEN3_8B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,     
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_8B_OTHER_ARGS = (
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
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "43008",
        "--max-prefill-tokens",
        "525000",
        "--tp-size",
        "2",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        "16",
        "--dtype",
        "bfloat16",    
    ]
)

