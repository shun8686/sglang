from test_ascend_single_mix_utils import NIC_NAME

Qwen3_Next_80B_A3B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct"

Qwen3_Next_80B_A3B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "2000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring"
}

Qwen3_Next_80B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    "8",
    "--context-length",
    "8000",
    "--chunked-prefill-size",
    "6400",
    "--max-prefill-tokens",
    "8000",
    "--max-total-tokens",
    "16000",
    "--disable-radix-cache",
    "--tp-size",
    "4",
    "--dp-size",
    "1",
    "--mem-fraction-static",
    "0.78",
    "--cuda-graph-bs",
    1,
    2,
    6,
    8,
    16,
    32,
]
