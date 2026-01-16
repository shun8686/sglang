import psutil
import socket
from types import SimpleNamespace

from sglang.test.run_eval import run_eval
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k

DEEPSEEK_R1_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"

DEEPSEEK_V32_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"

DEEPSEEK_CODER_V2_LITE_MODEL_PATH = "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/"

QWEN3_30B_A3B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"

QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8"

QWEN3_235B_A22B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
QWEN3_235B_A22B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"

QWEN3_CODER_480B_A35B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"

def get_nic_name():
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and (addr.address.startswith("172.") or addr.address.startswith("192.")):
                print("The nic name matched is {}".format(nic))
                return nic
    return None

NIC_NAME = get_nic_name()
NIC_NAME = "lo" if NIC_NAME is None else NIC_NAME

def test_mmlu(base_url, model):
    print("Starting gsm8k test...")
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=8,
        num_threads=32,
    )
    metrics = run_eval(args)
    return metrics

def test_gsm8k(base_url):
    print("Starting gsm8k test...")
    colon_index = base_url.rfind(":")
    host = base_url[:colon_index]
    print(f"{host=}")
    port = int(base_url[colon_index + 1:])
    print(f"{port=}")
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host=host,
        port=port,
    )
    metrics = run_gsm8k(args)
    return metrics
