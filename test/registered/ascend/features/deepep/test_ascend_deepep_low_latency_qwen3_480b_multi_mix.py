import unittest

from utils.test_ascend_multi_mix_utils import launch_server, TestMultiNodePdMixTestCaseBase
from utils.test_ascend_deepep_mode_config import QWEN3_CODER_480B_A35B_W8A8_MODEL_PATH, NIC_NAME


MODEL_CONFIG = {
    "model_path": QWEN3_CODER_480B_A35B_W8A8_MODEL_PATH,
    "node_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "2100",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_OP_EXPANSION_MODE": "AIV",
   },
    "other_args": [
        "--trust-remote-code",
        "--nnodes", "2",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--quantization", "modelslim",
        "--max-running-requests", 96,
        "--context-length", 8192,
        "--dtype", "bfloat16",
        "--chunked-prefill-size", 1024,
        "--max-prefill-tokens", 458880,
        "--disable-radix-cache",
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "low_latency",
        "--tp-size", 16,
        "--dp-size", 4,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static", 0.7,
        "--cuda-graph-bs", 16, 20, 24,
    ]
}

class TestDeepEpQwen(TestMultiNodePdMixTestCaseBase):
    model_config = MODEL_CONFIG
    # 0.625
    expect_score = 0.56
    # 0.985
    expect_accuracy = 0.9

    def test_qwen3_480b(self):
        launch_server(self.role, self.model_config)
        self.run_test_mmlu()
        self.run_test_gsm8k()


if __name__ == "__main__":
    unittest.main()
