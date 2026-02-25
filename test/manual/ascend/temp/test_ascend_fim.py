import unittest

import requests

from sglang.test.ascend.e2e.test_ascend_multi_node_utils import (
    NIC_NAME,
    TestAscendMultiNodePdSepTestCaseBase,
    check_role,
)

# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8"
)
MODEL_CONFIG = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "HCCL_BUFFSIZE": "1536",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "720",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.8,
        "--disable-radix-cache",
        "--dp-size",
        2,
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--page-size",
        "128",
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.2",
        "--hicache-size",
        "0",
        "--hicache-write-policy",
        "write_through",
        "--hicache-storage-backend",
        "file",
        "--hicache-storage-prefetch-policy",
        "wait_complete",
    ],
    "decode_args": [
        "--nnodes",
        "1",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        16,
        "--dp-size",
        16,
        "--mem-fraction-static",
        0.8,
        "--prefill-round-robin-balance",
        "--disable-shared-experts-fusion",
        "--load-balance-method",
        "round_robin",
        "--page-size",
        "64",
    ],
    "router_args": [],
}


class TestDeepSeekR1W4A8(TestAscendMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG

    @classmethod
    def setUpClass(cls):
        super(TestDeepSeekR1W4A8, cls).setUpClass()
        cls.start_pd_seperation_server()
        cls.start_router_server()

    @classmethod
    def tearDownClass(cls):
        super(TestDeepSeekR1W4A8, cls).tearDownClass()
        cls.stop_sglang_thread()

    @check_role(allowed_roles=["router"])
    def test_enable_hicache(self):
        for i in range(2):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, "
                    "just return me a string with of 5000 characters,just return me a string with of 5000 characters, ",
                    "max_tokens": 1,
                },
            )
            self.assertEqual(response.status_code, 200)
            print(
                "------------------------------------response.json()---------------------------------------------"
            )
            print(response.json())

            # if i == 1:
            #     cached_tokens = response.json()["usage"]['prompt_tokens_details']['cached_tokens']
            #     print(cached_tokens)
            #     self.assertEqual(256, cached_tokens)


if __name__ == "__main__":
    unittest.main()
