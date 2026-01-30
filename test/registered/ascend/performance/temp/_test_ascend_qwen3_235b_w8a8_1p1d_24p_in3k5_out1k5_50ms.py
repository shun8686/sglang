import unittest
import os

from sglang.test.ascend.performance.test_ascend_performance_utils import (
    TestAscendMultiNodePdSepTestCaseBase,
    NIC_NAME,
    run_command
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"

QWEN3_235B_A22B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
        "HCCL_BUFFSIZE": "4300",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
#        "ENABLE_ASCEND_MOE_NZ": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
#        "ENABLE_PROFILING": "1",
        "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR": "/data/d00662834/hot_map",
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "DP_ROUND_ROBIN": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
        "HCCL_BUFFSIZE": "512",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "STREAMS_PER_DEVICE": "32",
    },
    "prefill_args": [
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp-size",
        16,
        "--dp-size",
        16,
        "--mem-fraction-static",
        0.6,
        "--disable-radix-cache",
        # "--ep-dispatch-algorithm",
        # "static",
        # "--init-expert-location",
        # "/hot_map/xxx.pt",
        "--quantization",
        "modelslim",
       "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-draft-model-quantization",
        "unquant",
        "--max-running-requests",
        "128",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "2048",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--dtype",
        "bfloat16",
        "--expert-distribution-recorder-buffer-size",
        "-1",
        "--expert-distribution-recorder-mode",
        "stat",
        "--ep-dispatch-algorithm",
        "static",
        "--enable-expert-distribution-metrics",
    ],
    "decode_args": [
        "--disaggregation-mode",
        "decode",
        "--nnodes",
        "2",
        "--tp-size",
        32,
        "--dp-size",
        32,
        "--mem-fraction-static",
        0.83,
        "--max-running-requests",
        768,
        "--quantization",
        "modelslim",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--cuda-graph-bs",
        6,
        8,
        12,
        15,
        18,
        20,
        22,
        24,
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_235B_A22B_EAGLE_MODEL_PATH,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-draft-model-quantization",
        "unquant",
        "--watchdog-timeout",
        9000,
        "--context-length",
        8192,
        "--prefill-round-robin-balance",
        "--enable-dp-lm-head",
        "--tokenizer-worker-num",
        4,
        "--dtype",
        "bfloat16",
        "--load-balance-method",
        "round_robin",
    ],
}


class TestQwen3_235B_w8a8_1p2d_in3500_out1500(TestAscendMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 768
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1

    def test_throughput(self):
        self.run_throughput(run_cycles=0)
        if self.role == "router":
            os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = "/data/d00662834/hot_map"
            print("Begin to dump hotmap data")
            run_command(f"curl --location 'http://127.0.0.1:6688/dump_expert_distribution_record'")



if __name__ == "__main__":
    unittest.main()
