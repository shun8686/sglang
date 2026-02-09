import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase

from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)

class TestAscendQwenEagle3(TestDisaggregationBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode", "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size", 8,
            "--dp-size", 8,
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--watchdog-timeout", 9000,
            "--moe-a2a-backend", "deepep",
            "--deepep-mode", "normal",
            "--moe-dense-tp-size", 1,
            "--trust-remote-code",
            "--nnodes", "1",
            "--node-rank", "0",
            "--attention-backend", "ascend",
            "--quantization", "modelslim",
            "--max-running-requests", 192,
            "--disable-radix-cache",
            "--speculative-draft-model-quantization", "unquant",
            "--chunked-prefill-size", -1,
            "--max-prefill-tokens", 32768,
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3",
            "--speculative-num-steps", 3,
            "--speculative-eagle-topk", 1,
            "--speculative-num-draft-tokens", 4,
            "--mem-fraction-static", 0.7,
            "--disable-cuda-graph",
            "--dtype", "bfloat16",
        ]
        cls.extra_envs = {
            "SGLANG_SET_CPU_AFFINITY": "1",
            "STREAMS_PER_DEVICE": "32",
            "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
            "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
            "SGLANG_NPU_USE_MLAPO": "1",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
            "SGLANG_NPU_USE_MULTI_STREAM": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
            "HCCL_BUFFSIZE": "3000",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "TASK_QUEUE_ENABLE": "2",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode", "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size", 8,
            "--dp-size", 8,
            "--base-gpu-id", 8,
            "--moe-dense-tp-size", 1,
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--watchdog-timeout", 9000,
            "--moe-a2a-backend", "deepep",
            "--deepep-mode", "low_latency",
            "--prefill-round-robin-balance",
            "--load-balance-method", "round_robin",
            "--trust-remote-code",
            "--nnodes", "1",
            "--node-rank", "0",
            "--attention-backend", "ascend",
            "--quantization", "modelslim",
            "--max-running-requests", 192,
            "--disable-radix-cache",
            "--speculative-draft-model-quantization", "unquant",
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3",
            "--speculative-num-steps", 3,
            "--speculative-eagle-topk", 1,
            "--speculative-num-draft-tokens", 4,
            "--tokenizer-worker-num", 4,
            "--mem-fraction-static", 0.7,
            "--cuda-graph-bs", 16,
            # "--disable-cuda-graph",
            "--dtype", "bfloat16",
        ]
        cls.extra_envs = {
            "SGLANG_SET_CPU_AFFINITY": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "STREAMS_PER_DEVICE": "32",
            "SGLANG_NPU_USE_MULTI_STREAM": "1",
            "SGLANG_NPU_USE_MLAPO": "1",
            "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
            "TASK_QUEUE_ENABLE": "1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
            "HCCL_BUFFSIZE": "3000",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.77)

    def test_gsm8k(self):
        expect_accuracy = 0.83
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )
        metrics = run_gsm8k(args)
        achieved_accuracy = metrics["accuracy"]
        self.assertGreaterEqual(
            achieved_accuracy,
            expect_accuracy,
            f"Accuracy of {self.model} is {str(achieved_accuracy)}, is lower than {expect_accuracy}",
        )
        print(f"Model {self.model} achieved accuracy: {str(achieved_accuracy)}")

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()


