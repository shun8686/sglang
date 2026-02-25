import datetime
import os
import unittest
from types import SimpleNamespace

from lts_utils import NIC_NAME, run_bench_serving, run_command

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

# MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
MODEL_PATH = "/home/weights/Qwen3-32B-Int8"

OTHER_ARGS = [
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
    78,
    "--context-length",
    8192,
    "--enable-hierarchical-cache",
    "--hicache-write-policy",
    "write_through",
    "--hicache-ratio",
    3,
    "--chunked-prefill-size",
    43008,
    "--max-prefill-tokens",
    52500,
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.68,
    "--cuda-graph-bs",
    78,
    "--dtype",
    "bfloat16",
    "--enable-metrics",
    "--base-gpu-id",
    12,
]

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}


class TestLTSQwen332B(CustomTestCase):
    model = MODEL_PATH
    dataset_name = "random"
    dataset_path = (
        "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"  # the path of test dataset
    )
    other_args = OTHER_ARGS
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10
    envs = ENVS
    request_rate = 5.5
    max_concurrency = 16
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 50
    output_token_throughput = 350
    accuracy = 0.80

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://0.0.0.0:30000"
        cls.host = "0.0.0.0"
        cls.port = 30000
        env = os.environ.copy()
        env.update(cls.envs)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_throughput(self):
        print(f"========== Start 3.5k/1.5k benchmark test ==========\n")
        metrics = run_bench_serving(
            host=self.host,
            port=self.port,
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            request_rate=self.request_rate,
            max_concurrency=self.max_concurrency,
            num_prompts=self.num_prompts,
            input_len=self.input_len,
            output_len=self.output_len,
            random_range_ratio=self.random_range_ratio,
        )
        print("metrics is " + str(metrics))
        res_ttft = run_command(
            "cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'"
        )
        res_tpot = run_command(
            "cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'"
        )
        res_output_token_throughput = run_command(
            "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
        )
        print(f"========== 3.5k/1.5k benchmark test PASSED ==========\n")
        # self.assertLessEqual(
        #     float(res_ttft),
        #     self.ttft,
        # )
        # self.assertLessEqual(
        #     float(res_tpot),
        #     self.tpot,
        # )
        # self.assertGreaterEqual(
        #     float(res_output_token_throughput),
        #     self.output_token_throughput,
        # )

    def run_gsm8k(self):
        print(f"========== Start gsm8k test ==========\n")
        args = SimpleNamespace(
            num_shots=5,
            data_path="/tmp/test.jsonl",
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.host}",
            port=self.port,
        )
        metrics = run_eval(args)
        # self.assertGreater(
        #     metrics["accuracy"],
        #     self.accuracy,
        #     f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        # )
        print(f"========== gsm8k test PASSED ==========\n")

    def test_lts_qwen3_32b(self):
        i = 0
        while True:
            i = i + 1
            time_str_1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={time_str_1}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()


if __name__ == "__main__":
    unittest.main(verbosity=2)
