import os
import subprocess
import psutil
import socket
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

def get_nic_name():
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address.startswith("192."):
                print("The nic name matched is {}".format(nic))
                return nic
    return None

NIC_NAME = "lo" if get_nic_name() == None else get_nic_name()

QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
QWEN3_32B_OTHER_ARGS = [
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
        "78",
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

QWEN3_32B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
}


def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None

def run_bench_serving(host, port, dataset_name="random", request_rate=8, max_concurrency=8, num_prompts=32, input_len=1024, output_len=1024,
                      random_range_ratio=1):
    command = (f"python3 -m sglang.bench_serving --backend sglang --host {host} --port {port} --dataset-name {dataset_name} --request-rate {request_rate} "
               f"--max-concurrency {max_concurrency} --num-prompts {num_prompts} --random-input-len {input_len} "
               f"--random-output-len {output_len} --random-range-ratio {random_range_ratio}")
    print(f"command:{command}")
    metrics = run_command(f"{command} | tee ./bench_log.txt")
    return metrics

class TestLTSQwen332B(CustomTestCase):
    model = QWEN3_32B_MODEL_PATH
    dataset_name = "random"
    other_args = QWEN3_32B_OTHER_ARGS
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10
    envs = QWEN3_32B_ENVS
    request_rate = 5.5
    max_concurrency = 16
    num_prompts = int(max_concurrency) * 4
    input_len = 1024
    output_len = 300
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 30
    output_token_throughput = 500

    print("Nic name: {}".format(NIC_NAME))

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
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
        _, host, port = self.base_url.split(":")
        host = host[2:]
        metrics = run_bench_serving(
            host=host,
            port=port,
            dataset_name=self.dataset_name,
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
        self.assertGreater(
            float(res_ttft),
            0,
        )
        self.assertGreater(
            float(res_tpot),
            0,
        )
        self.assertGreater(
            float(res_output_token_throughput),
            0,
        )

if __name__ == "__main__":
    unittest.main()
