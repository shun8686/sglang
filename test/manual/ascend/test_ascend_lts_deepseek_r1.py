import os
import subprocess
import sys
import datetime

import psutil
import socket
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
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

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
#        "SGLANG_USE_AG_AFTER_QLORA": "1",
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
        "TASK_QUEUE_ENABLE": "0",
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
        0.6,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        8,
        "--context-length",
        8192,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        28680,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        "--dp-size",
        2,
        "--enable-dp-attention",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
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
        "--max-running-requests",
        384,
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--cuda-graph-bs",
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        "--watchdog-timeout",
        9000,
        "--context-length",
        65536,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        3,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        4,
        "--prefill-round-robin-balance",  
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--tokenizer-worker-num",
        4,
    ],
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

def run_bench_serving(host, port, dataset_name="random", dataset_path="", request_rate=8.0, max_concurrency=8, num_prompts=32, input_len=3500, output_len=1500,
                      random_range_ratio=1.0):
    command = (f"python3 -m sglang.bench_serving --backend sglang --host {host} --port {port} --dataset-name {dataset_name} --dataset-path {dataset_path} --request-rate {request_rate} "
               f"--max-concurrency {max_concurrency} --num-prompts {num_prompts} --random-input-len {input_len} "
               f"--random-output-len {output_len} --random-range-ratio {random_range_ratio}")
    print(f"command:{command}")
    metrics = run_command(f"{command} | tee ./bench_log.txt")
    return metrics

# 新增：通用单条长序列测试函数（适配16k+1k/32k+1k/64k+1k）
def run_single_long_seq_test(host, port, input_len, output_len, seq_type):
    command = (f"python3 -m sglang.bench_serving --backend sglang --host {host} --port {port} --dataset-name random "
               f"--request-rate 0 --max-concurrency 1 --num-prompts 1 "
               f"--random-input-len {input_len} --random-output-len {output_len} "
               f"--random-range-ratio 0.0")  # 固定长度，不随机
    print(f"{seq_type} single long sequence test command:{command}")
    # 不同长度日志分开保存，避免覆盖
    metrics = run_command(f"{command} | tee ./single_long_seq_{seq_type}_log.txt")
    return metrics

class TestLTSQwen3235B(CustomTestCase):
    model = QWEN3_235B_MODEL_PATH
    dataset_name = "random"
    dataset_path = "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"  # the path of test dataset
    other_args = QWEN3_235B_OTHER_ARGS
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10
    envs = QWEN3_235B_ENVS
    request_rate = 5.5
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 50
    output_token_throughput = 8314
    accuracy = 0.80

    # 新增：三种长序列配置（16k+1k/32k+1k/64k+1k）
    long_seq_configs = {
        "16k+1k": {
            "input_len": 16384,
            "output_len": 1024,
            "ttft_threshold": 40000, 
            "tpot_threshold": 200
        },
        "32k+1k": {
            "input_len": 32768,
            "output_len": 1024,
            "ttft_threshold": 70000,
            "tpot_threshold": 250
        },
        "64k+1k": {
            "input_len": 65536,
            "output_len": 1024,
            "ttft_threshold": 100000,
            "tpot_threshold": 350
        }
    }

    print("Nic name: {}".format(NIC_NAME))

    def run_throughput(self):
        _, host, port = self.base_url.split(":")
        host = host[2:]
        metrics = run_bench_serving(
            host=host,
            port=port,
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

    # 新增：批量执行三种长序列验证
    def run_all_long_seq_verify(self):
        """依次验证16k+1k、32k+1k、64k+1k三种单条长序列"""
        _, host, port = self.base_url.split(":")
        host = host[2:]

        for seq_type, config in self.long_seq_configs.items():
            print(f"\n========== Start {seq_type} single long sequence test ==========")
            # 执行单条长序列请求
            metrics = run_single_long_seq_test(
                host=host,
                port=port,
                input_len=config["input_len"],
                output_len=config["output_len"],
                seq_type=seq_type
            )
            print(f"{seq_type} metrics: {metrics}")

            log_file = f"./single_long_seq_{seq_type}_log.txt"
            res_ttft = run_command(f"cat {log_file} | grep 'Mean TTFT' | awk '{{print $4}}'")
            res_tpot = run_command(f"cat {log_file} | grep 'Mean TPOT' | awk '{{print $4}}'")
            res_error = run_command(f"cat {log_file} | grep 'Error'")

            res_ttft = res_ttft.strip() if res_ttft else "0"
            res_tpot = res_tpot.strip() if res_tpot else "0"

            self.assertLessEqual(
                float(res_ttft),
                config["ttft_threshold"],
                f"{seq_type} TTFT {res_ttft}ms exceeds threshold {config['ttft_threshold']}ms"
            )
            self.assertLessEqual(
                float(res_tpot),
                config["tpot_threshold"],
                f"{seq_type} TPOT {res_tpot}ms exceeds threshold {config['tpot_threshold']}ms"
            )
            # 验证无错误日志
            self.assertEqual(
                res_error, "",
                f"{seq_type} request failed with error: {res_error}"
            )
            print(f"========== {seq_type} single long sequence test PASSED ==========\n")

    def run_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )

    def test_lts_qwen3_235b(self):
        i = 0
        while True:
            i = i + 1
            time_str_1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"=============={time_str_1}  Execute the {i}-th long-term stability test==============")
            self.run_throughput()
            self.run_gsm8k()
            self.run_all_long_seq_verify()


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_file = "/tmp/lts_test_qwen3_235b_" + time_str + ".log"

    with open(log_file, 'w', encoding="utf-8") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f

        try:
            unittest.main(verbosity=2)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Test log saved to {log_file}")
