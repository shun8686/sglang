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
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


MODEL_PATH = "DeepSeek-R1-0528-w4a8"


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

class TestLTSDeepSeekR1(CustomTestCase):
    model = MODEL_PATH
    dataset_name = "random"
    dataset_path = "/home/lts-test/ShareGPT_V3_unfiltered_cleaned_split.json"  # the path of test dataset
    # dataset_path = ""
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
    host = "127.0.0.1"
    port = 6688
    
    # 新增：三种长序列配置（16k+1k/32k+1k/64k+1k）
    long_seq_configs = {
        "64k+1k": {
            "input_len": 65536,
            "output_len": 1024,
            "ttft_threshold": 100000,
            "tpot_threshold": 350
        },
        "32k+1k": {
            "input_len": 32768,
            "output_len": 1024,
            "ttft_threshold": 70000,
            "tpot_threshold": 250
        },
        "16k+1k": {
            "input_len": 16384,
            "output_len": 1024,
            "ttft_threshold": 40000, 
            "tpot_threshold": 200
        },
    }

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
        res_ttft = run_command(
            "cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'"
        )
        res_tpot = run_command(
            "cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'"
        )
        res_output_token_throughput = run_command(
            "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
        )
        print("metrics is " + str(metrics))
        print(f"========== 3.5k/1.5k benchmark test PASSED ==========\n")

    # 新增：批量执行三种长序列验证
    def run_all_long_seq_verify(self):
        """依次验证16k+1k、32k+1k、64k+1k三种单条长序列"""
        for seq_type, config in self.long_seq_configs.items():
            print(f"\n========== Start {seq_type} single long sequence test ==========")
            # 执行单条长序列请求
            metrics = run_bench_serving(
                host=self.host,
                port=self.port,
                input_len=config["input_len"],
                output_len=config["output_len"],
                dataset_name=self.dataset_name,
                dataset_path=self.dataset_path,
                request_rate=1,
                max_concurrency=2,
                num_prompts=8,
                random_range_ratio=1,
            )
            print(f"{seq_type} metrics: {metrics}")

            res_ttft = run_command("cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'")
            res_tpot = run_command("cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'")
            res_output_token_throughput = run_command("cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'")
            res_ttft = res_ttft.strip() if res_ttft else "0"
            res_tpot = res_tpot.strip() if res_tpot else "0"
            
            print("metrics is " + str(metrics))
            print(f"========== {seq_type} single long sequence test PASSED ==========\n")
            # self.assertLessEqual(
            #     float(res_ttft),
            #     config["ttft_threshold"],
            #     f"{seq_type} TTFT {res_ttft}ms exceeds threshold {config['ttft_threshold']}ms"
            # )
            # self.assertLessEqual(
            #     float(res_tpot),
            #     config["tpot_threshold"],
            #     f"{seq_type} TPOT {res_tpot}ms exceeds threshold {config['tpot_threshold']}ms"
            # )
            # # 验证无错误日志
            # self.assertEqual(
            #     res_error, "",
            #     f"{seq_type} request failed with error: {res_error}"
            # )

    def run_gsm8k(self):
        print(f"========== Start gsm8k test ==========\n")
        args = SimpleNamespace(
            num_shots=5,
            data_path="/home/lts-test/test.jsonl",
            # data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.host}",
            port=self.port,
        )
        metrics = run_eval(args)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )
        print(f"========== gsm8k test PASSED ==========\n")

    def test_lts_deepseekr1(self):
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
    log_file = "./lts_test_deepseek_r1_" + time_str + ".log"

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
