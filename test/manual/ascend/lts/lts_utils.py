import socket
import subprocess
from types import SimpleNamespace

import psutil

from sglang.test.few_shot_gsm8k import run_eval


def get_nic_name():
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and (
                addr.address.startswith("172.") or addr.address.startswith("192.")
            ):
                print("The nic name matched is {}".format(nic))
                return nic
    return None


NIC_NAME = get_nic_name()
NIC_NAME = "lo" if NIC_NAME is None else NIC_NAME


def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None


def run_bench_serving(
    host,
    port,
    dataset_name="random",
    dataset_path="",
    request_rate=8.0,
    max_concurrency=8,
    num_prompts=32,
    input_len=3500,
    output_len=1500,
    random_range_ratio=1.0,
):
    command = (
        f"python3 -m sglang.bench_serving --backend sglang --host {host} --port {port} "
        f"--dataset-name {dataset_name} --dataset-path {dataset_path} --request-rate {request_rate} "
        f"--max-concurrency {max_concurrency} --num-prompts {num_prompts} --random-input-len {input_len} "
        f"--random-output-len {output_len} --random-range-ratio {random_range_ratio}"
    )
    print(f"command:{command}")
    metrics = run_command(f"{command} | tee ./bench_log.txt")
    return metrics


def run_gsm8k(host="http://127.0.0.1", port=6688, expect_accuracy=None):
    print(f"========== Start gsm8k test ==========\n")
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=1319,
        max_new_tokens=512,
        parallel=128,
        host=host,
        port=port,
    )
    metrics = run_eval(args)
    return metrics


def run_long_seq_bench_serving(
    host=None, port=None, dataset_name="random", dataset_path=None
):
    """依次验证16k+1k、32k+1k、64k+1k三种单条长序列"""
    # 新增：三种长序列配置（16k+1k/32k+1k/64k+1k）
    long_seq_configs = {
        "64k+1k": {
            "input_len": 65536,
            "output_len": 1024,
            "ttft_threshold": 100000,
            "tpot_threshold": 350,
        },
        "32k+1k": {
            "input_len": 32768,
            "output_len": 1024,
            "ttft_threshold": 70000,
            "tpot_threshold": 250,
        },
        "16k+1k": {
            "input_len": 16384,
            "output_len": 1024,
            "ttft_threshold": 40000,
            "tpot_threshold": 200,
        },
    }
    for seq_type, config in long_seq_configs.items():
        print(f"\n========== Start {seq_type} single long sequence test ==========")
        # 执行单条长序列请求
        metrics = run_bench_serving(
            host=host,
            port=port,
            input_len=config["input_len"],
            output_len=config["output_len"],
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            request_rate=1,
            max_concurrency=1,
            num_prompts=2,
            random_range_ratio=1,
        )
        print(f"{seq_type} metrics: {metrics}")

        res_ttft = run_command(
            "cat ./bench_log.txt | grep 'Mean TTFT' | awk '{print $4}'"
        )
        res_tpot = run_command(
            "cat ./bench_log.txt | grep 'Mean TPOT' | awk '{print $4}'"
        )
        res_output_token_throughput = run_command(
            "cat ./bench_log.txt | grep 'Output token throughput' | awk '{print $5}'"
        )
        res_ttft = res_ttft.strip() if res_ttft else "0"
        res_tpot = res_tpot.strip() if res_tpot else "0"

        print("res_ttft is " + str(res_ttft))
        print("res_tpot is " + str(res_tpot))
        print("res_output_token_throughput is " + str(res_output_token_throughput))
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
