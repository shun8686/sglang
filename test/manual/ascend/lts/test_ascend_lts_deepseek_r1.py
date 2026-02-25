import datetime
import sys
import unittest

from lts_utils import (
    run_bench_serving,
    run_command,
    run_gsm8k,
    run_long_seq_bench_serving,
)

from sglang.test.test_utils import CustomTestCase

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"


class TestLTSDeepSeekR1(CustomTestCase):
    model = MODEL_PATH
    dataset_name = "random"
    dataset_path = (
        "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"  # the path of test dataset
    )
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
        print("res_ttft is " + str(res_ttft))
        print("res_tpot is " + str(res_tpot))
        print("res_output_token_throughput is " + str(res_output_token_throughput))
        print(f"========== 3.5k/1.5k benchmark test PASSED ==========\n")

    def run_gsm8k(self):
        metrics = run_gsm8k(host=f"http://{self.host}", port=self.port)
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )
        print(f"========== gsm8k test PASSED ==========\n")

    def run_all_long_seq_verify(self):
        run_long_seq_bench_serving(
            host=self.host,
            port=self.port,
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
        )

    def test_lts_deepseek_r1(self):
        i = 0
        while True:
            i = i + 1
            time_str_1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={time_str_1}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()
            self.run_all_long_seq_verify()


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_file = "./log/lts_test_deepseek_r1_" + time_str + ".log"

    with open(log_file, "w", encoding="utf-8") as f:
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
