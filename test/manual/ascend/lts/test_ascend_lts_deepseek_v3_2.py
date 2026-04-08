import datetime
import os
import sys
import unittest

from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"


class TestLTSDeepSeekV32(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
    input_len = 512
    output_len = 512
    random_range_ratio = 1
    ttft = 10000
    tpot = 100
    output_token_throughput = 5000
    accuracy = {"gsm8k": 0.80, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "127.0.0.1"
        cls.port = 6688

    def testLtsDeepseekV32(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()
            self.run_long_seq_testcase()
            long_seq_configs = {
                "128k+1k": {
                    "input_len": 131072,
                    "output_len": 1024,
                    "max_concurrency": 8,
                    "num_prompts": 8,
                    "ttft": 100000,
                    "tpot": 350,
                    "tps": 10,
                }
            }
            self.run_long_seq_testcase(long_seq_configs=long_seq_configs)


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs("log", exist_ok=True)
    log_file = (
        f"./log/lts_{os.path.splitext(os.path.basename(__file__))[0]}_{time_str}.log"
    )

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
