import datetime
import os
import sys
import unittest

from lts_utils import TestAscendLtsTestCaseBase

MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"


class TestLTSDeepSeekR1(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
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

    @classmethod
    def setUpClass(cls):
        cls.host = "127.0.0.1"
        cls.port = 6688

    def testLtsDeepSeekR1(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()
            self.run_all_long_seq_verify()


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
