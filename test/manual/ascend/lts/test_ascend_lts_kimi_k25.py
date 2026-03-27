import datetime
import os
import sys
import unittest

from lts_utils import TestAscendLtsTestCaseBase

MODEL_PATH = "/root/.cache/modelscope/hub/models/Eco-Tech/Kimi-K2.5-w4a8"


class TestLTSKimi(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    request_rate = None
    max_concurrency = 96
    num_prompts = 96
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    warmup_requests = 16
    image_resolution = None
    image_count = None
    seed = None
    ttft = 15000
    tpot = 55
    output_token_throughput = 1350
    accuracy = {"gsm8k": 0.90, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "127.0.0.1"
        cls.port = 8100
        cls.base_url = f"http://{cls.host}:{cls.port}"

    def test_lts_kimi_k25(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n=============={current_time}  Execute the {i}-th long-term stability test=============="
            )
            self.run_throughput()
            self.run_gsm8k()


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
