import unittest

from test_ascend_single_mix_utils import TestSingleMixUtils
from test_ascend_qwen3_next_80b_a3b_config import Qwen3_Next_80B_A3B_MODEL_PATH, Qwen3_Next_80B_A3B_ENVS, Qwen3_Next_80B_A3B_OTHER_ARGS

class TestQwen3_Next_80B_A3B(TestSingleMixUtils):
    model = Qwen3_Next_80B_A3B_MODEL_PATH
    other_args = Qwen3_Next_80B_A3B_OTHER_ARGS
    envs = Qwen3_Next_80B_A3B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 8
    input_len = 1000
    output_len = 300
    random_range_ratio = 0.5
    ttft = 5000
    tpot = 50
    output_token_throughput = 300

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
