import unittest

from test_ascend_single_mix_utils import TestSingleMixUtils
from test_ascend_qwen3_30b_a3b_config import QWEN3_30B_A3B_ENVS, QWEN3_30B_A3B_MODEL_PATH, QWEN3_30B_A3B_OTHER_ARGS

class TestQwen3_32B(TestSingleMixUtils):
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = QWEN3_30B_A3B_OTHER_ARGS
    envs = QWEN3_30B_A3B_ENVS
    dataset_name = "random"
    max_concurrency = 156
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    ttft = 10000
    tpot = 50
    output_token_throughput = 2400

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
