import unittest

from test_ascend_single_mix_utils import TestSingleMixUtils
from test_ascend_qwen3_32b_config import QWEN3_32B_MODEL_PATH, QWEN3_32B_ENVS, QWEN3_32B_OTHER_ARGS

class TestQwen3_32B(TestSingleMixUtils):
    model = QWEN3_32B_MODEL_PATH
    other_args = QWEN3_32B_OTHER_ARGS
    envs = QWEN3_32B_ENVS
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 32
    num_prompts = int(max_concurrency) * 4
    input_len = 1024
    output_len = 300
    random_range_ratio = 0.5
    ttft = 800
    tpot = 30
    output_token_throughput = 1000

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
