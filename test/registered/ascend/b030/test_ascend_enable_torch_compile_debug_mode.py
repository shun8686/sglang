import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH, run_bench_serving
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server, CustomTestCase,
)


class TestEnableTorchCompileDebugMode(CustomTestCase):
    model = QWEN3_32B_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "4",
    ]
    enable_args = [
        "--enable-torch-compile-debug-mode",
    ]

    def test_enable_torch_compile_debug_mode(self):
        res1 = run_bench_serving(
            model=self.model,
            dataset_name="random",
            num_prompts=128,
            random_input_len=3584,
            random_output_len=1,
            request_rate=float("inf"),
            max_concurrency=16,
            gsp_num_groups=1,
            gsp_prompts_per_group=128,
            gsp_system_prompt_len=1792,
            gsp_question_len=1792,
            gsp_output_len=1,
            other_server_args=self.other_args,
        )
        mean_ttft_ms1 = res1["mean_ttft_ms"]
        mean_tpot_ms1 = res1["mean_tpot_ms"]
        output_throughput1 = res1["output_throughput"]
        res2 = run_bench_serving(
            model=self.model,
            dataset_name="random",
            num_prompts=128,
            random_input_len=3584,
            random_output_len=1,
            request_rate=float("inf"),
            max_concurrency=16,
            gsp_num_groups=1,
            gsp_prompts_per_group=128,
            gsp_system_prompt_len=1792,
            gsp_question_len=1792,
            gsp_output_len=1,
            other_server_args=self.other_args + self.enable_args,
        )
        mean_ttft_ms2 = res2["mean_ttft_ms"]
        mean_tpot_ms2 = res2["mean_tpot_ms"]
        output_throughput2 = res2["output_throughput"]
        print(1111111111111111111111111111111111111111111111111)
        print(mean_ttft_ms1, mean_tpot_ms1, output_throughput1)
        print(2222222222222222222222222222222222222222222222222)
        print(mean_ttft_ms2, mean_tpot_ms2, output_throughput2)



if __name__ == "__main__":
    unittest.main()
