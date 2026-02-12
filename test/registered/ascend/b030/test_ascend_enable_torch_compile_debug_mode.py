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
    args_list = [other_args, other_args + enable_args]

    def test_enable_torch_compile_debug_mode(self):
        res_list = []
        for args in self.args_list:
            res = run_bench_serving(
                model=self.model,
                dataset_name="random",
                num_prompts=312,
                random_input_len=3500,
                random_output_len=1500,
                request_rate=float("inf"),
                max_concurrency=78,
                gsp_num_groups=1,
                gsp_prompts_per_group=128,
                gsp_system_prompt_len=1792,
                gsp_question_len=1792,
                gsp_output_len=1,
                other_server_args=args,
            )
            res_list.append(res)
        for res in res_list:
            print(f'output_throughput{res["output_throughput"]}')
            print(f'mean_ttft_ms{res["mean_ttft_ms"]}')
            print(f'mean_tpot_ms{res["mean_tpot_ms"]}')
        self.assertGreater(res_list[0]["output_throughput"], res_list[1]["output_throughput"])
        self.assertGreater(res_list[1]["mean_ttft_ms"], res_list[0]["mean_ttft_ms"])
        self.assertGreater(res_list[1]["mean_tpot_ms"], res_list[0]["mean_tpot_ms"])



if __name__ == "__main__":
    unittest.main()
