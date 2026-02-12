import time
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

    def setUp(self):
        """Execute before each test method"""
        self.base_url = DEFAULT_URL_FOR_TEST
        self.process = None

    def tearDown(self):
        """Execute after each test method to ensure cleanup of processes started by current test"""
        if hasattr(self, 'process') and self.process and self.process.pid:
            try:
                kill_process_tree(self.process.pid)
                self.process = None
            except:
                pass  # Process may have already exited

    def test_enable_torch_compile_debug_mode(self):
        """Test performance difference after enabling torch compile debug mode"""
        # First run: without debug mode
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args,
        )

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        # Run baseline test
        start_time1 = time.perf_counter()
        metrics = run_eval(args)
        end_time1 = time.perf_counter()
        run_gsm8k_time1 = round(end_time1 - start_time1, 6)

        # Clean up first process
        self.tearDown()
        time.sleep(5)

        # Second run: with debug mode enabled
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args + self.enable_args,
        )

        start_time2 = time.perf_counter()
        metrics = run_eval(args)
        end_time2 = time.perf_counter()
        run_gsm8k_time2 = round(end_time2 - start_time2, 6)
        print("run_gsm8k_time1:", run_gsm8k_time1)
        print("run_gsm8k_time2:", run_gsm8k_time2)
        # Assertion: Debug mode should be slower
        self.assertGreater(run_gsm8k_time2, run_gsm8k_time1,
                           f"Debug mode should be slower, but measured time: normal mode={run_gsm8k_time1}s, debug mode={run_gsm8k_time2}s")


if __name__ == "__main__":
    unittest.main()
