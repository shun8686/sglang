import os
import time
import unittest
import subprocess

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import MINICPM_O_2_6_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=500,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled=False,
)


class TestAscendCudaGraphGC(unittest.TestCase):
    """Testcase: Verify the function of --enable-cudagraph-gc parameter.

    [Test Category] Parameter
    [Test Target] --enable-cudagraph-gc
    """

    model = MINICPM_O_2_6_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    log_file = "./cudagraph_gc_log.txt"

    @classmethod
    def setUpClass(cls):
        cls.time_records = {}

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.log_file):
            os.remove(cls.log_file)

    def _launch_and_measure_time(self, enable_cudagraph_gc: bool):
        """Launch server with/without --enable-cudagraph-gc and measure startup time."""
        extra_args = [
            "--trust-remote-code",
            "--tp-size", "1",
            "--mem-fraction-static", "0.7",
            "--attention-backend", "ascend",
            "--cuda-graph-max-bs",
            "16",
        ]
        if enable_cudagraph_gc:
            extra_args.append("--enable-cudagraph-gc")

        # Start server and log
        with open(self.log_file, "w", encoding="utf-8") as f:
            start = time.time()
            proc = popen_launch_server(
                self.model,
                self.base_url,
                timeout=3600,
                other_args=extra_args,
                return_stdout_stderr=(f, f),
            )

            # Wait for CUDA graph capture
            time.sleep(45)

        # Record time
        elapsed = time.time() - start
        kill_process_tree(proc.pid)
        time.sleep(10)
        return elapsed

    def _check_cuda_graph_log(self):
        """Check if CUDA graph is captured in log."""
        if not os.path.exists(self.log_file):
            return False
        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()
        return "Capturing batches" in content

    def test_enable_cudagraph_gc_performance(self):
        """Test that enabling cudagraph-gc slows down CUDA graph capture (GC not frozen)."""

        # Test 1: default (disable --enable-cudagraph-gc, GC frozen, fast)
        time_off = self._launch_and_measure_time(enable_cudagraph_gc=False)
        self.assertTrue(self._check_cuda_graph_log(), "CUDA graph not captured")

        # Test 2: enable --enable-cudagraph-gc (GC not frozen, slow)
        time_on = self._launch_and_measure_time(enable_cudagraph_gc=True)
        self.assertTrue(self._check_cuda_graph_log(), "CUDA graph not captured")

        # Record
        self.time_records["gc_disabled"] = time_off
        self.time_records["gc_enabled"] = time_on

        # Verify: gc enabled → slower startup
        self.assertGreater(
            time_on, time_off,
            f"Enable cudagraph-gc should be slower. Off: {time_off:.2f}s, On: {time_on:.2f}s"
        )

        # Output result
        print("\n" + "=" * 60)
        print("             CUDA Graph GC Test Result")
        print("=" * 60)
        print(f"GC disabled (default): {time_off:.2f}s")
        print(f"GC enabled: {time_on:.2f}s")
        print(f"Slowdown: {time_on - time_off:.2f}s")
        print("=" * 60)

    def test_server_healthy(self):
        """Verify server can start and generate normally."""
        extra_args = [
            "--trust-remote-code",
            "--tp-size", "1",
            "--mem-fraction-static", "0.7",
            "--attention-backend", "ascend",
            "--enable-cudagraph-gc",
        ]

        with open(self.log_file, "w", encoding="utf-8") as f:
            proc = popen_launch_server(
                self.model,
                self.base_url,
                timeout=3600,
                other_args=extra_args,
                return_stdout_stderr=(f, f),
            )
            time.sleep(45)

        try:
            # Check server info
            resp = requests.get(f"{self.base_url}/get_server_info", timeout=10)
            self.assertEqual(resp.status_code, 200)

            # Check generate
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Paris", resp.text)
        finally:
            kill_process_tree(proc.pid)
            time.sleep(10)


if __name__ == "__main__":
    unittest.main()