import logging
import unittest
import time
import threading
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=3600, suite="stage-b-test-npu")


class TestHierarchicalCacheNPU(CustomTestCase):
    """Test Hierarchical Cache functionality on NPU environment.

    [Test Category] Functional
    [Test Target] Hierarchical Cache on NPU
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    test_prompt = "What is the capital of France?"
    expected_output = "Paris"

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _launch_server_with_hicache(
        self,
        hicache_ratio=2.0,
        hicache_size=0,
        hicache_write_policy="write_through",
        radix_eviction_policy="lru",
        hicache_io_backend="kernel",
        hicache_mem_layout="layer_first",
        disable_hicache_numa_detect=False,
        hicache_storage_backend=None,
    ):
        """Launch server with hierarchical cache parameters."""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            str(hicache_ratio),
            "--hicache-write-policy",
            hicache_write_policy,
            "--radix-eviction-policy",
            radix_eviction_policy,
            "--hicache-io-backend",
            hicache_io_backend,
            "--hicache-mem-layout",
            hicache_mem_layout,
        ]

        if hicache_size > 0:
            other_args.extend([
                "--hicache-size",
                str(hicache_size),
            ])

        if disable_hicache_numa_detect:
            other_args.append("--disable-hicache-numa-detect")

        if hicache_storage_backend is not None:
            other_args.extend([
                "--hicache-storage-backend",
                hicache_storage_backend,
            ])

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return process

    def _test_basic_inference(self):
        """Test basic inference functionality."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.expected_output, response.text)
        return response.text

    def _send_concurrent_requests(self, num_requests=10):
        """Send concurrent requests and return results."""
        results = []
        threads = []

        def send_request(rid):
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": f"Test request {rid}: What is AI?",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                        },
                    },
                    timeout=120,
                )
                results.append((rid, response.status_code, response.text))
            except Exception as e:
                results.append((rid, -1, str(e)))

        for i in range(num_requests):
            thread = threading.Thread(target=send_request, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def test_001_combined_params(self):
        """Test Hicache with combined parameters,hicache inference request reuse succeddfully."""
        logging.warning("\n=== Test 001: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=1.0,
            hicache_write_policy="write_back",
            radix_eviction_policy="lru",
            hicache_io_backend="direct",
            hicache_mem_layout="layer_first",
        )

        try:
            time.sleep(5)
            prompt = "What is The capital of France?What is The capital of France?What is The capital of France?" * 18
            for i in range(2):
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 10,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                # If the same request is made, the token will be reused.
                # cached_tokens: Number of tokens cached in KV Cache
                if i == 0:
                    self.assertEqual(int(response.json()["meta_info"]["cached_tokens"]), 0)
                else:
                    self.assertGreater(
                        int(response.json()["meta_info"]["cached_tokens"]), 0
                    )
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_002_combined_params(self):
        """Test Hicache with combined parameters, hicache_storage_backend is configured to be file, file storage is hosted under hicache."""
        logging.warning("\n=== Test 002: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=2.0,
            hicache_write_policy="write_through",
            radix_eviction_policy="lfu",
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
            hicache_storage_backend="file",
            disable_hicache_numa_detect=True,
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            logging.warning(f"Combined parameters test passed, result: {result[:50]}...")

            args = SimpleNamespace(
                num_shots=5,
                data_path="/tmp/test.jsonl",
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=21000,
            )
            run_eval(args)
            hicache_file = run_command(f"ls /tmp/hicache")
            self.assertNotEqual(hicache_file, None)
            run_command(f"rm -rf /tmp/hicache")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_003_combined_params(self):
        """Test Hicache with combined parameters, hicache with concurrent requests."""
        logging.warning("\n=== Test 003: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_size=80,
            hicache_write_policy="write_through_selective",
            hicache_io_backend="kernel_ascend",
            hicache_mem_layout="page_first_direct",
        )

        try:
            time.sleep(5)
            results = self._send_concurrent_requests(num_requests=20)

            success_count = sum(1 for r in results if r[1] == 200)
            self.assertGreaterEqual(success_count, 18)
            logging.warning(f"Concurrent requests test passed, {success_count}/20 succeeded.")

        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_004_combined_params(self):
        """Test Hicache with combined parameters, hicache with long sequence"""
        logging.warning("\n=== Test 004: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_size=100,
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_kv_split",
            disable_hicache_numa_detect=True,
        )

        try:
            time.sleep(5)
            long_prompt = "Explain the concept of machine learning in detail. " * 10
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": long_prompt,
                    "sampling_params": {
                        "temperature": 0.7,
                        "max_new_tokens": 128,
                    },
                },
                timeout=180,
            )
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.text), 50)
            logging.warning(f"Long sequence test passed, result length: {len(response.text)}")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


if __name__ == "__main__":
    unittest.main()
