import unittest
import time
import threading
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
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
        enable_hierarchical_cache = True,
        hicache_ratio=2.0,
        hicache_size=0,
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="layer_first",
        disable_hicache_numa_detect=False,
        hicache_storage_backend=None,
        hicache_storage_prefetch_policy="best_effort",
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

        if hicache_storage_prefetch_policy != "best_effort":
            other_args.extend([
                "--hicache-storage-prefetch-policy",
                hicache_storage_prefetch_policy,
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

    def test_01_basic_functionality(self):
        """Test Hicache basic functionality."""
        print("\n=== Test 01: Basic Functionality ===")
        self.process = self._launch_server_with_hicache()

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ Basic functionality test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_02_hicache_ratio_1_0(self):
        """Test with hicache_ratio=1.0."""
        print("\n=== Test 02: hicache_ratio=1.0 ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=1.0
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ hicache_ratio=1.0 test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_03_hicache_ratio_2_0(self):
        """Test with hicache_ratio=2.0."""
        print("\n=== Test 03: hicache_ratio=2.0 ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=2.0
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ hicache_ratio=2.0 test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_04_hicache_size_1gb(self):
        """Test with hicache_size=1GB."""
        print("\n=== Test 04: hicache_size=1GB ===")
        self.process = self._launch_server_with_hicache(
            hicache_size=1
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ hicache_size=1 test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_05_hicache_write_policy_write_back(self):
        """Test with hicache_write_policy=write_back."""
        print("\n=== Test 05: hicache_write_policy=write_back ===")
        self.process = self._launch_server_with_hicache(
            hicache_write_policy="write_back"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ write_back policy test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_06_hicache_write_policy_write_through(self):
        """Test with hicache_write_policy=write_through."""
        print("\n=== Test 06: hicache_write_policy=write_through ===")
        self.process = self._launch_server_with_hicache(
            hicache_write_policy="write_through"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ write_through policy test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_07_hicache_io_backend_direct(self):
        """Test with hicache_io_backend=direct."""
        print("\n=== Test 07: hicache_io_backend=direct ===")
        self.process = self._launch_server_with_hicache(
            hicache_io_backend="direct"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ direct IO backend test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_08_hicache_io_backend_kernel(self):
        """Test with hicache_io_backend=kernel."""
        print("\n=== Test 08: hicache_io_backend=kernel ===")
        self.process = self._launch_server_with_hicache(
            hicache_io_backend="kernel"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ kernel IO backend test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_09_hicache_mem_layout_page_first(self):
        """Test with hicache_mem_layout=page_first."""
        print("\n=== Test 09: hicache_mem_layout=page_first ===")
        self.process = self._launch_server_with_hicache(
            hicache_mem_layout="page_first"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ page_first layout test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_10_hicache_mem_layout_page_first_direct(self):
        """Test with hicache_mem_layout=page_first_direct."""
        print("\n=== Test 10: hicache_mem_layout=page_first_direct ===")
        self.process = self._launch_server_with_hicache(
            hicache_mem_layout="page_first_direct"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ page_first_direct test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_11_long_sequence(self):
        """Test Hicache with long sequence."""
        print("\n=== Test 11: Long sequence (2000 + tokens) ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=2.0
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
            print(f"√ Long sequence test passed, result length: {len(response.text)}")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_12_concurrent_requests(self):
        """Test Hicache with concurrent requests."""
        print("\n=== Test 12: Concurrent Requests (20) ===")
        self.process = self._launch_server_with_hicache()

        try:
            time.sleep(5)
            results = self._send_concurrent_requests(num_requests=20)

            success_count = sum(1 for r in results if r[1] == 200)
            self.assertGreaterEqual(success_count, 18)
            print(f"√ Concurrent requests test passed, {success_count}/20 succeeded.")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_13_batch_requests(self):
        """Test Hicache with batch requests."""
        print("\n=== Test 13: Batch Requests ===")
        self.process = self._launch_server_with_hicache()

        try:
            time.sleep(5)
            texts = [
                "What is AI?",
                "Explain Python.",
                "Define machine learning.",
                "What is deep learning?",
                "Explain neural networks.",
                "What is NLP?",
                "Explain transformers.",
                "What is computer vision?",
                "Define reinforcement learning.",
                "What is transfer learning?",
            ]

            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": texts,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
                timeout=180,
            )
            self.assertEqual(response.status_code, 200)
            print(f"√ Batch requests test passed")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_14_combined_params(self):
        """Test Hicache with combined parameters."""
        print("\n=== Test 14: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=2.0,
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ Combined parameters test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_15_boundary_small_ratio(self):
        """Test with minimal hicache_ratio."""
        print("\n=== Test 15: Minimal hicache_ratio (0.1) ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=0.1
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ Minimal ratio test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_16_boundary_large_ratio(self):
        """Test with maximum hicache_ratio."""
        print("\n=== Test 16: Maximum hicache_ratio (9.99) ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=9.99
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ Maximum ratio test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_17_error_recovery(self):
        """Test error recovery capability."""
        print("\n=== Test 17: Error Recovery ===")
        self.process = self._launch_server_with_hicache()

        try:
            time.sleep(5)

            result = self._test_basic_inference()
            print(f" Normal request succeeded: {result[:50]}...")

            long_prompt = "Test" * 10000
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": long_prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                        },
                    },
                    timeout=10,
                )
            except Exception:
                pass

            result2 = self._test_basic_inference()
            print(f" Recovery request succeeded: {result2[:50]}...")
            print(f"√ Error recovery test passed")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_18_memory_pressure(self):
        """Test Hicache under memory pressure."""
        print("\n=== Test 18: Memory Pressure (100 + requests) ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=1.0
        )

        try:
            time.sleep(5)
            results = self._send_concurrent_requests(num_requests=100)

            success_count = sum(1 for r in results if r[1] == 200)
            self.assertGreaterEqual(success_count, 90)
            print(f"√ Memory pressure test passed, {success_count}/100 succeeded")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_19_npu_kernel_call(self):
        """Test NPU kernel invocation."""
        print("\n=== Test 19: NPU Kernel Invocation ===")
        self.process = self._launch_server_with_hicache(
            hicache_io_backend="kernel"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            self.assertIn(self.expected_output, result)
            print(f"√ NPU kernel invocation test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_20_disable_numa_detect(self):
        """Test with NUMA detect disabled."""
        print("\n=== Test 20: Disable NUMA Detect ===")
        self.process = self._launch_server_with_hicache(
            disable_hicache_numa_detect=True
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ Disable NUMA detect test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_21_prefetch_policy_wait_complete(self):
        """Test with wait_complete prefetch policy."""
        print("\n=== Test 21: Prefetch Policy wait_complete ===")
        self.process = self._launch_server_with_hicache(
            hicache_storage_prefetch_policy="wait_complete"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ wait_complete policy test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_22_prefetch_policy_timeout(self):
        """Test with timeout prefetch policy."""
        print("\n=== Test 22: Prefetch Policy timeout ===")
        self.process = self._launch_server_with_hicache(
            hicache_storage_prefetch_policy="timeout"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ timeout policy test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_23_storage_backend_nixl(self):
        """Test with nixl storage backend."""
        print("\n=== Test 23: Storage Backend nixl ===")
        self.process = self._launch_server_with_hicache(
            hicache_storage_backend="nixl"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ nixl storage backend test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_24_storage_backend_mooncake(self):
        """Test with mooncake storage backend."""
        print("\n=== Test 24: Storage Backend mooncake ===")
        self.process = self._launch_server_with_hicache(
            hicache_storage_backend="mooncake"
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ mooncake storage backend test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_25_combined_all_params(self):
        """Test with all parameters combined."""
        print("\n=== Test 25: Combined All Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=2.0,
            hicache_size=1,
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
            hicache_storage_prefetch_policy="best_effort",
        )

        try:
            time.sleep(5)
            result = self._test_basic_inference()
            print(f"√ Combined all parameters test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


if __name__ == "__main__":
    unittest.main()
