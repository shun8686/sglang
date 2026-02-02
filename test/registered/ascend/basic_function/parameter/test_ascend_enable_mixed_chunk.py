# -*- coding: utf-8 -*-
import unittest
import requests
import time
import threading
from datetime import datetime

# 必要模块导入
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 全局配置（统一配置，保证对比公平性）
CONFIG = {
    "TARGET_TOKEN_COUNT": 2500,  # 超长输入token数（>2048）
    "CHUNK_SIZE": 1024,          # 分块预填充大小
    "REQUEST_COUNT": 50,         # 调整：请求数量改为50个
    "TIMEOUT": 600               # 延长超时，适配50个请求的高并发场景
    # 已删除：SERVER_WAIT_TIME 和 PROCESS_KILL_WAIT_TIME
}

# 全局变量：存储两份测试的最终统计结果（用于后续对比和断言）
FINAL_STATISTICS = {
    "mixed_enabled": None,
    "mixed_disabled": None
}

def build_long_input_text_for_token():
    """Construct long input text with enough tokens (common function for consistent input)"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (CONFIG["TARGET_TOKEN_COUNT"] // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

def send_generate_request(task_id, request_results):
    """Single request sending function (record single request elapsed time)"""
    try:
        # 1. Record single request start time
        single_start_time = time.time()
        
        # 2. Construct long input text
        long_input_text = build_long_input_text_for_token()
        
        # 3. Send /generate request
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=CONFIG["TIMEOUT"]
        )
        
        # 4. Record single request end time and calculate elapsed time
        single_end_time = time.time()
        single_elapsed_time = round(single_end_time - single_start_time, 4)
        
        # 5. Record request result
        request_results.append({
            "task_id": task_id,
            "status_code": response.status_code,
            "has_paris": "Paris" in response.text,
            "single_elapsed_time": single_elapsed_time
        })
        
        print(f"[Task {task_id}] Request completed, status code: {response.status_code}, elapsed time: {single_elapsed_time} seconds")
    except Exception as e:
        single_end_time = time.time()
        single_elapsed_time = round(single_end_time - time.time(), 4)
        request_results.append({
            "task_id": task_id,
            "status_code": -1,
            "error": str(e),
            "single_elapsed_time": single_elapsed_time
        })
        print(f"[Task {task_id}] Request failed, error: {e}, elapsed time: {single_elapsed_time} seconds")

def start_server(with_mixed: bool):
    """Common server start function: enable mixed chunk based on parameter"""
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chunked-prefill-size", str(CONFIG["CHUNK_SIZE"])
    ]
    
    # Add --enable-mixed-chunk parameter if needed
    if with_mixed:
        other_args.insert(0, "--enable-mixed-chunk")
    
    # Start server
    process = popen_launch_server(
        LLAMA_3_2_1B_WEIGHTS_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    
    # 直接使用 0 秒等待，删除配置变量依赖
    time.sleep(0)
    return process

def calculate_statistics(request_results):
    """Common statistics function: calculate average, max, min elapsed time (remove overall elapsed time)"""
    success_requests = [r for r in request_results if r["status_code"] == 200]
    if not success_requests:
        return None
    
    # Extract elapsed time of successful requests
    elapsed_times = [r["single_elapsed_time"] for r in success_requests]
    
    # Calculate statistical indicators (remove overall elapsed time)
    return {
        "success_count": len(success_requests),
        "total_count": len(request_results),
        "avg_elapsed": round(sum(elapsed_times) / len(elapsed_times), 4),
        "max_elapsed": round(max(elapsed_times), 4),
        "min_elapsed": round(min(elapsed_times), 4)
    }

class TestMixedChunkEnabled(CustomTestCase):
    """Test Class 1: Enable --enable-mixed-chunk"""
    @classmethod
    def setUpClass(cls):
        """Start server (with mixed chunk enabled)"""
        print("\n" + "="*60)
        print("=== Starting Server (--enable-mixed-chunk ENABLED) ===")
        cls.process = start_server(with_mixed=True)

    @classmethod
    def tearDownClass(cls):
        """Stop server and wait 10 seconds (直接使用 10 秒，删除配置变量依赖)"""
        kill_process_tree(cls.process.pid)
        print("=== Server Stopped (--enable-mixed-chunk ENABLED) ===")
        print(f"=== Waiting 10 seconds after process termination ===")
        # 直接使用 10 秒等待，删除配置变量依赖
        time.sleep(10)
        print("="*60 + "\n")

class TestMixedChunkDisabled(CustomTestCase):
    """Test Class 2: Disable --enable-mixed-chunk"""
    @classmethod
    def setUpClass(cls):
        """Start server (with mixed chunk disabled)"""
        print("\n" + "="*60)
        print("=== Starting Server (--enable-mixed-chunk DISABLED) ===")
        cls.process = start_server(with_mixed=False)

    @classmethod
    def tearDownClass(cls):
        """Stop server and wait 10 seconds (直接使用 10 秒，删除配置变量依赖)"""
        kill_process_tree(cls.process.pid)
        print("=== Server Stopped (--enable-mixed-chunk DISABLED) ===")
        print(f"=== Waiting 10 seconds after process termination ===")
        # 直接使用 10 秒等待，删除配置变量依赖
        time.sleep(10)
        print("="*60 + "\n")

    def test_mixed_chunk_with_multi_requests(self):
        """Multi-request test (mixed chunk disabled), collect statistics (remove overall elapsed time)"""
        request_results = []
        
        # Print test start information
        print(f"\n=== [Mixed Chunk DISABLED] Test started, timestamp: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== Starting {CONFIG['REQUEST_COUNT']} request threads to create queue scenario ===")
        
        # Create and start multiple threads
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Calculate statistics (remove overall elapsed time calculation)
        statistics = calculate_statistics(request_results)
        
        # Store final statistics
        FINAL_STATISTICS["mixed_disabled"] = {
            "detail": statistics
        }
        
        # Print current test result
        self._print_test_result("Mixed Chunk DISABLED", request_results, statistics)
        
        # Add assertGreater after both tests are completed (verify optimization effect)
        self._run_performance_assertions()

    def _print_test_result(self, test_name, request_results, statistics):
        """Helper function: Format and print test results (in English)"""
        print(f"\n=== [{test_name}] All requests completed ===")
        
        if not statistics:
            print(f"=== [{test_name}] No successful requests, cannot calculate detailed statistics ===")
            return
        
        print(f"=== [{test_name}] Statistics Summary ===")
        print(f"  Successful requests: {statistics['success_count']} / {statistics['total_count']}")
        print(f"  Average elapsed time per request: {statistics['avg_elapsed']} seconds")
        print(f"  Maximum elapsed time per request: {statistics['max_elapsed']} seconds")
        print(f"  Minimum elapsed time per request: {statistics['min_elapsed']} seconds")

    def _run_performance_assertions(self):
        """Add assertGreater assertions to verify performance optimization (core adjustment)"""
        print("\n" + "="*80)
        print("=== Running Performance Assertions ===")
        print("="*80)
        
        enabled = FINAL_STATISTICS["mixed_enabled"]
        disabled = FINAL_STATISTICS["mixed_disabled"]
        
        # Verify if statistics data is complete
        if not enabled or not disabled or not enabled["detail"] or not disabled["detail"]:
            self.fail("Assertion Failed: Incomplete test data, cannot perform performance assertions")
            return
        
        # Extract core statistical data
        enabled_avg = enabled["detail"]["avg_elapsed"]
        disabled_avg = disabled["detail"]["avg_elapsed"]
        enabled_max = enabled["detail"]["max_elapsed"]
        disabled_max = disabled["detail"]["max_elapsed"]
        
        # Calculate optimization rate (for reference only)
        avg_optimize_rate = round(((disabled_avg - enabled_avg) / disabled_avg) * 100, 2)
        max_optimize_rate = round(((disabled_max - enabled_max) / disabled_max) * 100, 2)
        
        # Print assertion preparation information
        print(f"\n1. Average Elapsed Time Comparison")
        print(f"   Mixed Enabled: {enabled_avg}s | Mixed Disabled: {disabled_avg}s | Optimization Rate: {avg_optimize_rate}%")
        print(f"\n2. Maximum Elapsed Time Comparison (Long-tail Effect)")
        print(f"   Mixed Enabled: {enabled_max}s | Mixed Disabled: {disabled_max}s | Optimization Rate: {max_optimize_rate}%")
        
        # Core assertion: assertGreater (verify that disabled time > enabled time, i.e., mixed chunk has optimization effect)
        # Assertion 1: Average elapsed time (disabled > enabled → mixed chunk optimizes average time)
        try:
            self.assertGreater(disabled_avg, enabled_avg, 
                               f"Assertion Failed: Average elapsed time - Mixed Disabled ({disabled_avg}s) is not greater than Mixed Enabled ({enabled_avg}s)")
            print("\n✅ Assertion Passed: Average Elapsed Time Optimization Verified")
        except AssertionError as e:
            print(f"\n❌ {e}")
            raise
        
        # Assertion 2: Maximum elapsed time (disabled > enabled → mixed chunk alleviates long-tail effect)
        try:
            self.assertGreater(disabled_max, enabled_max, 
                               f"Assertion Failed: Maximum elapsed time - Mixed Disabled ({disabled_max}s) is not greater than Mixed Enabled ({enabled_max}s)")
            print("✅ Assertion Passed: Long-tail Effect Alleviation Verified")
        except AssertionError as e:
            print(f"\n❌ {e}")
            raise
        
        print("\n=== All Performance Assertions Passed Successfully ===")

if __name__ == "__main__":
    # Execute all test cases with detailed output
    unittest.main(verbosity=2, exit=False)
