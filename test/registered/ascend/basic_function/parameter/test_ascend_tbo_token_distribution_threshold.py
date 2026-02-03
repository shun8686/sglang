import unittest
import requests
import time
import threading
from datetime import datetime
from typing import Dict, Any

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

CONFIG = {
    "REQUEST_COUNT": 30,
    "TARGET_TOKEN_COUNT": 512,
    "TIMEOUT": 600,
    "MAX_NEW_TOKENS": 128
}

FINAL_STATISTICS: Dict[str, Dict[str, Any]] = {
    "tbo_enabled_0.8": {},
    "tbo_disabled_0": {}
}

def build_long_input_text_for_token():
    """Construct long input text with enough tokens (common function for consistent input)"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (CONFIG["TARGET_TOKEN_COUNT"] // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

def send_generate_request(task_id, request_results):
    single_start_time = time.time()
    
    request_text = "The capital of France is"

    long_input_text = build_long_input_text_for_token()
    
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
    
    single_elapsed_time = round(time.time() - single_start_time, 4)

    request_results.append({
        "task_id": task_id,
        "status_code": response.status_code,
        "single_elapsed_time": single_elapsed_time
    })
    
    print(f"[Task {task_id}] Request completed, status code: {response.status_code}, elapsed time: {single_elapsed_time} seconds")

def start_tbo_server(tbo_threshold: float):
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tbo-token-distribution-threshold",
        str(tbo_threshold)
    ]
    
    process = popen_launch_server(
        LLAMA_3_2_1B_WEIGHTS_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )

    return process

def calculate_statistics(request_results):
    success_requests = [r for r in request_results if r["status_code"] == 200]
    if not success_requests:
        return None

    elapsed_times = [r["single_elapsed_time"] for r in success_requests]
    avg_elapsed = round(sum(elapsed_times) / len(elapsed_times), 4)
    
    return {
        "success_count": len(success_requests),
        "total_count": len(request_results),
        "avg_elapsed": avg_elapsed
    }

class TestTbo08(CustomTestCase):
    """Testcase: Verify TBO performance with threshold 0.8 (enabled auto-switch)

    [Test Category] Parameter
    [Test Target] --tbo-token-distribution-threshold;
    """
    @classmethod
    def setUpClass(cls):
        print("=== Starting Server (--tbo-token-distribution-threshold=0.8 ENABLED) ===")
        cls.process = start_tbo_server(tbo_threshold=0.8)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(10)

    def test_tbo_with_multi_requests(self):
        request_results = []
        
        # Print test start information
        print(f"\n=== [TBO 0.8 ENABLED] Test started, timestamp: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== Starting {CONFIG['REQUEST_COUNT']} request threads to create queue scenario ===")
        
        # Create and start multiple threads
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Calculate statistics
        statistics = calculate_statistics(request_results)
        
        FINAL_STATISTICS["tbo_enabled_0.8"] = {
            "detail": statistics
        }
        
        print(f"  Average elapsed time per request: {statistics['avg_elapsed']} seconds")

class TestTboDisabled(CustomTestCase):
    """Testcase: Verify TBO performance with threshold 0 (disabled two-chunk-overlap)

    [Test Category] Parameter
    [Test Target] --tbo-token-distribution-threshold;
    """
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("=== Starting Server (--tbo-token-distribution-threshold=0 DISABLED) ===")
        cls.process = start_tbo_server(tbo_threshold=0)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(10)

    def test_tbo_with_multi_requests(self):
        request_results = []
        
        print(f"\n=== [TBO 0 DISABLED] Test started, timestamp: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== Starting {CONFIG['REQUEST_COUNT']} request threads to create queue scenario ===")
        
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        statistics = calculate_statistics(request_results)
        
        FINAL_STATISTICS["tbo_disabled_0"] = {
            "detail": statistics
        }

        print(f"  Average elapsed time per request: {statistics['avg_elapsed']} seconds")

        self._run_performance_assertions()

    def _run_performance_assertions(self):
        # verify performance optimization
        print("\n=== Running Performance Assertions===")
        
        # Extract core statistical data
        tbo_08_stats = FINAL_STATISTICS["tbo_enabled_0.8"]
        tbo_0_stats = FINAL_STATISTICS["tbo_disabled_0"]

        tbo_08_avg = tbo_08_stats["detail"]["avg_elapsed"]
        tbo_0_avg = tbo_0_stats["detail"]["avg_elapsed"]
        
        
        print(f"Average Elapsed Time Comparison")
        print(f"   TBO 0.8 Enabled: {tbo_08_avg}s | TBO 0 Disabled: {tbo_0_avg}s ")
 
        self.assertGreater(
            tbo_08_avg, tbo_0_avg, 
            f"Assertion Failed: Average elapsed time - TBO 0 ({tbo_0_avg}s) is not lesser than TBO 0.8 ({tbo_08_avg}s)"
        )
        print("\n Assertion Passed: TBO 0 Average Latency < TBO 0.8 Average Latency")


if __name__ == "__main__":
    unittest.main()
