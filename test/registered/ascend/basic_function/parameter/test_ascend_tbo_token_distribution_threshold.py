import unittest
import requests
import threading
import time
from datetime import datetime

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from typing import Dict, Any
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

CONCURRENT_CONFIG = {
    "REQUEST_COUNT": 50,
    "TIMEOUT": 600,
    "MAX_NEW_TOKENS": 32
}


TBO_LATENCY_STATISTICS: Dict[str, Dict[str, Any]] = {
    "tbo_enabled_0.8": {},
    "tbo_disabled_0": {}
}


def send_concurrent_request(task_id, request_results):
    start_time = time.time()
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": CONCURRENT_CONFIG["MAX_NEW_TOKENS"],
            },
        },
        timeout=CONCURRENT_CONFIG["TIMEOUT"]
    )
    request_results.append({
        "task_id": task_id,
        "status_code": response.status_code,
        "elapsed_time": round(time.time() - start_time, 4)
    })
    print(
        f"[Task {task_id}] Completed, status: {response.status_code}, elapsed: {request_results[-1]['elapsed_time']}s")



def calculate_latency_stats(request_results):
    total_count = len(request_results)
    success_requests = [r for r in request_results if r["status_code"] == 200]
    success_count = len(success_requests)

    if not success_requests:
        return None

    # Extract elapsed time of successful requests
    elapsed_times = [r["elapsed_time"] for r in success_requests]

    # Calculate statistical indicators
    return {
        "total_count": total_count,
        "success_count": success_count,
        "success_rate": round((success_count / total_count) * 100, 2),
        "avg_elapsed": round(sum(elapsed_times) / success_count, 4),
        "max_elapsed": round(max(elapsed_times), 4),
        "min_elapsed": round(min(elapsed_times), 4)
    }


class TestTboTokenDistributionThresholdBase8(CustomTestCase):
    """Testcase: Verify the baseline performance of --tbo-token-distribution-threshold with 50 concurrent long-token requests (TARGET_TOKEN_COUNT=2500).

    [Test Category] Parameter
    [Test Target] --tbo-token-distribution-threshold;
    """
    tbo_token_distribution_threshold = 0.8
    tbo_type = "tbo_enabled_0.8"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tbo-token-distribution-threshold",
            str(cls.tbo_token_distribution_threshold),
        ]

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(10)

    def test_tbo_concurrent_latency(self):
        request_results = []
        print("\n" + "=" * 60)
        print(f"=== Starting TBO Concurrent Test ({self.tbo_type}) ===")
        print(
            f"=== Threshold: {self.tbo_token_distribution_threshold} | Concurrent: {CONCURRENT_CONFIG['REQUEST_COUNT']} ===")
        print(f"=== Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print("=" * 60)

        # Create and start multiple threads
        threads = [
            threading.Thread(target=send_concurrent_request, args=(i, request_results))
            for i in range(CONCURRENT_CONFIG["REQUEST_COUNT"])
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]

        latency_stats = calculate_latency_stats(request_results)
        TBO_LATENCY_STATISTICS[self.tbo_type] = latency_stats

        # Wait for all threads to complete
        self._print_latency_stats(latency_stats)

        self.assertEqual(
            latency_stats["success_count"],
            latency_stats["total_count"],
            f"Concurrent Success Rate Error: {latency_stats['success_count']}/{latency_stats['total_count']} (not 100%)"
        )
        print(f"\n Assertion 1 Passed: 100% Concurrent Request Success")

        server_info_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(server_info_resp.status_code, 200, "/get_server_info Request Failed (status != 200)")
        self.assertEqual(
            server_info_resp.json()["tbo_token_distribution_threshold"],
            self.tbo_token_distribution_threshold,
            f"TBO Parameter Config Error: Expected {self.tbo_token_distribution_threshold}, Got {server_info_resp.json()['tbo_token_distribution_threshold']}"
        )
        print(f" Assertion 2 Passed: TBO Threshold Config Correct ({self.tbo_token_distribution_threshold})")

    def _print_latency_stats(self, stats):
        if not stats:
            print("\n=== [Error] No Successful Requests, Cannot Calculate Latency ===")
            return
        print("\n" + "-" * 50)
        print(f"=== {self.tbo_type.upper()} Latency Statistics ===")
        print(f"  Total Requests: {stats['total_count']}")
        print(f"  Success Requests: {stats['success_count']} (Rate: {stats['success_rate']}%)")
        print(f"  Avg Elapsed Time: {stats['avg_elapsed']}s")


class TestTboTokenDistributionThresholdBase0(TestTboTokenDistributionThresholdBase8):
    """Testcase: Verify the baseline performance of --tbo-token-distribution-threshold with 50 concurrent long-token requests (TARGET_TOKEN_COUNT=2500).

    [Test Category] Parameter
    [Test Target] --tbo-token-distribution-threshold;
    """
    tbo_token_distribution_threshold = 0
    tbo_type = "tbo_disabled_0"

    def test_tbo_latency_comparison(self):
        # verify performance optimization
        print("\n" + "=" * 80)
        print("=== TBO Enabled(0.8) vs Disabled(0) Latency Comparison ===")
        print("=" * 80)

        # Extract core statistical data
        enabled_stats = TBO_LATENCY_STATISTICS["tbo_enabled_0.8"]
        disabled_stats = TBO_LATENCY_STATISTICS["tbo_disabled_0"]


        e_avg = enabled_stats["avg_elapsed"]
        d_avg = disabled_stats["avg_elapsed"]
        e_max = enabled_stats["max_elapsed"]
        d_max = disabled_stats["max_elapsed"]

        # Print assertion preparation information
        avg_optimize_rate = round(((d_avg - e_avg) / d_avg) * 100, 2) if d_avg > 0 else 0.0
        max_optimize_rate = round(((d_max - e_max) / d_max) * 100, 2) if d_max > 0 else 0.0
        print(
            f"1. Average Elapsed Time: ENABLED({e_avg}s) | DISABLED({d_avg}s) | Optimization Rate: {avg_optimize_rate}%")
        print(f"2. Max Elapsed Time: ENABLED({e_max}s) | DISABLED({d_max}s) | Optimization Rate: {max_optimize_rate}%")

        self.assertLess(
            e_avg, d_avg,
            f"TBO Avg Latency Assert Failed: Enabled({e_avg}s) ≥ Disabled({d_avg}s)"
        )
        print("\n✅ Latency Assert 1 Passed: TBO Enabled Avg Latency < Disabled")


if __name__ == "__main__":
    unittest.main()
