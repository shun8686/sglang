import os
import io
import json
import tempfile
import threading
import time
import unittest
import subprocess
import signal
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH as MODEL_PATH
MODEL_PATH = "/home/weights/Llama-3.2-1B-Instruct"
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=7200, suite="stage-b-test-npu")


class TestAscendLoggingNPUFullBase(CustomTestCase):
    """Comprehensive test for all Logging parameters on NPU environment.

    [Test Category] Functional
    [Test Target] All Logging parameters on NPU
    """

    model = MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST
    test_prompt = "What is the capital of France?"
    expected_output = "Paris"

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls._temp_dir_obj = None
        cls.temp_dir = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls._temp_dir_obj:
            cls._temp_dir_obj.cleanup()

    def _launch_server_with_logging(
        self,
        log_level="info",
        log_level_http=None,
        log_requests=False,
        log_requests_level=2,
        log_requests_format="text",
        log_requests_target=None,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
        collect_tokens_histogram=False,
        bucket_time_to_first_token=None,
        bucket_inter_token_latency=None,
        bucket_e2e_request_latency=None,
        prompt_tokens_buckets=None,
        generation_tokens_buckets=None,
        gc_warning_threshold_secs=0.0,
        decode_log_interval=40,
        enable_request_time_stats_logging=False,
        enable_trace=False,
        otlp_traces_endpoint="localhost:4317",
        crash_dump_folder=None,
        tp_size=1,
    ):
        """Launch server with logging parameters."""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--log-level",
            log_level,
            "--tp-size",
            str(tp_size),
        ]

        if log_level_http is not None:
            other_args.extend(["--log-level-http", log_level_http])

        if log_requests:
            other_args.append("--log-requests")
            other_args.extend(["--log-requests-level", str(log_requests_level)])
            other_args.extend(["--log-requests-format", log_requests_format])

            if log_requests_target is not None:
                other_args.extend(["--log-requests-target"] + log_requests_target)

        if enable_metrics:
            other_args.append("--enable-metrics")

        if enable_metrics_for_all_schedulers:
            other_args.append("--enable-metrics-for-all-schedulers")

        if collect_tokens_histogram:
            other_args.append("--collect-tokens-histogram")

        if bucket_time_to_first_token is not None:
            other_args.extend(["--bucket-time-to-first-token"] + [str(x) for x in bucket_time_to_first_token])

        if bucket_inter_token_latency is not None:
            other_args.extend(["--bucket-inter-token-latency"] + [str(x) for x in bucket_inter_token_latency])

        if bucket_e2e_request_latency is not None:
            other_args.extend(["--bucket-e2e-request-latency"] + [str(x) for x in bucket_e2e_request_latency])

        if prompt_tokens_buckets is not None:
            other_args.extend(["--prompt-tokens-buckets"] + prompt_tokens_buckets)

        if generation_tokens_buckets is not None:
            other_args.extend(["--generation-tokens-buckets"] + generation_tokens_buckets)

        if gc_warning_threshold_secs > 0:
            other_args.extend(["--gc-warning-threshold-secs", str(gc_warning_threshold_secs)])

        other_args.extend(["--decode-log-interval", str(decode_log_interval)])

        if enable_request_time_stats_logging:
            other_args.append("--enable-request-time-stats-logging")

        if enable_trace:
            other_args.append("--enable-trace")
            other_args.extend(["--otlp-traces-endpoint", otlp_traces_endpoint])

        if crash_dump_folder is not None:
            other_args.extend(["--crash-dump-folder", crash_dump_folder])

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return process

    def _send_inference_request(self, max_new_tokens=32):
        """Send a basic inference request."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.expected_output, response.text)
        return response.text

    def _check_metrics_endpoint(self):
        """Check if metrics endpoint is accessible and returns valid Prometheus metrics."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            self.assertEqual(response.status_code, 200)
            metrics_content = response.text
            self.assertIn("sglang_", metrics_content)
            return metrics_content
        except requests.exceptions.RequestException as e:
            self.fail(f"Metrics endpoint not accessible: {e}")

class TestAscendLoggingNPULevel(TestAscendLoggingNPUFullBase):
    def test_log_level_info(self):
        level_list = ["info", "debug",  "warning", "error", "critical"]
        http_level_list = ["info", "critical", "error", "warning", "debug", ]

        for level, http_level in level_list, http_level_list:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.temp_dir = self._temp_dir_obj.name

            try:
                self.process = self._launch_server_with_logging(
                    log_level=level,
                    log_level_http=http_level,
                    log_requests=True,
                    log_requests_level=2,
                    log_requests_format="text",
                    log_requests_target=["stdout", self.temp_dir],
                )
                time.sleep(5)

                result = self._send_inference_request()
                print(f"✓ log-level=debug test passed, result: {result[:50]}...")

                log_files = list(Path(self.temp_dir).glob("*.log"))
                self.assertGreater(len(log_files), 0)

                file_content = log_files[0].read_text()
                self.assertIn("Receive:", file_content)
                self.assertIn("Finish:", file_content)
            finally:
                if self.process is not None:
                    kill_process_tree(self.process.pid)
                    self.process = None


    # def test_02_log_level_error(self):
    #     """Test log-level=error."""
    #     print("\n=== Test 02: log-level=error ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             log_level="error",
    #             log_requests=True,
    #             log_requests_level=2,
    #             log_requests_format="text",
    #             log_requests_target=["stdout", self.temp_dir],
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ log-level=error test passed, result: {result[:50]}...")
    #
    #         log_files = list(Path(self.temp_dir).glob("*.log"))
    #         self.assertGreater(len(log_files), 0)
    #
    #         file_content = log_files[0].read_text()
    #         self.assertIn("Receive:", file_content)
    #         self.assertIn("Finish:", file_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
# class TestAscendLoggingNPUHTTPLevel(TestAscendLoggingNPUFullBase):
#     def test_03_log_level_http_info(self):
#         """Test log-level-http=info."""
#         print("\n=== Test 03: log-level-http=info ===")
#         self._temp_dir_obj = tempfile.TemporaryDirectory()
#         self.temp_dir = self._temp_dir_obj.name
#
#         try:
#             self.process = self._launch_server_with_logging(
#                 log_level="error",
#                 log_level_http="info",
#                 log_requests=True,
#                 log_requests_level=2,
#                 log_requests_format="text",
#                 log_requests_target=["stdout", self.temp_dir],
#             )
#             time.sleep(5)
#
#             result = self._send_inference_request()
#             print(f"✓ log-level-http=info test passed, result: {result[:50]}...")
#
#             log_files = list(Path(self.temp_dir).glob("*.log"))
#             self.assertGreater(len(log_files), 0)
#         finally:
#             if self.process is not None:
#                 kill_process_tree(self.process.pid)
#                 self.process = None

# class TestAscendLoggingNPULevel(TestAscendLoggingNPUFullBase):
#     def test_04_log_requests_level_all(self):
#         """Test all log-requests-level values."""
#         print("\n=== Test 04: log-requests-level (0, 1, 2, 3) ===")
#         self._temp_dir_obj = tempfile.TemporaryDirectory()
#         self.temp_dir = self._temp_dir_obj.name
#
#         for level in [0, 1, 2, 3]:
#             try:
#                 self.process = self._launch_server_with_logging(
#                     log_requests=True,
#                     log_requests_level=level,
#                     log_requests_format="text",
#                     log_requests_target=["stdout", self.temp_dir],
#                 )
#                 time.sleep(5)
#
#                 result = self._send_inference_request()
#                 print(f"  Level {level} test passed")
#
#                 log_files = list(Path(self.temp_dir).glob("*.log"))
#                 self.assertGreater(len(log_files), 0)
#
#                 file_content = log_files[0].read_text()
#                 self.assertIn("Receive:", file_content)
#                 self.assertIn("Finish:", file_content)
#             finally:
#                 kill_process_tree(self.process.pid)
#                 self.process = None
#
#         print(f"✓ All log-requests-level test passed")

    # def test_05_log_requests_format_json(self):
    #     """Test log-requests-format=json."""
    #     print("\n=== Test 05: log-requests-format=json ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             log_requests=True,
    #             log_requests_level=2,
    #             log_requests_format="json",
    #             log_requests_target=["stdout", self.temp_dir],
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ log-requests-format=json test passed, result: {result[:50]}...")
    #
    #         log_files = list(Path(self.temp_dir).glob("*.log"))
    #         self.assertGreater(len(log_files), 0)
    #
    #         file_content = log_files[0].read_text()
    #         json_lines = [line for line in file_content.splitlines() if line.strip().startswith("{")]
    #         self.assertGreater(len(json_lines), 0)
    #
    #         for line in json_lines:
    #             data = json.loads(line)
    #             self.assertIn("event", data)
    #             self.assertIn("rid", data)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_06_log_requests_target_variations(self):
    #     """Test log-requests-target variations."""
    #     print("\n=== Test 06: log-requests-target variations ===")
    #
    #     for target_config in [["stdout"], [self.temp_dir], ["stdout", self.temp_dir]]:
    #         self._temp_dir_obj = tempfile.TemporaryDirectory()
    #         self.temp_dir = self._temp_dir_obj.name
    #
    #         try:
    #             self.process = self._launch_server_with_logging(
    #                 log_requests=True,
    #                 log_requests_level=2,
    #                 log_requests_format="text",
    #                 log_requests_target=target_config,
    #             )
    #             time.sleep(5)
    #
    #             result = self._send_inference_request()
    #             print(f"  Target {target_config} test passed")
    #
    #             if self.temp_dir in target_config:
    #                 log_files = list(Path(self.temp_dir).glob("*.log"))
    #                 self.assertGreater(len(log_files), 0)
    #
    #                 file_content = log_files[0].read_text()
    #                 self.assertIn("Receive:", file_content)
    #                 self.assertIn("Finish:", file_content)
    #         finally:
    #             kill_process_tree(self.process.pid)
    #             self.process = None
    #
    #     print(f"✓ All log-requests-target variations test passed")
    #
    # def test_07_enable_metrics(self):
    #     """Test enable-metrics."""
    #     print("\n=== Test 07: enable-metrics ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ enable-metrics test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_cache_hit_rate", metrics_content)
    #         self.assertIn("sglang_num_running_reqs", metrics_content)
    #         self.assertIn("sglang_gen_throughput", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_08_enable_metrics_for_all_schedulers(self):
    #     """Test enable-metrics-for-all-schedulers with TP2."""
    #     print("\n=== Test 08: enable-metrics-for-all-schedulers (TP2) ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             enable_metrics_for_all_schedulers=True,
    #             tp_size=2,
    #         )
    #         time.sleep(8)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ enable-metrics-for-all-schedulers test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("tp_rank", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_09_custom_buckets(self):
    #     """Test custom metric buckets."""
    #     print("\n=== Test 09: custom metric buckets ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             bucket_time_to_first_token=[0.1, 0.5, 1.0, 2.0, 5.0],
    #             bucket_inter_token_latency=[0.01, 0.05, 0.1, 0.5],
    #             bucket_e2e_request_latency=[1.0, 5.0, 10.0, 30.0],
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ custom buckets test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_time_to_first_token_bucket", metrics_content)
    #         self.assertIn("sglang_e2e_request_latency_bucket", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_10_collect_tokens_histogram(self):
    #     """Test collect-tokens-histogram."""
    #     print("\n=== Test 10: collect-tokens-histogram ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             collect_tokens_histogram=True,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ collect-tokens-histogram test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_prompt_tokens", metrics_content)
    #         self.assertIn("sglang_generation_tokens", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_11_prompt_tokens_buckets_default(self):
    #     """Test prompt-tokens-buckets with default."""
    #     print("\n=== Test 11: prompt-tokens-buckets default ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             collect_tokens_histogram=True,
    #             prompt_tokens_buckets=["default"],
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ prompt-tokens-buckets default test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_prompt_tokens_bucket", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_12_prompt_tokens_buckets_tse(self):
    #     """Test prompt-tokens-buckets with tse."""
    #     print("\n=== Test 12: (prompt-tokens-buckets tse ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             collect_tokens_histogram=True,
    #             prompt_tokens_buckets=["tse", "512", "2", "8"],
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ prompt-tokens-buckets tse test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_prompt_tokens_bucket", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_13_prompt_tokens_buckets_custom(self):
    #     """Test prompt-tokens-buckets with custom."""
    #     print("\n=== Test 13: prompt-tokens-buckets custom ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             collect_tokens_histogram=True,
    #             prompt_tokens_buckets=["custom", "100", "500", "1000", "5000"],
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ prompt-tokens-buckets custom test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_prompt_tokens_bucket", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_14_generation_tokens_buckets_variations(self):
    #     """Test generation-tokens-buckets variations."""
    #     print("\n=== Test 14: generation-tokens-buckets variations ===")
    #
    #     for bucket_config in [["default"], ["tse", "256", "2", "8"], ["custom", "50", "100", "200", "500"]]:
    #         try:
    #             self.process = self._launch_server_with_logging(
    #                 enable_metrics=True,
    #                 collect_tokens_histogram=True,
    #                 generation_tokens_buckets=bucket_config,
    #             )
    #             time.sleep(5)
    #
    #             result = self._send_inference_request()
    #             print(f"  Generation tokens buckets {bucket_config[0]} test passed")
    #
    #             metrics_content = self._check_metrics_endpoint()
    #             self.assertIn("sglang_generation_tokens_bucket", metrics_content)
    #         finally:
    #             kill_process_tree(self.process.pid)
    #             self.process = None
    #
    #     print(f"✓ All generation-tokens-buckets variations test passed")
    #
    # def test_15_decode_log_interval(self):
    #     """Test decode-log-interval."""
    #     print("\n=== Test 15: decode-log-interval ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             log_level="debug",
    #             decode_log_interval=10,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request(max_new_tokens=100)
    #         print(f"✓ decode-log-interval test passed, result: {result[:50]}...")
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_16_enable_request_time_stats_logging(self):
    #     """Test enable-request-time-stats-logging."""
    #     print("\n=== Test 16: enable-request-time-stats-logging ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_request_time_stats_logging=True,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ enable-request-time-stats-logging test passed, result: {result[:50]}...")
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_17_enable_trace(self):
    #     """Test enable-trace (requires OTLP collector)."""
    #     print("\n=== Test 17: enable-trace ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_trace=True,
    #             otlp_traces_endpoint="localhost:4317",
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ enable-trace test passed (server started successfully), result: {result[:50]}...")
    #     except Exception as e:
    #         print(f"⚠ enable-trace test skipped (OTLP collector may not be available): {e}")
    #     finally:
    #         if self.process:
    #             kill_process_tree(self.process.pid)
    #             self.process = None
    #
    # def test_18_gc_warning_threshold_secs(self):
    #     """Test gc-warning-threshold-secs."""
    #     print("\n=== Test 18: gc-warning-threshold-secs ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             gc_warning_threshold_secs=0.1,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ gc-warning-threshold-secs test passed, result: {result[:50]}...")
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_19_crash_dump_folder(self):
    #     """Test crash-dump-folder."""
    #     print("\n=== Test 19: crash-dump-folder ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #     crash_dir = os.path.join(self.temp_dir, "crash_dumps")
    #     os.makedirs(crash_dir, exist_ok=True)
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             crash_dump_folder=crash_dir,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ crash-dump-folder test passed (server started successfully), result: {result[:50]}...")
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_20_combined_logging_params(self):
    #     """Test combined logging parameters."""
    #     print("\n=== Test 20: Combined logging parameters ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             log_level="debug",
    #             log_requests=True,
    #             log_requests_level=2,
    #             log_requests_format="json",
    #             log_requests_target=["stdout", self.temp_dir],
    #             enable_metrics=True,
    #             collect_tokens_histogram=True,
    #             enable_request_time_stats_logging=True,
    #         )
    #         time.sleep(5)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ Combined logging parameters test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_cache_hit_rate", metrics_content)
    #
    #         log_files = list(Path(self.temp_dir).glob("*.log"))
    #         self.assertGreater(len(log_files), 0)
    #
    #         file_content = log_files[0].read_text()
    #         json_lines = [line for line in file_content.splitlines() if line.strip().startswith("{")]
    #         self.assertGreater(len(json_lines), 0)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_21_concurrent_requests_logging(self):
    #     """Test logging with concurrent requests."""
    #     print("\n=== Test 21: Concurrent requests logging ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             log_requests=True,
    #             log_requests_level=2,
    #             log_requests_format="text",
    #             log_requests_target=["stdout", self.temp_dir],
    #             enable_metrics=True,
    #         )
    #         time.sleep(5)
    #
    #         threads = []
    #         results = []
    #
    #         def send_request(rid):
    #             response = requests.post(
    #                 f"{self.base_url}/generate",
    #                 json={
    #                     "text": f"Test request {rid}",
    #                     "sampling_params": {
    #                         "temperature": 0,
    #                         "max_new_tokens": 16,
    #                     },
    #                 },
    #                 timeout=60,
    #             )
    #             results.append(response.status_code)
    #
    #         for i in range(20):
    #             thread = threading.Thread(target=send_request, args=(i,))
    #             threads.append(thread)
    #             thread.start()
    #
    #         for thread in threads:
    #             thread.join()
    #
    #         success_count = sum(1 for r in results if r == 200)
    #         self.assertGreaterEqual(success_count, 18)
    #         print(f"✓ Concurrent requests logging test passed, {success_count}/20 succeeded")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_num_running_reqs", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_22_tp4_metrics(self):
    #     """Test metrics with TP4."""
    #     print("\n=== Test 22: TP4 metrics ===")
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             enable_metrics=True,
    #             tp_size=4,
    #         )
    #         time.sleep(10)
    #
    #         result = self._send_inference_request()
    #         print(f"✓ TP4 metrics test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_cache_hit_rate", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None
    #
    # def test_23_long_running_stability(self):
    #     """Test logging stability under long running."""
    #     print("\n=== Test 23: Long running stability ===")
    #     self._temp_dir_obj = tempfile.TemporaryDirectory()
    #     self.temp_dir = self._temp_dir_obj.name
    #
    #     try:
    #         self.process = self._launch_server_with_logging(
    #             log_requests=True,
    #             log_requests_level=2,
    #             log_requests_format="text",
    #             log_requests_target=["stdout", self.temp_dir],
    #             enable_metrics=True,
    #         )
    #         time.sleep(5)
    #
    #         start_time = time.time()
    #         request_count = 0
    #         while time.time() - start_time < 60:
    #             result = self._send_inference_request()
    #             request_count += 1
    #             time.sleep(0.5)
    #
    #         print(f"✓ Long running stability test passed, {request_count} requests completed")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn("sglang_cache_hit_rate", metrics_content)
    #     finally:
    #         kill_process_tree(self.process.pid)
    #         self.process = None


if __name__ == "__main__":
    unittest.main()
