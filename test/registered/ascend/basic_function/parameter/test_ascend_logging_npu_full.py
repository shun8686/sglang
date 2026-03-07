import os
import io
import json
import re
import tempfile
import threading
import time
import unittest
import subprocess
import signal
from pathlib import Path
from time import sleep

import requests

# from docs.advanced_features.structured_outputs_for_reasoning_models import messages
from sglang.srt.utils import kill_process_tree

# from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH as MODEL_PATH
MODEL_PATH = "/home/weights/Llama-3.2-1B-Instruct"
# MODEL_PATH = "/home/weights/Qwen3-0.6B"
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=7200, suite="stage-b-test-npu")


# Done
# TestAscendLoggingNPURequests
# --log-requests、--log-requests-level

# 求助开发，验证是否充分
# --log-level、 --log-level-http
# 已有用例覆盖基本功能


# 求助开发，李果
# --log-requests-target TODO 多级路径


# 社区用例
# sglang/test/registered/utils/test_request_logger.py
# --log-requests-format（测试参数取值json，TestAscendLoggingNPULevel 覆盖默认值text）

# 开发定位中
# TestAscendLoggingNPUMetric
# --enable-metrics、--enable-metrics-for-all-schedulers TODO 基础、详细监控指标
# --bucket-time-to-first-token、--bucket-inter-token-latency、--bucket-e2e-request-latency
# 请求到达到首个token生成-响应时间；token输出间隔-生成速度稳定性；请求到达到完整返回时间-整体服务性能
# --decode-log-interval TODO 观测点

# TestAscendLoggingNPUCollectTokensHistogram TODO 观测点
# --collect-tokens-histogram、--prompt-tokens-buckets、--generation-tokens-buckets
# TestAscendLoggingNPUDecodeLogInterval

# TestAscendLoggingNPUGCWarningThresholdSecs
# --gc-warning-threshold-secs
# TestAscendLoggingNPUEnableRequestTimeStatsLogging
# --enable-request-time-stats-logging
# TestAscendLoggingNPUEnableTrace
# --enable-trace、 --otlp-traces-endpoint

# TODO --uvicorn-access-log-exclude-prefixes 排除以这些前缀开头的uvicorn访问日志
# TestAscendLoggingNPUCrashDumpFolder TODO  注入错误
# --crash-dump-folder 崩溃转储路径
# TODO --show-time-cost 打印阶段耗时
# TODO --tokenizer-metrics-custom-labels-header、--tokenizer-metrics-allowed-custom-labels
# 指定用于传递自定义标签以获取分词器指标的HTTP头， 允许用于分词器指标的自定义标签
# TODO --kv-events-config

class TestAscendLoggingNPUFullBase(CustomTestCase):
    """Comprehensive test for all Logging parameters on NPU environment.

    [Test Category] Functional
    [Test Target] All Logging parameters on NPU
    """

    # Note:
    # During the test, a fixed-length string needs to be returned.
    # The returned value of the same prompt may vary depending on the model
    model = MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST
    test_prompt = "What is the capital of France?"
    expected_output = "Paris"

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.out_log_name = "./log_requests_level_out_log.txt"
        cls.err_log_name = "./log_requests_level_err_log.txt"
        cls._temp_dir_obj = None
        cls.temp_dir = None
        cls.prepare_data()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls._temp_dir_obj:
            cls._temp_dir_obj.cleanup()

    @classmethod
    def prepare_data(cls):
        cls.other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        # 默认场景
        # --log-requests=False;
        cls.message = r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, .*"
        cls.out_log_name = "./log_requests_level_out_log.txt"
        cls.err_log_name = "./log_requests_level_err_log.txt"

        # 拉起4次服务
        # --log-requests、--log-requests-level
        # 实际使用4次服务
        # --log-requests=True,--log-requests-level=[0, 1, 2, 3]
        cls.log_request_message_dict = {
            "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
            "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
            "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
            "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
        }
        cls.keyword_Finish = r".*Finish: obj=GenerateReqInput\(.*http_worker_ipc=None, text='just.*"
        cls.keyword_start = "out={'text': '"
        cls.keyword_end = "', 'output_ids'"

        # --enable-metrics
        # 实际使用两次服务 i=[0, 1, 2]
        # --bucket-time-to-first-token、--bucket-inter-token-latency、--bucket-e2e-request-latency
        # 实际使用两次服务 i=[0, 1]
        # --enable-metrics=True, i=0 使用默认桶边界, i=1 使用自定义桶边界
        cls.my_bucket = ["0.1", "0.5", "1.0", "5.0", "10.0"]
        # --bucket-time-to-first-token
        cls.default_time_to_first_token_bucket = [
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
            "10.0", "20.0", "40.0", "60.0", "80.0",
            "100.0", "200.0", "400.0",
        ]
        # --bucket-inter-token-latency
        cls.default_inter_token_latency_bucket = [
            "0.002", "0.004", "0.006", "0.008",
            "0.01", "0.015", "0.02", "0.025", "0.03", "0.035", "0.04", "0.06", "0.08",
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
        ]
        # --bucket-e2e-request-latency
        cls.default_e2e_request_latency_bucket = [
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
            "10.0", "20.0", "40.0", "60.0", "80.0",
            "100.0", "200.0", "400.0", "600.0",
            "1200.0", "1800.0", "2400.0",
        ]
        # --collect-tokens-histogram
        # --prompt-tokens-buckets、--generation-tokens-bucket
        cls.default_tokens_bucket = [
            "100.0", "300.0", "500.0", "700.0",
            "1000.0", "1500.0", "2000.0", "3000.0", "4000.0", "5000.0", "6000.0", "7000.0", "8000.0", "9000.0",
            "10000.0", "12000.0", "15000.0", "20000.0", "22000.0", "25000.0",
            "30000.0", "35000.0", "40000.0", "66000.0", "99000.0",
            "132000.0", "300000.0", "600000.0", "900000.0",
            "1.1e+06",
        ]
        cls.my_tokens_bucket = [
            "100.0", "1000.0", "10000.0", "100000.0", "300000.0", "600000.0", "900000.0",
        ]
        cls.my_tse_set = ["1000", "2", "8"]
        cls.my_tse_bucket = ["984.0", "992.0", "996.0", "998.0", "1000.0", "1002.0", "1004.0", "1008.0", "1016.0"]

        # --tokenizer-metrics-custom-labels-header、--tokenizer-metrics-allowed-custom-labels
        cls.labels_header = "X-Metrics-Labels"
        cls.my_label = "business_line"

    def _launch_server_with_logging(
        self,
        log_level="info",
        log_level_http=None,
        log_requests=False,
        log_requests_level=None,
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
        decode_log_interval=None,
        enable_request_time_stats_logging=False,
        enable_trace=False,
        otlp_traces_endpoint="localhost:4317",
        crash_dump_folder=None,
        tp_size=None,
        enable_dp_attention=None,
        dp_size=None,
        uvicorn_access_log_exclude_prefixes=None,
        show_time_cost=None,
        tokenizer_metrics_custom_labels_header=None,
        tokenizer_metrics_allowed_custom_labels=None,
        kv_events_config=None,
        out_log_file=None,
        err_log_file=None,
    ):
        """Launch server with logging parameters."""

        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        if tp_size:
            other_args.extend(["--tp-size", str(tp_size)])

        if enable_dp_attention:
            other_args.extend(["--enable-dp-attention"])
            other_args.extend(["--dp-size", str(dp_size)])

        if log_level is not None:
            other_args.extend(["--log-level", log_level])

        if log_level_http is not None:
            other_args.extend(["--log-level-http", log_level_http])

        if log_requests:
            other_args.append("--log-requests")
            if log_requests_level is not None:
                other_args.extend(["--log-requests-level", str(log_requests_level)])
            if log_requests_format is not None:
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

        if decode_log_interval:
            other_args.extend(["--decode-log-interval", str(decode_log_interval)])

        if enable_request_time_stats_logging:
            other_args.append("--enable-request-time-stats-logging")

        if enable_trace:
            other_args.append("--enable-trace")
            other_args.extend(["--otlp-traces-endpoint", otlp_traces_endpoint])

        if crash_dump_folder is not None:
            other_args.extend(["--crash-dump-folder", crash_dump_folder])

        if uvicorn_access_log_exclude_prefixes is not None:
            other_args.extend(["--uvicorn-access-log-exclude-prefixes", uvicorn_access_log_exclude_prefixes])

        if show_time_cost is not None:
            other_args.extend(["--show-time-cost", show_time_cost])

        if tokenizer_metrics_custom_labels_header is not None:
            other_args.extend(["--tokenizer-metrics-custom-labels-header", tokenizer_metrics_custom_labels_header])

        if tokenizer_metrics_allowed_custom_labels is not None:
            other_args.extend(["--tokenizer-metrics-allowed-custom-labels", tokenizer_metrics_allowed_custom_labels])

        if kv_events_config is not None:
            other_args.extend(["--kv-events-config", kv_events_config])

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        ) if out_log_file is None and err_log_file is None else popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        return process

    def _clean_environment(self, process, out_log_file, err_log_file):
        """Clean up environment variables used by tests."""
        if process:
            kill_process_tree(process.pid)
        if out_log_file:
            out_log_file.close()
            os.remove(self.out_log_name)
        if err_log_file:
            err_log_file.close()
            os.remove(self.err_log_name)

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

    def _test_inference_function(self, max_new_tokens=32):
        """Send a basic inference request to test inference function."""
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

    @staticmethod
    def get_lines_with_keyword(filename, keyword):
        results = []
        try:
            with open(filename, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    if re.match(keyword, line):
                        results.append(
                            {
                                "line_number": line_num,
                                "content": line.strip(),
                            }
                        )
            return results
        except Exception as e:
            print(f"error:{e}")
            return []

    def _test_log_requests_level(self, log_requests_level, out_log_file):
        """Send a basic inference request to test inference function."""
        # When the log level is 2, input and output will be truncated, retaining a length of 2048.
        # When the log level is 3, complete input and output will be retained.
        # In other cases, only basic functions are tested, and reducing token output increases testing speed.
        max_new_token = 2500 if log_requests_level >= 2 else 100
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": f"just return me a string with of {max_new_token} characters.",
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_token},
            },
        )
        self.assertEqual(response.status_code, 200)
        out_log_file.seek(0)
        content = out_log_file.read()

        self.assertTrue(len(content) > 0)
        self.assertIsNotNone(re.search(self.log_request_message_dict[str(log_requests_level)], content))
        if log_requests_level >= 2:
            lines = self.get_lines_with_keyword(self.out_log_name, self.keyword_Finish)
            Finish_message = lines[0]["content"]
            start_index = Finish_message.find(self.keyword_start) + len(self.keyword_start)
            end_index = Finish_message.find(self.keyword_end)
            out_text = Finish_message[start_index:end_index]
            out_text_length = len(out_text)
            out_text_length_n = len(out_text.replace("\\n", " "))
            if log_requests_level == 2:
                self.assertIn("' ... '", out_text)
                self.assertTrue(out_text_length_n - len("' ... '") == 2048)
            else:
                self.assertNotIn("' ... '", out_text)
                self.assertTrue(out_text_length > 2048)

    def _test_log_metrics_tokenizer_label(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "Content-Type": "application/json",
                "X-Metrics-Labels": f"{self.my_label}=cunstomer_service",
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(self.expected_output, response.text)

        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        self.assertEqual(response.status_code, 200)
        metrics_content = response.text
        message = f'sglang:time_to_first_token_seconds_bucket{{{self.my_label}="'
        self.assertIn(message, metrics_content)
        message = f'sglang:inter_token_latency_seconds_bucket{{{self.my_label}='
        self.assertIn(message, metrics_content)
        message = f'sglang:e2e_request_latency_seconds_bucket{{{self.my_label}='
        self.assertIn(message, metrics_content)

    def get_default_other_args(self):
        return [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

    def _test_metrics(
        self,
        expected_time_to_first_token_bucket=None,
        expected_inter_token_latency_bucket=None,
        expected_e2e_request_latency_bucket=None,
        expected_prompt_tokens_bucket=None,
        expected_generation_tokens_bucket=None,
    ):
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        self.assertEqual(response.status_code, 200)
        metrics_content = response.text
        if expected_time_to_first_token_bucket is not None:
            for le in expected_time_to_first_token_bucket:
                message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_inter_token_latency_bucket is not None:
            for le in expected_inter_token_latency_bucket:
                message = f'sglang:inter_token_latency_seconds_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_e2e_request_latency_bucket is not None:
            for le in expected_e2e_request_latency_bucket:
                message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_prompt_tokens_bucket is not None:
            for le in expected_prompt_tokens_bucket:
                message = f'sglang:prompt_tokens_histogram_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_generation_tokens_bucket is not None:
            for le in expected_generation_tokens_bucket:
                message = f'sglang:generation_tokens_histogram_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        return metrics_content

    def _test_gc_warning_threshold(self, err_log_file):
        prompt_template = "just return me a string with of 10000 characters: " + "A" * 5000
        max_token = 1000

        def send_request():
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": prompt_template,
                        "sampling_params": {"temperature": 0, "max_new_tokens": max_token},
                    },
                )
            except Exception as e:
                print(e)

        threads = []
        for _ in range(200):
            t = threading.Thread(target=send_request)
            t.start()
            threads.append(t)
            sleep(0.01)

        for t in threads:
            t.join()

        GC_info = "LONG GARBAGE COLLECTION DETECTED"
        err_log_file.seek(0)
        content = err_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIn(GC_info, content)

    def _safe_kill_process(self):
        if self.process is not None:
            kill_process_tree(self.process.pid)
            self.process = None


class TestAscendLoggingDefault(TestAscendLoggingNPUFullBase):
    def test_logging_default(self):
        other_args = self.get_default_other_args()
        out_log_file = open(self.out_log_name, "w+", encoding="utf-8")
        err_log_file = open(self.err_log_name, "w+", encoding="utf-8")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            self._send_inference_request()

            out_log_file.seek(0)
            content = out_log_file.read()
            self.assertTrue(len(content) > 0)
            self.assertIsNone(re.search(self.message, content))

            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            self.assertEqual(response.status_code, 404)
        finally:
            self._clean_environment(process, out_log_file, err_log_file)




class TestAscendLoggingCase0(TestAscendLoggingNPUFullBase):
    def test_logging_case_0(self):
        other_args = self.get_default_other_args()
        out_log_file = open(self.out_log_name, "w+", encoding="utf-8")
        err_log_file = open(self.err_log_name, "w+", encoding="utf-8")

        other_args.append("--log-requests")
        log_requests_level = 0
        other_args.extend(["--log-requests-level", str(log_requests_level)])

        other_args.extend(["--enable-metrics"])

        expected_time_to_first_token_bucket = self.default_time_to_first_token_bucket
        expected_inter_token_latency_bucket = self.default_inter_token_latency_bucket
        expected_e2e_request_latency_bucket = self.default_e2e_request_latency_bucket

        other_args.extend(["--collect-tokens-histogram"])

        expected_prompt_tokens_bucket = self.default_tokens_bucket
        expected_generation_tokens_bucket = self.default_tokens_bucket

        other_args.extend(["--gc-warning-threshold-secs", "0.01"])

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            self._test_inference_function()

            self._test_log_requests_level(log_requests_level, out_log_file)

            self._test_metrics(
                expected_time_to_first_token_bucket=expected_time_to_first_token_bucket,
                expected_inter_token_latency_bucket=expected_inter_token_latency_bucket,
                expected_e2e_request_latency_bucket=expected_e2e_request_latency_bucket,
                expected_prompt_tokens_bucket=expected_prompt_tokens_bucket,
                expected_generation_tokens_bucket=expected_generation_tokens_bucket,
            )

            self._test_gc_warning_threshold(err_log_file)
        finally:
            self._clean_environment(process, out_log_file, err_log_file)


class TestAscendLoggingCase1(TestAscendLoggingNPUFullBase):
    def test_logging_case_1(self):
        other_args = self.get_default_other_args()
        out_log_file = open(self.out_log_name, "w+", encoding="utf-8")
        err_log_file = open(self.err_log_name, "w+", encoding="utf-8")

        other_args.append("--log-requests")
        log_requests_level = 1
        other_args.extend(["--log-requests-level", str(log_requests_level)])

        other_args.extend(["--enable-metrics"])

        other_args.extend(["--bucket-time-to-first-token"] + self.my_bucket)
        other_args.extend(["--bucket-inter-token-latency"] + self.my_bucket)
        other_args.extend(["--bucket-e2e-request-latency"] + self.my_bucket)
        expected_time_to_first_token_bucket = self.my_bucket
        expected_inter_token_latency_bucket = self.my_bucket
        expected_e2e_request_latency_bucket = self.my_bucket

        other_args.extend(["--collect-tokens-histogram"])

        other_args.extend(["--prompt-tokens-buckets"] + ["custom"] + self.my_tokens_bucket)
        other_args.extend(["--generation-tokens-buckets"] + ["custom"] + self.my_tokens_bucket)
        expected_prompt_tokens_bucket = self.my_tokens_bucket
        expected_generation_tokens_bucket = self.my_tokens_bucket

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            self._test_inference_function()

            self._test_log_requests_level(log_requests_level, out_log_file)

            self._test_metrics(
                expected_time_to_first_token_bucket=expected_time_to_first_token_bucket,
                expected_inter_token_latency_bucket=expected_inter_token_latency_bucket,
                expected_e2e_request_latency_bucket=expected_e2e_request_latency_bucket,
                expected_prompt_tokens_bucket=expected_prompt_tokens_bucket,
                expected_generation_tokens_bucket=expected_generation_tokens_bucket,
            )
        finally:
            self._clean_environment(process, out_log_file, err_log_file)


class TestAscendLoggingCase2(TestAscendLoggingNPUFullBase):
    def test_logging_case_2(self):
        other_args = self.get_default_other_args()
        out_log_file = open(self.out_log_name, "w+", encoding="utf-8")
        err_log_file = open(self.err_log_name, "w+", encoding="utf-8")

        other_args.append("--log-requests")
        log_requests_level = 2
        other_args.extend(["--log-requests-level", str(log_requests_level)])

        other_args.extend(["--enable-metrics"])
        other_args.extend(["--collect-tokens-histogram"])
        other_args.extend(["--prompt-tokens-buckets"] + ["tse"] + self.my_tse_set)
        other_args.extend(["--generation-tokens-buckets"] + ["tse"] + self.my_tse_set)
        expected_prompt_tokens_bucket = self.my_tse_bucket
        expected_generation_tokens_bucket = self.my_tse_bucket

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            self._test_inference_function()

            self._test_log_requests_level(log_requests_level, out_log_file)

            self._test_metrics(
                expected_prompt_tokens_bucket=expected_prompt_tokens_bucket,
                expected_generation_tokens_bucket=expected_generation_tokens_bucket,
            )
        finally:
            self._clean_environment(process, out_log_file, err_log_file)


class TestAscendLoggingCase3(TestAscendLoggingNPUFullBase):
    def test_logging_case_3(self):
        other_args = self.get_default_other_args()
        out_log_file = open(self.out_log_name, "w+", encoding="utf-8")
        err_log_file = open(self.err_log_name, "w+", encoding="utf-8")

        other_args.append("--log-requests")
        log_requests_level = 3
        other_args.extend(["--log-requests-level", str(log_requests_level)])

        other_args.extend(["--enable-metrics"])
        other_args.extend(["--tokenizer-metrics-custom-labels-header", self.labels_header])
        other_args.extend(["--tokenizer-metrics-allowed-custom-labels", self.my_label])

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            self._test_inference_function()

            self._test_log_requests_level(log_requests_level, out_log_file)

            # test --tokenizer-metrics-custom-labels-header、--tokenizer-metrics-allowed-custom-labels
            self._test_log_metrics_tokenizer_label()
        finally:
            self._clean_environment(process, out_log_file, err_log_file)


# TODO 多级目录
class TestAscendLoggingNPURequestsTarget(TestAscendLoggingNPUFullBase):
    def test_06_log_requests_target_variations(self):
        """Test log-requests-target variations."""
        print("\n=== Test 06: log-requests-target variations ===")

        for target_config in [["stdout"], [self.temp_dir], ["stdout", self.temp_dir]]:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.temp_dir = self._temp_dir_obj.name

            try:
                self.process = self._launch_server_with_logging(
                    log_requests=True,
                    log_requests_level=2,
                    log_requests_format="text",
                    log_requests_target=target_config,
                )
                time.sleep(5)

                result = self._send_inference_request()
                print(f"  Target {target_config} test passed")

                if self.temp_dir in target_config:
                    log_files = list(Path(self.temp_dir).glob("*.log"))
                    self.assertGreater(len(log_files), 0)

                    file_content = log_files[0].read_text()
                    self.assertIn("Receive:", file_content)
                    self.assertIn("Finish:", file_content)
            finally:
                self._safe_kill_process()

        print(f"✓ All log-requests-target variations test passed")


if __name__ == "__main__":
    # unittest.main()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLogging))

    suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingDefault))

    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingCase0))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingCase1))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingCase2))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingCase3))

    # DONE
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLogRequests))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUCollectTokensHistogram))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPULabel))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUGCWarningThresholdSecs))

    # TODO
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUMetric))

    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPURequestsFormat))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPURequestsTarget))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUMetric))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUCollectTokensHistogram))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUDecodeLogInterval))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUEnableRequestTimeStatsLogging))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUEnableTrace))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUCrashDumpFolder))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUBucket))
    runner = unittest.TextTestRunner()
    runner.run(suite)
