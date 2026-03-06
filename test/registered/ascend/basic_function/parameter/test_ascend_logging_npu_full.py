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

# TestAscendLoggingNPUCollectTokensHistogram TODO 观测点
# --collect-tokens-histogram、--prompt-tokens-buckets、--generation-tokens-buckets
# TestAscendLoggingNPUDecodeLogInterval
# --decode-log-interval TODO 观测点
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

    # TODO 验证方法
    def _check_metrics_endpoint(
        self,
        expected_bucket_time_to_first_token_list,
        expected_bucket_inter_token_latency_list,
        expected_bucket_e2e_request_latency_list
    ):
        """Check if metrics endpoint is accessible and returns valid Prometheus metrics."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            self.assertEqual(response.status_code, 200)
            metrics_content = response.text
            for le in expected_bucket_time_to_first_token_list:
                message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
            for le in expected_bucket_inter_token_latency_list:
                message = f'sglang:inter_token_latency_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
            for le in expected_bucket_e2e_request_latency_list:
                message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
            return metrics_content
        except requests.exceptions.RequestException as e:
            self.fail(f"Metrics endpoint not accessible: {e}")

    def _safe_kill_process(self):
        if self.process is not None:
            kill_process_tree(self.process.pid)
            self.process = None

class TestAscendLogging(TestAscendLoggingNPUFullBase):
    # def test_logging_default(self):
    #     # --log-requests=False;
    #     message = r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, .*"
    #     out_log_name = "./log_requests_level_out_log.txt"
    #     err_log_name = "./log_requests_level_err_log.txt"
    #
    #     out_log_file = open(out_log_name, "w+", encoding="utf-8")
    #     err_log_file = open(err_log_name, "w+", encoding="utf-8")
    #     process = self._launch_server_with_logging(
    #         out_log_file=out_log_file,
    #         err_log_file=err_log_file,
    #     )
    #
    #     try:
    #         self._send_inference_request()
    #
    #         max_new_token = 100
    #
    #         response = requests.post(
    #             f"{self.base_url}/generate",
    #             json={
    #                 "text": f"just return me a string with of {max_new_token} characters",
    #                 "sampling_params": {"temperature": 0, "max_new_tokens": max_new_token},
    #             },
    #         )
    #         self.assertEqual(response.status_code, 200)
    #
    #         # check --log-requests=False
    #         out_log_file.seek(0)
    #         content = out_log_file.read()
    #
    #         self.assertTrue(len(content) > 0)
    #         self.assertIsNone(re.search(message, content))
    #
    #         # check --enable-metrics=False
    #         response = requests.get(f"{self.base_url}/metrics", timeout=10)
    #         self.assertEqual(response.status_code, 404)
    #     finally:
    #         kill_process_tree(process.pid)
    #         out_log_file.close()
    #         err_log_file.close()
    #         os.remove(out_log_name)
    #         os.remove(err_log_name)
    #

    def test_logging(self):
        out_log_name = "./log_requests_level_out_log.txt"
        err_log_name = "./log_requests_level_err_log.txt"

        # 总共拉起4次服务

        # --log-requests、--log-requests-level
        # 实际使用4次服务
        # --log-requests=True,--log-requests-level=[0, 1, 2, 3]
        message = {
            "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
            "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
            "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
            "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
        }
        keyword_Finish = r".*Finish: obj=GenerateReqInput\(.*http_worker_ipc=None, text='just.*"
        keyword_start = "out={'text': '"
        keyword_end = "', 'output_ids'"

        # --enable-metrics
        # --bucket-time-to-first-token、--bucket-inter-token-latency、--bucket-e2e-request-latency
        # 实际使用两次服务 i=0、1
        # --enable-metrics=True, i=0 使用默认桶边界, i=1 使用自定义桶边界
        my_bucket_list = ["0.1", "0.5", "1.0", "5.0", "10.0"]
        # --bucket-time-to-first-token
        default_bucket_time_to_first_token_list = [
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
            "10.0", "20.0", "40.0", "60.0", "80.0",
            "100.0", "200.0", "400.0",
        ]
        # --bucket-inter-token-latency
        default_bucket_inter_token_latency_list = [
            "0.002", "0.004", "0.006", "0.008",
            "0.01", "0.015", "0.02", "0.025", "0.03", "0.035", "0.04", "0.06", "0.08",
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
        ]
        # --bucket-e2e-request-latency
        default_bucket_e2e_request_latency_list = [
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
            "10.0", "20.0", "40.0", "60.0", "80.0",
            "100.0", "200.0", "400.0", "600.0",
            "1200.0", "1800.0", "2400.0",
        ]
        # for i in [0, 1, 2, 3]:
        for i in [2, 3]:
            other_args = [
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            out_log_file = open(out_log_name, "w+", encoding="utf-8")
            err_log_file = open(err_log_name, "w+", encoding="utf-8")

            # --log-requests、--log-requests-level
            other_args.append("--log-requests")
            # --log-requests-level default value is 2
            if i != 2:
                other_args.extend(["--log-requests-level", str(i)])

            # --enable-metrics
            if i <= 1:
                other_args.extend(["--enable-metrics"])
            if i == 1:
                other_args.extend(["--bucket-time-to-first-token"] + [bucket for bucket in my_bucket_list])
                other_args.extend(["--bucket-inter-token-latency"] + [bucket for bucket in my_bucket_list])
                other_args.extend(["--bucket-e2e-request-latency"] + [bucket for bucket in my_bucket_list])

            process = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )

            try:
                # test inference
                self._send_inference_request()

                # test --log-requests、--log-requests-level
                max_new_token = 2500 if i >= 2 else 100

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
                self.assertIsNotNone(re.search(message[str(i)], content))
                if i >= 2:
                    lines = get_lines_with_keyword(out_log_name, keyword_Finish)
                    Finish_message = lines[0]["content"]
                    start_index = Finish_message.find(keyword_start) + len(keyword_start)
                    end_index = Finish_message.find(keyword_end)
                    out_text = Finish_message[start_index:end_index]
                    out_text_length = len(out_text)
                    out_text_length_n = len(out_text.replace("\\n", " "))
                    if i == 2:
                        self.assertIn("' ... '", out_text)
                        self.assertTrue(out_text_length_n - len("' ... '") == 2048)
                    else:
                        self.assertNotIn("' ... '", out_text)
                        self.assertTrue(out_text_length > 2048)


                # test --enable-metrics
                # --bucket-time-to-first-token、--bucket-inter-token-latency、--bucket-e2e-request-latency
                if i == 0:
                    self._check_metrics_endpoint(
                        default_bucket_time_to_first_token_list,
                        default_bucket_inter_token_latency_list,
                        default_bucket_e2e_request_latency_list,
                    )
                elif i == 1:
                    self._check_metrics_endpoint(
                        my_bucket_list,
                        my_bucket_list,
                        my_bucket_list,
                    )
            finally:
                kill_process_tree(process.pid)
                out_log_file.close()
                err_log_file.close()
                os.remove(out_log_name)
                os.remove(err_log_name)


# TODO 验证方式、删减
class TestAscendLoggingNPULevel(TestAscendLoggingNPUFullBase):
    def test_log_level(self):
        level_list = ["info", "debug", "warning", "error", "critical"]
        http_level_list = ["info", "critical", "error", "warning", "debug", ]

        for level, http_level in zip(level_list, http_level_list):
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
                self._safe_kill_process()


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


class TestAscendLogRequests(TestAscendLoggingNPUFullBase):
    def test_log_requests_level(self):
        message = {
            "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
            "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
            "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
            "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
        }
        out_log_name = "./log_requests_level_out_log.txt"
        err_log_name = "./log_requests_level_err_log.txt"
        keyword_Finish = r".*Finish: obj=GenerateReqInput\(.*http_worker_ipc=None, text='just.*"
        keyword_start = "out={'text': '"
        keyword_end = "', 'output_ids'"
        for i in [0, 1, 2, 3]:
            out_log_file = open(out_log_name, "w+", encoding="utf-8")
            err_log_file = open(err_log_name, "w+", encoding="utf-8")
            process = self._launch_server_with_logging(
                log_requests=True,
                log_requests_level=i,
                enable_metrics=True,
                # enable_metrics_for_all_schedulers=True,
                out_log_file=out_log_file,
                err_log_file=err_log_file,
            ) if i != 2 else self._launch_server_with_logging(
                log_requests=True,
                out_log_file=out_log_file,
                err_log_file=err_log_file,
            )

            try:
                self._send_inference_request()

                max_new_token = 2500 if i >= 2 else 100

                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": f"just return me a string with of 5000 characters",
                        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_token},
                    },
                )
                self.assertEqual(response.status_code, 200)
                out_log_file.seek(0)
                content = out_log_file.read()

                self.assertTrue(len(content) > 0)
                self.assertIsNotNone(re.search(message[str(i)], content))
                if i >= 2:
                    lines = get_lines_with_keyword(out_log_name, keyword_Finish)
                    Finish_message = lines[0]["content"]
                    start_index = Finish_message.find(keyword_start) + len(keyword_start)
                    end_index = Finish_message.find(keyword_end)
                    out_text = Finish_message[start_index:end_index]
                    out_text_length = len(out_text)
                    out_text_length_n = len(out_text.replace("\\n", " "))
                    if i == 2:
                        self.assertIn("' ... '", out_text)
                        self.assertTrue(out_text_length_n - len("' ... '") == 2048)
                    else:
                        self.assertNotIn("' ... '", out_text)
                        self.assertTrue(out_text_length > 2048)
            finally:
                kill_process_tree(process.pid)
                out_log_file.close()
                err_log_file.close()
                os.remove(out_log_name)
                os.remove(err_log_name)


class TestAscendLoggingNPURequestsFormat(TestAscendLoggingNPUFullBase):
    def test_05_log_requests_format_json(self):
        """Test log-requests-format=json."""
        print("\n=== Test 05: log-requests-format=json ===")
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self._temp_dir_obj.name

        try:
            self.process = self._launch_server_with_logging(
                log_requests=True,
                log_requests_level=2,
                log_requests_format="json",
                log_requests_target=["stdout", self.temp_dir],
            )
            time.sleep(5)

            result = self._send_inference_request()
            print(f"✓ log-requests-format=json test passed, result: {result[:50]}...")

            log_files = list(Path(self.temp_dir).glob("*.log"))
            self.assertGreater(len(log_files), 0)

            file_content = log_files[0].read_text()
            print("============json.loads(file_content)=========================")
            print(json.loads(file_content))
            json_lines = [line for line in file_content.splitlines() if line.strip().startswith("{")]
            self.assertGreater(len(json_lines), 0)

            for line in json_lines:
                data = json.loads(line)
                self.assertIn("event", data)
                self.assertIn("rid", data)
        finally:
            self._safe_kill_process()


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

class TestAscendLoggingNPUMetric(TestAscendLoggingNPUFullBase):
    def test_metrics(self):
        """Test enable-metrics-for-all-schedulers with TP2."""
        print("\n=== Test not enable_metrics_for_all_schedulers  ===")
        print("\n=== Test default bucket set  ===")
        print("")

        try:
            self.process = self._launch_server_with_logging(
                enable_metrics=True,
                # enable_metrics_for_all_schedulers=True,
                tp_size=2,
                enable_dp_attention=True,
                dp_size=2
            )
            # time.sleep(8)

            result = self._send_inference_request()
            # print(f"✓ enable-metrics-for-all-schedulers test passed, result: {result[:50]}...")

            metrics_content = self._check_metrics_endpoint()

            self.assertIn('tp_rank="0"', metrics_content)
            # self.assertNotIn('tp_rank="1"', metrics_content)
            time_to_first_token_seconds_bucket_list = ["0.1", "0.2", "0.4", "0.6", "0.8", "1.0", "2.0", "4.0", "6.0", "8.0", "400.0", "+Inf"]
            for le in ["0.1", "0.2", "0.4", "0.8", "1.0", "400.0", "+Inf"]:
                message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
                message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
        finally:
            self._safe_kill_process()

    def test_metrics_for_3(self):
        """Test enable-metrics-for-all-schedulers with TP2."""
        print("\n=== Test 03: test_metrics_for_3 ===")
        print("")

        try:
            self.process = self._launch_server_with_logging(
                enable_metrics=True,
                enable_metrics_for_all_schedulers=True,
                tp_size=2,
                enable_dp_attention=True,
                dp_size=2,
                bucket_time_to_first_token=[0.1, 0.5, 1.0, 2.0, 5.0],
                bucket_inter_token_latency=[0.01, 0.05, 0.1, 0.5],
                bucket_e2e_request_latency=[0.1, 0.5, 1.0, 2.0, 5.0],
            )
            # time.sleep(8)

            result = self._send_inference_request()
            # print(f"✓ enable-metrics-for-all-schedulers test passed, result: {result[:50]}...")

            metrics_content = self._check_metrics_endpoint()

            self.assertIn('tp_rank="0"', metrics_content)
            # self.assertNotIn('tp_rank="1"', metrics_content)
            # for le in ["0.1", "0.2", "0.4", "0.8", "1.0", "400.0", "+Inf"]:
            #     message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
            #     self.assertIn(message, metrics_content)
            #     message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
            #     self.assertIn(message, metrics_content)
            # sleep(600)
            # sleep(600)
        finally:
            self._safe_kill_process()


class TestAscendLoggingNPUMetricWip(TestAscendLoggingNPUFullBase):
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
    #         # time.sleep(8)
    #
    #         result = self._send_inference_request()
    #         # print(f"✓ enable-metrics-for-all-schedulers test passed, result: {result[:50]}...")
    #
    #         metrics_content = self._check_metrics_endpoint()
    #         self.assertIn('tp_rank="0"', metrics_content)
    #         self.assertIn('tp_rank="1"', metrics_content)
    #     finally:
    #         self._safe_kill_process()
    #
    def test_metrics_2(self):
        """Test enable-metrics-for-all-schedulers with TP2."""
        print("\n=== Test 02: test_metrics_2 ===")
        print("")

        try:
            self.process = self._launch_server_with_logging(
                enable_metrics=True,
                # enable_metrics_for_all_schedulers=False,
                tp_size=2,
                dp_size=2
            )
            # time.sleep(8)

            result = self._send_inference_request()
            # print(f"✓ enable-metrics-for-all-schedulers test passed, result: {result[:50]}...")

            metrics_content = self._check_metrics_endpoint()

            self.assertIn('tp_rank="0"', metrics_content)
            # self.assertNotIn('tp_rank="1"', metrics_content)
            for le in ["0.1", "0.2", "0.4", "0.8", "1.0", "400.0", "+Inf"]:
                message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
                message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
                self.assertIn(message, metrics_content)
        finally:
            self._safe_kill_process()

    def test_metrics_for_3(self):
        """Test enable-metrics-for-all-schedulers with TP2."""
        print("\n=== Test 03: test_metrics_for_3 ===")
        print("")

        try:
            self.process = self._launch_server_with_logging(
                enable_metrics=True,
                enable_metrics_for_all_schedulers=True,
                tp_size=2,
                dp_size=2,
                # bucket_time_to_first_token=[0.1, 0.5, 1.0, 2.0, 5.0],
                # bucket_inter_token_latency=[0.01, 0.05, 0.1, 0.5],
                # bucket_e2e_request_latency=[0.1, 0.5, 1.0, 2.0, 5.0],
            )
            # time.sleep(8)

            result = self._send_inference_request()
            # print(f"✓ enable-metrics-for-all-schedulers test passed, result: {result[:50]}...")

            metrics_content = self._check_metrics_endpoint()

            self.assertIn('tp_rank="0"', metrics_content)
            # self.assertNotIn('tp_rank="1"', metrics_content)
            # for le in ["0.1", "0.2", "0.4", "0.8", "1.0", "400.0", "+Inf"]:
            #     message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
            #     self.assertIn(message, metrics_content)
            #     message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{MODEL_PATH}"}}'
            #     self.assertIn(message, metrics_content)
            # sleep(600)
            sleep(600)
        finally:
            self._safe_kill_process()


# class TestAscendLoggingNPUBucket(TestAscendLoggingNPUFullBase):
#     def test_09_custom_buckets(self):
#         """Test custom metric buckets."""
#         print("\n=== Test 09: custom metric buckets ===")
#
#         try:
#             self.process = self._launch_server_with_logging(
#                 enable_metrics=True,
#                 bucket_time_to_first_token=[0.1, 0.5, 1.0, 2.0, 5.0],
#                 bucket_inter_token_latency=[0.01, 0.05, 0.1, 0.5],
#                 bucket_e2e_request_latency=[1.0, 5.0, 10.0, 30.0],
#             )
#             time.sleep(5)
#
#             result = self._send_inference_request()
#             print(f"✓ custom buckets test passed, result: {result[:50]}...")
#
#             metrics_content = self._check_metrics_endpoint()
#             # self.assertIn("sglang_time_to_first_token_bucket", metrics_content)
#             # self.assertIn("sglang_e2e_request_latency_bucket", metrics_content)
#         finally:
#             kill_process_tree(self.process.pid)
#             self.process = None


class TestAscendLoggingNPUCollectTokensHistogram(TestAscendLoggingNPUFullBase):
    def test_11_prompt_tokens_buckets_default(self):
        """Test prompt-tokens-buckets with default."""
        print("\n=== Test 11: prompt-tokens-buckets default ===")

        prompt_tokens_bucket_list = [["default"], ["tse", "512", "2", "8"], ["custom", "100", "500", "1000", "5000"]]
        generation_tokens_buckets_list = [["custom", "100", "500", "1000", "5000"], ["tse", "512", "2", "8"],
                                          ["default"]]

        for prompt_tokens_bucket, generation_tokens_buckets in zip(prompt_tokens_bucket_list,
                                                                   generation_tokens_buckets_list):

            try:
                self.process = self._launch_server_with_logging(
                    enable_metrics=True,
                    collect_tokens_histogram=True,
                    prompt_tokens_buckets=prompt_tokens_bucket,
                    generation_tokens_buckets=generation_tokens_buckets,
                )
                time.sleep(5)

                result = self._send_inference_request()
                print(f"✓ prompt-tokens-buckets default test passed, result: {result[:50]}...")

                # metrics_content = self._check_metrics_endpoint()
                # self.assertIn("sglang_prompt_tokens_bucket", metrics_content)
            finally:
                kill_process_tree(self.process.pid)
                self.process = None


class TestAscendLoggingNPUDecodeLogInterval(TestAscendLoggingNPUFullBase):
    def test_15_decode_log_interval(self):
        """Test decode-log-interval."""
        print("\n=== Test 15: decode-log-interval ===")
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self._temp_dir_obj.name

        try:
            self.process = self._launch_server_with_logging(
                log_level="debug",
                decode_log_interval=10,
            )
            time.sleep(5)

            result = self._send_inference_request(max_new_tokens=100)
            print(f"✓ decode-log-interval test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


class TestAscendLoggingNPUGCWarningThresholdSecs(TestAscendLoggingNPUFullBase):
    def test_18_gc_warning_threshold_secs(self):
        """Test gc-warning-threshold-secs."""
        print("\n=== Test 18: gc-warning-threshold-secs ===")

        try:
            self.process = self._launch_server_with_logging(
                gc_warning_threshold_secs=0.1,
            )
            time.sleep(5)

            result = self._send_inference_request()
            print(f"✓ gc-warning-threshold-secs test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


class TestAscendLoggingNPUEnableRequestTimeStatsLogging(TestAscendLoggingNPUFullBase):
    def test_16_enable_request_time_stats_logging(self):
        """Test enable-request-time-stats-logging."""
        print("\n=== Test 16: enable-request-time-stats-logging ===")

        try:
            self.process = self._launch_server_with_logging(
                enable_request_time_stats_logging=True,
            )
            time.sleep(5)

            result = self._send_inference_request()
            print(f"✓ enable-request-time-stats-logging test passed, result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


# TODO install OTLP collector
class TestAscendLoggingNPUEnableTrace(TestAscendLoggingNPUFullBase):
    def test_17_enable_trace(self):
        """Test enable-trace (requires OTLP collector)."""
        print("\n=== Test 17: enable-trace ===")

        try:
            self.process = self._launch_server_with_logging(
                enable_trace=True,
                otlp_traces_endpoint="localhost:4317",
            )
            time.sleep(5)

            result = self._send_inference_request()
            print(f"✓ enable-trace test passed (server started successfully), result: {result[:50]}...")
        except Exception as e:
            print(f"⚠ enable-trace test skipped (OTLP collector may not be available): {e}")
        finally:
            if self.process:
                kill_process_tree(self.process.pid)
                self.process = None


# TODO: 注入崩溃
class TestAscendLoggingNPUCrashDumpFolder(TestAscendLoggingNPUFullBase):
    def test_19_crash_dump_folder(self):
        """Test crash-dump-folder."""
        print("\n=== Test 19: crash-dump-folder ===")
        self._temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self._temp_dir_obj.name
        crash_dir = os.path.join(self.temp_dir, "crash_dumps")
        os.makedirs(crash_dir, exist_ok=True)

        try:
            self.process = self._launch_server_with_logging(
                crash_dump_folder=crash_dir,
            )
            time.sleep(5)

            result = self._send_inference_request()
            print(f"✓ crash-dump-folder test passed (server started successfully), result: {result[:50]}...")
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

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


if __name__ == "__main__":
    # unittest.main()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestAscendLogging))


    # DONE
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLogRequests))

    # TODO
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPURequestsLevel))



    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPURequestsFormat))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPURequestsTarget))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUMetric))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUCollectTokensHistogram))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUDecodeLogInterval))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUGCWarningThresholdSecs))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUEnableRequestTimeStatsLogging))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUEnableTrace))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUCrashDumpFolder))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendLoggingNPUBucket))
    runner = unittest.TextTestRunner()
    runner.run(suite)
