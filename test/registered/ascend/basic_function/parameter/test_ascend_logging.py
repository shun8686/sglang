import os
import re
import tempfile
import threading
import unittest
from pathlib import Path
from time import sleep

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH as MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-2-npu-a3", nightly=True)


class TestAscendLoggingNPUFullBase(CustomTestCase):
    """Testcase：Verify the correct functionality of parameters in the logging feature.

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level; --log-requests-target; --uvicorn-access-log-exclude-prefixes;
    --enable-metrics; --enable-metrics-for-all-scheduler;
    --bucket-time-to-first-token; --bucket-inter-token-latency; --bucket-e2e-request-latency;
    --collect-tokens-histogram; --prompt-tokens-buckets; --generation-tokens-buckets;
    --tokenizer-metrics-custom-labels-header; --tokenizer-metrics-allowed-custom-labels;
    --gc-warning-threshold-secs
    """

    @classmethod
    def setUpClass(cls):
        # Note: During the test, a fixed-length string needs to be returned.
        # The returned value of the same prompt may vary depending on the model
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.out_log_name = "./log_requests_level_out_log.txt"
        cls.err_log_name = "./log_requests_level_err_log.txt"
        cls.out_log_file = open(cls.out_log_name, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.err_log_name, "w+", encoding="utf-8")
        cls.test_prompt = "What is the capital of France?"
        cls.expected_output = "Paris"

        cls.prepare_args_related_data()

        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        os.remove(cls.out_log_name)
        cls.err_log_file.close()
        os.remove(cls.err_log_name)

    @classmethod
    def prepare_args_related_data(cls):
        # --log-requests
        # Basic log content
        cls.message = r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, .*"

        # --log-requests-level: Log content at different log level
        cls.log_request_message_dict = {
            "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
            "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
            "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
            "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
        }
        # Log content at the default log level (2)
        cls.keyword_Finish = r".*Finish: obj=GenerateReqInput\(.*http_worker_ipc=None, text='just.*"
        cls.keyword_start = "out={'text': '"
        cls.keyword_end = "', 'output_ids'"

        # --uvicorn-access-log-exclude-prefixes
        cls.log_exclude_prefixes = ["/health", "/get_server_info"]

        # --enable-metrics
        ## --bucket-time-to-first-token、--bucket-inter-token-latency、--bucket-e2e-request-latency
        ### Default bucket boundaries
        cls.default_time_to_first_token_bucket = [
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
            "10.0", "20.0", "40.0", "60.0", "80.0",
            "100.0", "200.0", "400.0",
        ]
        cls.default_inter_token_latency_bucket = [
            "0.002", "0.004", "0.006", "0.008",
            "0.01", "0.015", "0.02", "0.025", "0.03", "0.035", "0.04", "0.06", "0.08",
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
        ]
        cls.default_e2e_request_latency_bucket = [
            "0.1", "0.2", "0.4", "0.6", "0.8",
            "1.0", "2.0", "4.0", "6.0", "8.0",
            "10.0", "20.0", "40.0", "60.0", "80.0",
            "100.0", "200.0", "400.0", "600.0",
            "1200.0", "1800.0", "2400.0",
        ]
        ### Custom bucket boundaries
        cls.my_bucket = ["0.1", "0.5", "1.0", "5.0", "10.0"]

        ## --collect-tokens-histogram
        ### --prompt-tokens-buckets、--generation-tokens-bucket
        #### Default token bucket boundaries
        #### The default bucket boundaries of prompt tokens and generated tokens are consistent.
        cls.default_tokens_bucket = [
            "100.0", "300.0", "500.0", "700.0",
            "1000.0", "1500.0", "2000.0", "3000.0", "4000.0", "5000.0", "6000.0", "7000.0", "8000.0", "9000.0",
            "10000.0", "12000.0", "15000.0", "20000.0", "22000.0", "25000.0",
            "30000.0", "35000.0", "40000.0", "66000.0", "99000.0",
            "132000.0", "300000.0", "600000.0", "900000.0",
            "1.1e+06",
        ]
        #### Custom bucket boundaries
        cls.my_tokens_bucket = [
            "100.0", "1000.0", "10000.0", "100000.0", "300000.0", "600000.0", "900000.0",
        ]
        #### Two-Sided Exponential bucket Strategy
        cls.my_tse_set = ["1000", "2", "8"]
        cls.my_tse_bucket = ["984.0", "992.0", "996.0", "998.0", "1000.0", "1002.0", "1004.0", "1008.0", "1016.0"]

        ## --tokenizer-metrics-custom-labels-header、--tokenizer-metrics-allowed-custom-labels
        cls.labels_header = "X-Metrics-Labels"
        cls.my_label = "business_line"

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

    def _test_log_exclude_prefixes(self, if_enable, out_log_file):
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)
        response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
        self.assertEqual(response.status_code, 200)
        out_log_file.seek(0)
        content = out_log_file.read()
        health_message = '"GET /health HTTP/1.1" 200 OK'
        get_server_info_message = '"GET /get_server_info HTTP/1.1" 200 OK'
        if if_enable:
            self.assertNotIn(health_message, content)
            self.assertNotIn(get_server_info_message, content)
        else:
            self.assertIn(health_message, content)
            self.assertIn(get_server_info_message, content)

    @classmethod
    def _prepare_log_requests_target_obj(cls):
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name

        cls.temp_level1_dir = os.path.join(cls.temp_dir, "level1")
        cls.temp_level2_dir = os.path.join(cls.temp_dir, "level2")
        cls.temp_level3_dir = os.path.join(cls.temp_dir, "level3")

        os.makedirs(cls.temp_level3_dir, exist_ok=True)

        target_config = ["stdout", cls.temp_dir, cls.temp_level3_dir]
        cls.other_args.extend(["--log-requests-target"] + target_config)

    def _test_log_requests_target(self):
        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)

        file_content = log_files[0].read_text()
        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)

        log_files = list(Path(self.temp_level3_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)

        file_content = log_files[0].read_text()
        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)

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

    def _test_enable_metrics_for_all_scheduler(self, if_enable):
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        message_0 = (f'sglang:num_decode_transfer_queue_reqs{{engine_type="unified",model_name="{self.model}"'
                     f',moe_ep_rank="0",pp_rank="0",tp_rank="0"}}')
        message_1 = (f'sglang:num_decode_transfer_queue_reqs{{engine_type="unified",model_name="{self.model}"'
                     f',moe_ep_rank="0",pp_rank="0",tp_rank="1"}}')
        self.assertIn(message_0, response.text)
        if if_enable:
            self.assertIn(message_1, response.text)
        else:
            self.assertNotIn(message_1, response.text)

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


class TestAscendLoggingCase0(TestAscendLoggingNPUFullBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.other_args.append("--log-requests")
        cls.log_requests_level = 0
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])

        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--tp-size", 2])

        cls.expected_time_to_first_token_bucket = cls.default_time_to_first_token_bucket
        cls.expected_inter_token_latency_bucket = cls.default_inter_token_latency_bucket
        cls.expected_e2e_request_latency_bucket = cls.default_e2e_request_latency_bucket

        cls.other_args.extend(["--collect-tokens-histogram"])

        cls.expected_prompt_tokens_bucket = cls.default_tokens_bucket
        cls.expected_generation_tokens_bucket = cls.default_tokens_bucket

        cls.other_args.extend(["--gc-warning-threshold-secs", "0.01"])

        cls._prepare_log_requests_target_obj()

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_0(self):
        self._test_inference_function()

        self._test_log_requests_level(self.log_requests_level, self.out_log_file)

        self._test_log_requests_target()

        # test --uvicorn-access-log-exclude-prefixes
        self._test_log_exclude_prefixes(False, self.out_log_file)

        self._test_enable_metrics_for_all_scheduler(False)

        self._test_metrics(
            expected_time_to_first_token_bucket=self.expected_time_to_first_token_bucket,
            expected_inter_token_latency_bucket=self.expected_inter_token_latency_bucket,
            expected_e2e_request_latency_bucket=self.expected_e2e_request_latency_bucket,
            expected_prompt_tokens_bucket=self.expected_prompt_tokens_bucket,
            expected_generation_tokens_bucket=self.expected_generation_tokens_bucket,
        )

        self._test_log_requests_target()

        self._test_gc_warning_threshold(self.err_log_file)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._temp_dir_obj.cleanup()


class TestAscendLoggingCase1(TestAscendLoggingNPUFullBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.other_args.append("--log-requests")
        cls.log_requests_level = 1
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])

        cls.other_args.extend(["--uvicorn-access-log-exclude-prefixes"] + cls.log_exclude_prefixes)

        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--tp-size", 2])
        cls.other_args.extend(["--enable-metrics-for-all-scheduler"])

        cls.other_args.extend(["--bucket-time-to-first-token"] + cls.my_bucket)
        cls.other_args.extend(["--bucket-inter-token-latency"] + cls.my_bucket)
        cls.other_args.extend(["--bucket-e2e-request-latency"] + cls.my_bucket)
        cls.expected_time_to_first_token_bucket = cls.my_bucket
        cls.expected_inter_token_latency_bucket = cls.my_bucket
        cls.expected_e2e_request_latency_bucket = cls.my_bucket

        cls.other_args.extend(["--collect-tokens-histogram"])

        cls.other_args.extend(["--prompt-tokens-buckets"] + ["custom"] + cls.my_tokens_bucket)
        cls.other_args.extend(["--generation-tokens-buckets"] + ["custom"] + cls.my_tokens_bucket)
        cls.expected_prompt_tokens_bucket = cls.my_tokens_bucket
        cls.expected_generation_tokens_bucket = cls.my_tokens_bucket

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_1(self):
        self._test_inference_function()

        self._test_log_requests_level(self.log_requests_level, self.out_log_file)

        # test --uvicorn-access-log-exclude-prefixes
        self._test_log_exclude_prefixes(True, self.out_log_file)

        self._test_enable_metrics_for_all_scheduler(True)

        self._test_metrics(
            expected_time_to_first_token_bucket=self.expected_time_to_first_token_bucket,
            expected_inter_token_latency_bucket=self.expected_inter_token_latency_bucket,
            expected_e2e_request_latency_bucket=self.expected_e2e_request_latency_bucket,
            expected_prompt_tokens_bucket=self.expected_prompt_tokens_bucket,
            expected_generation_tokens_bucket=self.expected_generation_tokens_bucket,
        )


class TestAscendLoggingCase2(TestAscendLoggingNPUFullBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.other_args.append("--log-requests")
        cls.log_requests_level = 2
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])

        cls.other_args.extend(["--enable-metrics"])

        cls.other_args.extend(["--collect-tokens-histogram"])

        cls.other_args.extend(["--prompt-tokens-buckets"] + ["tse"] + cls.my_tse_set)
        cls.other_args.extend(["--generation-tokens-buckets"] + ["tse"] + cls.my_tse_set)
        cls.expected_prompt_tokens_bucket = cls.my_tse_bucket
        cls.expected_generation_tokens_bucket = cls.my_tse_bucket

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_2(self):
        self._test_inference_function()

        self._test_log_requests_level(self.log_requests_level, self.out_log_file)

        self._test_metrics(
            expected_prompt_tokens_bucket=self.expected_prompt_tokens_bucket,
            expected_generation_tokens_bucket=self.expected_generation_tokens_bucket,
        )


class TestAscendLoggingCase3(TestAscendLoggingNPUFullBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.other_args.append("--log-requests")
        cls.log_requests_level = 3
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])

        cls.other_args.extend(["--enable-metrics"])

        cls.other_args.extend(["--tokenizer-metrics-custom-labels-header", cls.labels_header])
        cls.other_args.extend(["--tokenizer-metrics-allowed-custom-labels", cls.my_label])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_3(self):
        self._test_inference_function()

        self._test_log_requests_level(self.log_requests_level, self.out_log_file)

        # test --tokenizer-metrics-custom-labels-header、--tokenizer-metrics-allowed-custom-labels
        self._test_log_metrics_tokenizer_label()


if __name__ == "__main__":
    unittest.main()
