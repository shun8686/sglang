import os
import json
import shutil
import tempfile
import time
import unittest

import logging
import requests
from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMetricsExporter(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.metrics_dir = tempfile.mkdtemp(prefix="sglang-request-metrics-")
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--tp-size",
            2,
            "--disable-cuda-graph",
            "--export-metrics-to-file",
            "--export-metrics-to-file-dir",
            cls.metrics_dir,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.exists(cls.metrics_dir):
            shutil.rmtree(cls.metrics_dir)

    def _get_metrics_files(self):
        files = []
        if os.path.exists(self.metrics_dir):
            for file in os.listdir(self.metrics_dir):
                if file.startswith("sglang-request-metrics-") and file.endswith(
                    ".log"
                ):
                    files.append(os.path.join(self.metrics_dir, file))
        return sorted(files)


    def _read_metrics_records(self, files):
        records = []
        for file in files:
            with open(file, 'r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError:
                            pass
        return records

    def test_metrics_single_request(self):
        """Send a single request"""
        logging.warning("****test1: Send a single request")
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100
                },
            },
        )

        self.assertEqual(response.status_code, 200)

        time.sleep(1)
        metrics_files = self._get_metrics_files()
        self.assertGreater(len(metrics_files), 0, "Generate specified file")

        metrics_records = self._read_metrics_records(metrics_files)
        self.assertGreater(len(metrics_records), 0, "It contains at least one record")

        record = metrics_records[0]
        self.assertIn("request_parameters", record)
        self.assertIn("prompt_tokens", record)
        self.assertIn("completion_tokens", record)

        request_parameters = json.loads(record["request_parameters"])
        self.assertIn("text", request_parameters)
        self.assertIn("sampling_params", request_parameters)

        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)

    def test_metrics_multiple_request(self):
        """Test multiple requests"""
        logging.warning("****test2: Test multiple requests")
        for i in range(5):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": f"The capital of France is {i}",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 100},
                },
            )
            self.assertEqual(response.status_code, 200)
        time.sleep(1)

        metrics_files = self._get_metrics_files()
        metrics_records = self._read_metrics_records(metrics_files)

        self.assertEqual(len(metrics_records), 5, "It should contain 5 requests")
        for record in metrics_records:
            self.assertIn("request_parameters", record)
            self.assertIn("prompt_tokens", record)
            self.assertIn("completion_tokens", record)

        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)

    def test_metrics_health_check(self):
        """Test health check request not exported"""
        logging.warning("****test3: Test health check request not exported")
        response = requests.get(f"{self.base_url}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.get(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 118},
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(1)
        metrics_files = self._get_metrics_files()
        metrics_records = self._read_metrics_records(metrics_files)

        self.assertEqual(len(metrics_records), 1, "Includes a normal request record")
        for record in metrics_records:
            request_parameters = json.loads(record["request_parameters"])
            rid = request_parameters.get("rid", "")
            self.assertNotIn("HEALTH_CHECK", rid, "Health check requests should not be included.")

        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)

    def test_different_sampling_params(self):
        """Test different sampling parameters and request export"""
        logging.warning("****test4: Test different sampling parameters and request export")
        sampling_cinfigs = [
            {"temperature": 0, "max_new_tokens": 32},
            {"temperature": 0.5, "max_new_tokens": 64},
            {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 127},
        ]

        for config in sampling_cinfigs:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": config,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200)

        time.sleep(1)

        metrics_files = self._get_metrics_files()
        metrics_records = self._read_metrics_records(metrics_files)

        self.assertEqual(len(metrics_records), 3, "Contains 3 request records")
        for i, record in enumerate(metrics_records):
            request_parameters = json.loads(record["request_parameters"])
            recorded_sampling = request_parameters.get("sampling_params", {})
            for key, param_value in sampling_cinfigs[i].items():
                self.assertIn(key, recorded_sampling)
                self.assertEqual(recorded_sampling[key], param_value)

        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)

    def test_stream_and_no_stream(self):
        """Test streaming and non-streaming request exports"""
        logging.warning("****test5: Test streaming and non-streaming request exports")
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Stream test",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            stream=True,
        )
        for _ in response.iter_lines(decode_unicode=False):
            pass

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Non-stream test",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(1)
        metrics_files = self._get_metrics_files()
        metrics_records = self._read_metrics_records(metrics_files)
        self.assertEqual(len(metrics_records), 2, "It should contain 2 requests")

        for record in metrics_records:
            self.assertIn("request_parameters", record)
            self.assertIn("prompt_tokens", record)
            self.assertIn("completion_tokens", record)

            request_parameters = json.loads(record["request_parameters"])
            self.assertIn("stream", request_parameters["sampling_params"])

        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=21000,
        )
        run_eval(args)
        metrics_files = self._get_metrics_files()
        metrics_records = self._read_metrics_records(metrics_files)
        logging.warning(f"Batch processing requests successful.")
        for record in metrics_records:
            self.assertIn("request_parameters", record)

        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)


        # with open(metrics_file, 'r', encoding="utf-8") as f:
        #     log_content = f.read()
        #     # Split by line (log may contain multiple JSON entries)
        #     log_lines = [
        #         line.strip() for line in log_content.split("\n") if line.strip()
        #     ]
        #     # Get last valid log entry (latest request)
        #     last_log = log_lines[-1] if log_lines else ""
        #     # Clean line breaks and extra spaces
        #     clean_content = last_log.replace("\n", "").replace("  ", " ").strip()
        #     logging.warning(f"\n📝 Cleaned latest log content:\n{clean_content[:800]}...")
        # try:
        #     # Parse outer JSON
        #     log_data = json.loads(clean_content)
        #     self.assertIn("request_parameters", log_data)
        #     # Parse request_parameters field (string to JSON)
        #     req_params = json.loads(log_data["request_parameters"])
        #     self.assertIn("text", req_params)
        #     # Extract sampling_params
        #     self.assertIn("sampling_params", req_params)
        #     self.assertIn("prompt_tokens", req_params)
        #     self.assertIn("completion_tokens", req_params)
        #
        # except json.JSONDecodeError as e:
        #     self.fail(
        #         f"❌ JSON parsing failed: {e}, Original content: {clean_content[:500]}"
        #     )

    # def test_metrics_multiple_request(self):
    #     """Send a multiple request"""
    #     for i in range(5):
    #         response = requests.post(
    #             f"{self.base_url}/generate",
    #             json={
    #                 "text": f"Explain the concept of machine learning in detail {i}",
    #                 "sampling_params": {
    #                     "temperature": 0,
    #                     "max_new_tokens": 100
    #                 },
    #             },
    #         )
    #         self.assertEqual(response.status_code, 200)
    #
    #         metrics_dir = Path(os.path.abspath("."))
    #         metrics_files = list(metrics_dir.glob("sglang-request-metrics-*.log"))
    #
    #         self.assertGreater(len(metrics_files), 5, "It should contain 5 requests")

if __name__ == "__main__":
    unittest.main()
















