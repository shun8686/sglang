import logging
import os
import json

from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH, run_command
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
            os.path.abspath("."),
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

    def test_metrics_single_request(self):
        """Send a single request"""
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
        metrics_dir = Path(os.path.abspath("."))
        metrics_files = list(metrics_dir.glob("sglang-request-metrics-*.log"))

        self.assertGreater(len(metrics_files), 0, "should generate file")
        logging.warning(f"\n🔍 Matched metrics files: {metrics_files}")
        with open(metrics_files, "r", encoding="utf-8") as f:
            log_content = f.read()
            # Split by line (log may contain multiple JSON entries)
            log_lines = [
                line.strip() for line in log_content.split("\n") if line.strip()
            ]
            # Get last valid log entry (latest request)
            last_log = log_lines[-1] if log_lines else ""
            # Clean line breaks and extra spaces
            clean_content = last_log.replace("\n", "").replace("  ", " ").strip()
            logging.warning(f"\n📝 Cleaned latest log content:\n{clean_content[:800]}...")
        try:
            # Parse outer JSON
            log_data = json.loads(clean_content)
            self.assertIn("request_parameters", log_data)
            # Parse request_parameters field (string to JSON)
            req_params = json.loads(log_data["request_parameters"])
            self.assertIn("text", req_params)
            # Extract sampling_params
            self.assertIn("sampling_params", req_params)
            self.assertIn("prompt_tokens", req_params)
            self.assertIn("completion_tokens", req_params)

        except json.JSONDecodeError as e:
            self.fail(
                f"❌ JSON parsing failed: {e}, Original content: {clean_content[:500]}"
            )

    def test_metrics_multiple_request(self):
        """Send a multiple request"""
        for i in range(5):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": f"Explain the concept of machine learning in detail {i}",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 100
                    },
                },
            )
            self.assertEqual(response.status_code, 200)

            metrics_dir = Path(os.path.abspath("."))
            metrics_files = list(metrics_dir.glob("sglang-request-metrics-*.log"))

            self.assertGreater(len(metrics_files), 5, "It should contain 5 requests")
















