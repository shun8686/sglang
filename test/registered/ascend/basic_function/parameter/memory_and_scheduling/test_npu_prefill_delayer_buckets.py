import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNpuPrefillDelayerBuckets(CustomTestCase):
    """Test Case: Verify that when the service startup parameters --prefill-delayer-forward-passes-buckets
    and --prefill-delayer-wait-seconds-buckets are configured, querying the /metrics interface shows
    that the statistical groups are consistent with the configuration

    [Test Category] Parameter
    [Test Target] --prefill-delayer-forward-passes-buckets; --prefill-delayer-wait-seconds-buckets
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.forward_passes_buckets = [10.0, 20.0, 30.0]
        cls.wait_seconds_buckets = [10.0, 20.0, 30.0]
        forward_buckets_args = [str(int(b)) for b in cls.forward_passes_buckets]
        wait_seconds_args = [str(int(b)) for b in cls.wait_seconds_buckets]
        other_args = [
            "--tp-size",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-prefill-delayer",
            "--enable-dp-attention",
            "--dp-size",
            "2",
            "--prefill-delayer-max-delay-passes",
            "100",
            "--prefill-delayer-forward-passes-buckets",
            *forward_buckets_args,
            "--prefill-delayer-wait-seconds-buckets",
            *wait_seconds_args,
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _get_metrics_lines(self):
        try:
            response = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/metrics",
                timeout=10
            )
            response.raise_for_status()
            lines = []
            for line in response.text.splitlines():
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith("#"):
                    lines.append(stripped_line)
            return lines
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to fetch metrics: {str(e)}")

    def _check_bucket_in_metric_line(self, metric_name, expected_buckets):
        metrics_lines = self._get_metrics_lines()
        # Define target feature string (metric name + _bucket{)
        target_metric_feature = f"{metric_name}_bucket{{"

        # Filter all lines containing the target metric's bucket configuration
        target_lines = [
            line for line in metrics_lines
            if target_metric_feature in line and 'le="' in line
        ]

        self.assertNotEqual(
            len(target_lines), 0,
            f"No lines found for metric {metric_name} in /metrics response. "
            f"Checked feature: {target_metric_feature}"
        )

        # Extract all values of the le label
        le_values = []
        for line in target_lines:
            matches = re.findall(r'le="([\d\.]+)"', line)
            le_values.extend([float(v) for v in matches])

        print(le_values)

        unique_le_values = sorted(list(set(le_values)))
        expected_buckets_sorted = sorted(expected_buckets)

        # Verify all expected bucket values exist
        for bucket in expected_buckets_sorted:
            self.assertIn(
                bucket,
                unique_le_values,
                f"Expected bucket {bucket} not found in {metric_name}. "
                f"Expected: {expected_buckets_sorted}, Actual: {unique_le_values}"
            )

    def test_buckets_params(self):
        # Verify forward passes buckets take effect
        self._check_bucket_in_metric_line(
            "sglang:prefill_delayer_wait_forward_passes",
            self.forward_passes_buckets
        )

        # Verify wait seconds buckets take effect
        self._check_bucket_in_metric_line(
            "sglang:prefill_delayer_wait_seconds",
            self.wait_seconds_buckets
        )


if __name__ == "__main__":
    unittest.main()
