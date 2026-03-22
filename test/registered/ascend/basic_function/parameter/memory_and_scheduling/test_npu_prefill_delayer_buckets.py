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
    """Test Case: Verify the accuracy of LLM models under TP+PP hybrid parallelism

    [Test Category] Parameter
    [Test Target] --pp-size; --tp-size
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.forward_passes_buckets = [10.0, 20.0, 30.0]
        cls.wait_seconds_buckets = [1.0, 5.0, 10.0]
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
            int(cls.forward_passes_buckets[0]),
            int(cls.forward_passes_buckets[1]),
            int(cls.forward_passes_buckets[2]),
            "--prefill-delayer-wait-seconds-buckets",
            int(cls.wait_seconds_buckets[0]),
            int(cls.wait_seconds_buckets[1]),
            int(cls.wait_seconds_buckets[2]),
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

    def test_buckets_params(self):
        metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text
        # prefill_delayer_metrics = [
        #     line for line in metrics_text.split("\n") if "prefill_delayer" in line
        # ]
        # print("=== PrefillDelayer Metrics ===")
        # for line in prefill_delayer_metrics:
        #     print(line)
        # 查监控指标，验证 buckets 生效
        # metrics_text = _print_prefill_delayer_metrics(DEFAULT_URL_FOR_TEST, expect_metrics=True)
        # 检查轮次 buckets
        for forward_pass in self.forward_passes_buckets:
            self.assertIn(f'le="{forward_pass}"', metrics_text)
        # 检查时间 buckets
        for wait_seconds_bucket in self.wait_seconds_buckets:
            self.assertIn(f'le="{wait_seconds_bucket}"', metrics_text)


# def _print_prefill_delayer_metrics(base_url: str, expect_metrics: bool) -> str:
#     metrics_response = requests.get(f"{base_url}/metrics")
#     assert metrics_response.status_code == 200
#     metrics_text = metrics_response.text
#     prefill_delayer_metrics = [
#         line for line in metrics_text.split("\n") if "prefill_delayer" in line
#     ]
#     print("=== PrefillDelayer Metrics ===")
#     for line in prefill_delayer_metrics:
#         print(line)
#     if expect_metrics:
#         assert "sglang:prefill_delayer_wait_forward_passes" in metrics_text
#         assert "sglang:prefill_delayer_wait_seconds" in metrics_text
#         assert "sglang:prefill_delayer_outcomes_total" in metrics_text
#     return metrics_text

if __name__ == "__main__":
    unittest.main()
