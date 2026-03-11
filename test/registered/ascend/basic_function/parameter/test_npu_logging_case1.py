import unittest
import requests

from test_npu_logging import TestNPULoggingBase

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=50, suite="nightly-2-npu-a3", nightly=True)


class TestNPULoggingCase1(TestNPULoggingBase):
    """Test case class for validating specific scenarios of logging feature parameters (complementary to Case0/2/3).

    Core Functionality:
        This class tests the behavior of SGLang logging/metrics features under a specific parameter configuration set,
        working with Case0/2/3 to fully cover the parameter value range of the logging feature. The tested configuration includes:
          --log-requests; --log-requests-level=1; --uvicorn-access-log-exclude-prefixes exclude_prefixes_list;
          --enable-metrics; --enable-metrics-for-all-scheduler; --bucket-time-to-first-token custom_bucket_boundary;
          --bucket-inter-token-latency custom_bucket_boundary; --bucket-e2e-request-latency custom_bucket_boundary;
          --collect-tokens-histogram; --prompt-tokens-buckets custom_bucket_boundary; --generation-tokens-buckets custom_bucket_boundary;

        Note:
        1. --tp-size=2 is configured to verify that all TP rank metrics are recorded when --enable-metrics-for-all-scheduler=True
        2. --log-requests, --enable-metrics, and --collect-tokens-histogram are feature-enabling parameters;
            their validity is verified indirectly through the behavior of other parameters (no dedicated validation required).

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level; --uvicorn-access-log-exclude-prefixes;--enable-metrics;
    --enable-metrics-for-all-scheduler; --bucket-time-to-first-token; --bucket-inter-token-latency;
    --bucket-e2e-request-latency; --collect-tokens-histogram; --prompt-tokens-buckets; --generation-tokens-buckets;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.log_requests_level = 1

        cls.other_args.extend(["--log-requests"])
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])
        cls.other_args.extend(["--uvicorn-access-log-exclude-prefixes", *cls.log_exclude_prefixes])
        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--tp-size", 2])
        cls.other_args.extend(["--enable-metrics-for-all-scheduler"])
        cls.other_args.extend(["--bucket-time-to-first-token", *cls.my_bucket])
        cls.other_args.extend(["--bucket-inter-token-latency", *cls.my_bucket])
        cls.other_args.extend(["--bucket-e2e-request-latency", *cls.my_bucket])
        cls.other_args.extend(["--collect-tokens-histogram"])
        cls.other_args.extend(["--prompt-tokens-buckets", "custom", *cls.my_tokens_bucket])
        cls.other_args.extend(["--generation-tokens-buckets", "custom", *cls.my_tokens_bucket])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_1(self):
        self._verify_inference()

        self._verify_log_requests_level(self.log_requests_level, self.out_log_file)

        # test --uvicorn-access-log-exclude-prefixes
        self._verify_log_exclude_prefixes(True, self.out_log_file)

        self._verify_enable_metrics_for_all_scheduler(True)

        self._verify_metrics_and_bucket_boundary(
            expected_time_to_first_token_bucket=self.my_bucket,
            expected_inter_token_latency_bucket=self.my_bucket,
            expected_e2e_request_latency_bucket=self.my_bucket,
            expected_prompt_tokens_bucket=self.my_tokens_bucket,
            expected_generation_tokens_bucket=self.my_tokens_bucket,
        )


if __name__ == "__main__":
    unittest.main()
