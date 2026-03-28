import unittest

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULoggingCase2(TestNPULoggingBase):
    """Test case class for validating specific scenarios of logging feature parameters (complementary to Case0/1/3).

    Core Functionality:
        This class tests the behavior of SGLang logging/metrics features under a specific parameter configuration set,
        working with Case0/1/3 to fully cover the parameter value range of the logging feature. The tested configuration includes:
          --log-requests; --log-requests-level=2; --enable-metrics; --collect-tokens-histogram;
          --prompt-tokens-buckets custom_bucket_boundary; --generation-tokens-buckets custom_bucket_boundary;

        Note:
            --log-requests, --enable-metrics, and --collect-tokens-histogram are feature-enabling parameters;
            their validity is verified indirectly through the behavior of other parameters (no dedicated validation required).

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level; --enable-metrics;  --collect-tokens-histogram;
    --prompt-tokens-buckets; --generation-tokens-buckets;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.log_requests_level = 2

        cls.other_args.extend(["--log-requests"])
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])
        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--collect-tokens-histogram"])
        cls.other_args.extend(["--prompt-tokens-buckets", "tse", *cls.my_tse_set])
        cls.other_args.extend(["--generation-tokens-buckets", "tse", *cls.my_tse_set])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_2(self):
        self._verify_inference()

        self._verify_log_requests_level(self.log_requests_level, self.out_log_file)

        self._verify_metrics_and_bucket_boundary(
            expected_prompt_tokens_bucket=self.my_tse_bucket,
            expected_generation_tokens_bucket=self.my_tse_bucket,
        )


if __name__ == "__main__":
    unittest.main()
