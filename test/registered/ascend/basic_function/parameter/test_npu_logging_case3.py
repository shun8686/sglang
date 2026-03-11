import unittest
import requests

from test_npu_logging import TestNPULoggingBase

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULoggingCase3(TestNPULoggingBase):
    """Test case class for validating specific scenarios of logging feature parameters (complementary to Case0/1/2).

    Core Functionality:
        This class tests the behavior of SGLang logging/metrics features under a specific parameter configuration set,
        working with Case0/1/2 to fully cover the parameter value range of the logging feature. The tested configuration includes:
          --log-requests; --log-requests-level=3; --enable-metrics;
          --tokenizer-metrics-custom-labels-header custom_labels_header; --tokenizer-metrics-allowed-custom-labels allowed_custom_labels_list;

        Note:
            --log-requests, --enable-metrics, and --collect-tokens-histogram are feature-enabling parameters;
            their validity is verified indirectly through the behavior of other parameters (no dedicated validation required).

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level; --enable-metrics;
    --tokenizer-metrics-custom-labels-header; --tokenizer-metrics-allowed-custom-labels;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.log_requests_level = 3

        cls.other_args.extend(["--log-requests"])
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
        self._verify_inference()

        self._verify_log_requests_level(self.log_requests_level, self.out_log_file)

        # test --tokenizer-metrics-custom-labels-header、--tokenizer-metrics-allowed-custom-labels
        self._verify_log_metrics_tokenizer_label()


if __name__ == "__main__":
    unittest.main()
