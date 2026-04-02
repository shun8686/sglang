import math
import unittest

import requests

from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_ascend_utils import run_command
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci


register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDecodeLogInterval(TestNPULoggingBase):
    """Testcase: Test configuration --decode-log-interval is set to 10, generating 52 decode batches.

    [Test Category] Parameter
    [Test Target] --decode-log-interval
    """

    decode_numbers = 10

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--decode-log-interval", cls.decode_numbers])
        cls.launch_server()

    def test_decode_log_interval(self):
        max_tokens = 512
        # response = requests.get(f"{self.base_url}/health_generate")
        # self.assertEqual(response.status_code, 200)
        # response = requests.post(
        #     f"{self.base_url}/generate",
        #     json={
        #         "text": "The capital of France is",
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_new_tokens": max_tokens,
        #         },
        #     },
        # )
        # self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        self.inference_once(max_tokens=512)
        result = run_command("cat ./cache_err_log.txt | grep 'Decode batch' | wc -l")
        decod_batch_result = math.floor((max_tokens + 9) / self.decode_numbers)
        self.assertEqual(decod_batch_result, int(result.strip()))


class TestDecodeLogIntervalOther(TestDecodeLogInterval):
    decode_numbers = 30


if __name__ == "__main__":
    unittest.main()
