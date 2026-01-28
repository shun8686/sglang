"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import unittest

from sglang.test.test_utils import CustomTestCase, run_mmlu_test
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestOverlapSchedule(CustomTestCase):
    """Testcase：Verify successful processing of inference requests when overlap scheduler is disabled with different radix cache and chunked prefill combinations

    [Test Category] Parameter
    [Test Target] --disable-radix-cache;--disable-overlap
    """

    def test_no_radix_attention_chunked_prefill(self):
        chunked_prefill_size = 128
        run_mmlu_test(
            disable_radix_cache=True,
            chunked_prefill_size=chunked_prefill_size,
            disable_overlap=True,
        )

    def test_no_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        chunked_prefill_size = 128
        run_mmlu_test(
            disable_radix_cache=False,
            chunked_prefill_size=chunked_prefill_size,
            disable_overlap=True,
        )

    def test_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
