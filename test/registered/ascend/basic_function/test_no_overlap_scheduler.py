"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import unittest

from sglang.srt.utils import is_npu
from sglang.test.test_utils import CustomTestCase, run_mmlu_test
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestOverlapSchedule(CustomTestCase):
    """Test class for overlap schedule with radix attention and chunked prefill.

    Tests MMLU accuracy with different combinations:
    - no-radix+chunked-prefill: Disable radix cache, enable chunked prefill
    - no-radix+no-chunked-prefill: Disable radix cache and chunked prefill
    - radix+chunked-prefill: Enable radix cache, enable chunked prefill
    - radix+no-chunked-prefill: Enable radix cache, disable chunked prefill
    """

    def test_no_radix_attention_chunked_prefill(self):
        chunked_prefill_size = 128 if is_npu() else 32
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
        chunked_prefill_size = 128 if is_npu() else 32
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
