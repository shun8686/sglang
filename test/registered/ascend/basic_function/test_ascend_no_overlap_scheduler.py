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
    """Testcase: Verify the model's ability to successfully process inference requests and ensure inference accuracy ≥ 0.65, under the condition that the overlap scheduler is disabled, 
                 with all combinations of radix cache (enabled/disabled) and chunked prefill (enabled/disabled).

    [Test Category] Parameter
    [Test Target] --disable-radix-cache;--disable-overlap
    """

    def test_no_radix_attention_chunked_prefill(self):
        """Test inference with radix cache disabled + chunked prefill enabled (size=128) & overlap scheduler disabled"""
        chunked_prefill_size = 128
        # Configure MMLU test parameters and evaluation returns accuracy ≥ 0.65
        metrics = run_mmlu_test(
            disable_radix_cache=True,
            chunked_prefill_size=chunked_prefill_size,
            disable_overlap=True,
        )

    def test_no_radix_attention_no_chunked_prefill(self):
         """Test inference with radix cache disabled + chunked prefill disabled & overlap scheduler disabled"""
        # Configure MMLU test parameters and evaluation returns accuracy ≥ 0.65
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        """Test inference with radix cache enabled + chunked prefill enabled (size=128) & overlap scheduler disabled"""
        chunked_prefill_size = 128
        # Configure MMLU test parameters and evaluation returns accuracy ≥ 0.65
        run_mmlu_test(
            disable_radix_cache=False,
            chunked_prefill_size=chunked_prefill_size,
            disable_overlap=True,
        )

    def test_radix_attention_no_chunked_prefill(self):
        """Test inference with radix cache enabled + chunked prefill disabled & overlap scheduler disabled"""
        # Configure MMLU test parameters and evaluation returns accuracy ≥ 0.65
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
