import random
import time
import unittest
import logging

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu
from sglang.test.ci.ci_register import register_npu_ci

# Configure logging module to replace direct print statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

torch.set_printoptions(threshold=float("inf"))


class TestMatrixMultiplication(unittest.TestCase):
    """Testcase: Tests using the transpose+batch matmul fusion operator showed a 65% optimization for aingle operators.

    [Test Category] Interface
    [Test Target] transpose+batch matmul
    """

    def compute_golden(self, a, b, res1, m, n):
        """Compute reference result (golden)"""
        torch.bmm(a.transpose(0, 1), b, out=res1.view(-1, m, n).transpose(0, 1))

    def assert_tensors_almost_equal(self, actual, expected, dtype):
        """Check if two tensors are approximately equal (considering floating point errors)"""
        self.assertEqual(actual.shape, expected.shape, "Shape mismatch")

        # Check for NaN
        self.assertFalse(torch.isnan(actual).any(), "Actual result contains NaN")
        self.assertFalse(torch.isnan(expected).any(), "Expected result contains NaN")

        # Check for Inf
        self.assertFalse(torch.isinf(actual).any(), "Actual result contains Inf")
        self.assertFalse(torch.isinf(expected).any(), "Expected result contains Inf")

        # Set different tolerances based on data type
        if dtype == torch.float16:
            rtol, atol = 1e-5, 1e-5
        else:  # bfloat16
            rtol, atol = 1.5e-5, 1.5e-5

        # Compare values
        diff = torch.abs(actual - expected)
        max_diff = diff.max().item()
        max_expected = torch.abs(expected).max().item()

        # Check relative and absolute errors
        if max_expected > 0:
            relative_diff = max_diff / max_expected
            self.assertLessEqual(
                relative_diff,
                rtol,
                f"Relative error too large: {relative_diff} > {rtol}. Max difference: {max_diff}",
            )

        self.assertLessEqual(
            max_diff, atol, f"Absolute error too large: {max_diff} > {atol}"
        )

    def test_boundary_conditions(self):
        """Test boundary conditions"""
        test_cases = [
            (1, 1, 1, 1),  # Minimum size
            (1, 10, 1, 1),  # b=1
            (10, 1, 1, 10),  # m=1
            (5, 5, 1, 5),  # k=1
            (2, 2, 2, 1),  # n=1
            (100, 1, 1, 100),  # Flat case
            (1, 100, 100, 1),  # Flat case
            (2, 3, 4, 5),  # Random small size
            (10, 20, 30, 40),  # Medium size
            (36, 128, 512, 128),  # target case
            (8, 160, 512, 128),
        ]

        dtypes = [torch.float16, torch.bfloat16]
        # Performance assertion threshold: fused operator should be at least 60% faster than native operator
        performance_speedup_threshold = 0.6

        # Initialize lists to collect timing results of all test cases
        all_golden_times = []
        all_fused_times = []

        for dtype in dtypes:
            for b, m, k, n in test_cases:
                with self.subTest(dtype=dtype, shape=f"({b}, {m}, {k}, {n})"):
                    a = torch.randn(b, m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(m, k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                    res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                    # Measure time for golden (native) computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    self.compute_golden(a, b_tensor, res1, m, n)
                    torch.npu.synchronize()
                    golden_time = time.time() - start_time

                    # Measure time for fused computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    torch.ops.npu.batch_matmul_transpose(a, b_tensor, res2)
                    torch.npu.synchronize()
                    fused_time = time.time() - start_time

                    # Verify result correctness for current test case
                    self.assert_tensors_almost_equal(res1.view(-1, m, n), res2, dtype)

                    # Collect timing results of current test case
                    all_golden_times.append(golden_time)
                    all_fused_times.append(fused_time)

                    # Log current test case result
                    logger.info(
                        f"Shape: ({b}, {m}, {k}, {n}), dtype: {dtype}, Golden time: {golden_time:.6f}s, Fused time: {fused_time:.6f}s"
                    )

        # Calculate overall average time after all test cases are executed
        if all_golden_times and all_fused_times:
            avg_golden_time = sum(all_golden_times) / len(all_golden_times)
            avg_fused_time = sum(all_fused_times) / len(all_fused_times)

            # Calculate overall speedup ratio and assert (avoid division by zero)
            if avg_golden_time > 1e-9:
                overall_speedup_ratio = (avg_golden_time - avg_fused_time) / avg_golden_time
                logger.info(
                    f"\n===== Test Boundary Conditions Overall Result ====="
                    f"\nAverage Golden time: {avg_golden_time:.6f}s"
                    f"\nAverage Fused time: {avg_fused_time:.6f}s"
                    f"\nOverall Speedup Ratio: {overall_speedup_ratio:.4f}"
                )

                # Final performance assertion (execute only once)
                self.assertGreaterEqual(
                    overall_speedup_ratio,
                    performance_speedup_threshold,
                    f"Overall performance optimization not meet requirement! Overall speedup ratio: {overall_speedup_ratio:.4f} < {performance_speedup_threshold}"
                )
            else:
                logger.warning("Golden average time is too small to calculate valid speedup ratio")
        else:
            logger.warning("No valid timing results collected for boundary conditions test")

    def test_random_shapes(self):
        """Test randomly generated shapes"""
        num_tests = 1
        dtypes = [torch.float16, torch.bfloat16]
        # Performance assertion threshold: fused operator should be at least 60% faster than native operator
        performance_speedup_threshold = 0.6

        # Initialize lists to collect timing results of all test cases
        all_golden_times = []
        all_fused_times = []

        for dtype in dtypes:
            for _ in range(num_tests):
                # Generate reasonable random sizes
                b = random.randint(1, 500)
                m = random.randint(1, 500)
                k = random.randint(1, 500)
                n = random.randint(1, 500)

                with self.subTest(dtype=dtype, shape=f"Random ({b}, {m}, {k}, {n})"):
                    a = torch.randn(b, m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(m, k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                    res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                    # Measure time for golden (native) computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    self.compute_golden(a, b_tensor, res1, m, n)
                    torch.npu.synchronize()
                    golden_time = time.time() - start_time

                    # Measure time for fused computation
                    torch.npu.synchronize()
                    start_time = time.time()
                    torch.ops.npu.batch_matmul_transpose(a, b_tensor, res2)
                    torch.npu.synchronize()
                    fused_time = time.time() - start_time

                    # Verify result correctness for current test case
                    self.assert_tensors_almost_equal(res1.view(-1, m, n), res2, dtype)

                    # Collect timing results of current test case
                    all_golden_times.append(golden_time)
                    all_fused_times.append(fused_time)

                    # Log current test case result
                    logger.info(
                        f"Shape: Random ({b}, {m}, {k}, {n}), dtype: {dtype}, Golden time: {golden_time:.6f}s, Fused time: {fused_time:.6f}s"
                    )

        # Calculate overall average time after all test cases are executed
        if all_golden_times and all_fused_times:
            avg_golden_time = sum(all_golden_times) / len(all_golden_times)
            avg_fused_time = sum(all_fused_times) / len(all_fused_times)

            # Calculate overall speedup ratio and assert (avoid division by zero)
            if avg_golden_time > 1e-9:
                overall_speedup_ratio = (avg_golden_time - avg_fused_time) / avg_golden_time
                logger.info(
                    f"\n===== Test Random Shapes Overall Result ====="
                    f"\nAverage Golden time: {avg_golden_time:.6f}s"
                    f"\nAverage Fused time: {avg_fused_time:.6f}s"
                    f"\nOverall Speedup Ratio: {overall_speedup_ratio:.4f}"
                )

                # Final performance assertion (execute only once)
                self.assertGreaterEqual(
                    overall_speedup_ratio,
                    performance_speedup_threshold,
                    f"Overall performance optimization not meet requirement! Overall speedup ratio: {overall_speedup_ratio:.4f} < {performance_speedup_threshold}"
                )
            else:
                logger.warning("Golden average time is too small to calculate valid speedup ratio")
        else:
            logger.warning("No valid timing results collected for random shapes test")

    def test_zero_values(self):
        """Test zero input values"""
        dtypes = [torch.float16, torch.bfloat16]
        b, m, k, n = 5, 4, 3, 2
        # Performance assertion threshold: fused operator should be at least 60% faster than native operator
        performance_speedup_threshold = 0.6

        # Initialize lists to collect timing results of all test cases
        all_golden_times = []
        all_fused_times = []

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a = torch.zeros(b, m, k, dtype=dtype, device="npu")
                b_tensor = torch.zeros(m, k, n, dtype=dtype, device="npu")
                res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                # Measure time for golden (native) computation
                torch.npu.synchronize()
                start_time = time.time()
                self.compute_golden(a, b_tensor, res1, m, n)
                torch.npu.synchronize()
                golden_time = time.time() - start_time

                # Measure time for fused computation
                torch.npu.synchronize()
                start_time = time.time()
                torch.ops.npu.batch_matmul_transpose(a, b_tensor, res2)
                torch.npu.synchronize()
                fused_time = time.time() - start_time

                # Verify result correctness for current test case
                self.assert_tensors_almost_equal(res1.view(-1, m, n), res2, dtype)
                self.assertTrue(torch.all(res2 == 0))

                # Collect timing results of current test case
                all_golden_times.append(golden_time)
                all_fused_times.append(fused_time)

                # Log current test case result
                logger.info(
                    f"Shape: ({b}, {m}, {k}, {n}), dtype: {dtype}, Golden time: {golden_time:.6f}s, Fused time: {fused_time:.6f}s"
                )

        # Calculate overall average time after all test cases are executed
        if all_golden_times and all_fused_times:
            avg_golden_time = sum(all_golden_times) / len(all_golden_times)
            avg_fused_time = sum(all_fused_times) / len(all_fused_times)

            # Calculate overall speedup ratio and assert (avoid division by zero)
            if avg_golden_time > 1e-9:
                overall_speedup_ratio = (avg_golden_time - avg_fused_time) / avg_golden_time
                logger.info(
                    f"\n===== Test Zero Values Overall Result ====="
                    f"\nAverage Golden time: {avg_golden_time:.6f}s"
                    f"\nAverage Fused time: {avg_fused_time:.6f}s"
                    f"\nOverall Speedup Ratio: {overall_speedup_ratio:.4f}"
                )

                # Final performance assertion (execute only once)
                self.assertGreaterEqual(
                    overall_speedup_ratio,
                    performance_speedup_threshold,
                    f"Overall performance optimization not meet requirement! Overall speedup ratio: {overall_speedup_ratio:.4f} < {performance_speedup_threshold}"
                )
            else:
                logger.warning("Golden average time is too small to calculate valid speedup ratio")
        else:
            logger.warning("No valid timing results collected for zero values test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
