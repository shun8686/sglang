import unittest
from types import SimpleNamespace
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableMultimodalNonMlm(CustomTestCase):
    """Testcase: Verify that when the --enable-multimodal parameter is set, the mmlu accuracy is greater than or equal to that when the parameter is not set.

        [Test Category] Parameter
        [Test Target] --enable-multimodal
        """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    # Fix 1: Define as a class variable (shared by all instances) to pass scores across test methods
    score_with_param = None
    score_without_param = None

    def _launch_server(self, enable_multimodal: bool):
        """Universal server launch method, add --enable-multimodal based on parameters"""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        # Add multimodal parameter as needed
        if enable_multimodal:
            other_args.insert(1, "--enable-multimodal")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        # Automatically register cleanup method, no manual tearDown required
        self.addCleanup(kill_process_tree, process.pid)
        return process

    def _verify_inference(self):
        """Universal inference function verification: health check + basic generation"""
        # Health check
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)
        # Basic generation request verification
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def _run_mmlu_eval(self) -> float:
        """Universal MMLU evaluation execution method, returns evaluation score"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        # Retain basic score lower limit assertion
        self.assertGreaterEqual(metrics["score"], 0.2)
        return metrics["score"]

    def test_01_enable_multimodal(self):
        """Test 1: With --enable-multimodal parameter, execute evaluation and save the score"""
        # Launch server
        self._launch_server(enable_multimodal=True)
        # Verify inference function
        self._verify_inference()
        # Fix 3: Assign class variable via class name for cross-instance sharing
        TestEnableMultimodalNonMlm.score_with_param = self._run_mmlu_eval()

    def test_02_disable_multimodal(self):
        """Test 2: Without --enable-multimodal parameter, execute evaluation and save the score"""
        # Launch server
        self._launch_server(enable_multimodal=False)
        # Verify inference function
        self._verify_inference()
        # Fix 3: Assign class variable via class name for cross-instance sharing
        TestEnableMultimodalNonMlm.score_without_param = self._run_mmlu_eval()

    def test_03_assert_score(self):
        """Test 3: Assert that the score with parameter is ≥ the score without parameter"""
        # Fix 4: Read class variable via class name and verify non-null
        self.assertIsNotNone(TestEnableMultimodalNonMlm.score_with_param, "MMLU score with parameter not obtained")
        self.assertIsNotNone(TestEnableMultimodalNonMlm.score_without_param,
                             "MMLU score without parameter not obtained")
        # Core assertion: Score with --enable-multimodal ≥ Score without the parameter
        self.assertGreaterEqual(
            TestEnableMultimodalNonMlm.score_with_param,
            TestEnableMultimodalNonMlm.score_without_param,
            f"MMLU score with --enable-multimodal ({TestEnableMultimodalNonMlm.score_with_param:.4f}) is less than the score without it ({TestEnableMultimodalNonMlm.score_without_param:.4f})"
        )


if __name__ == "__main__":
    # Optional: Add verbosity=2 to print more detailed test logs
    unittest.main(verbosity=2)
