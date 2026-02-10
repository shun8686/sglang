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
    """Testcase: Verify that when the --enable-multimodal parameter is set, mmlu accuracy is within the margin of error compared
    that when the parameter is not set.

        [Test Category] Parameter
        [Test Target] --enable-multimodal
        """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    score_with_param = None
    score_without_param = None

    def launch_server(self, enable_multimodal: bool):
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
        self.addCleanup(kill_process_tree, process.pid)
        return process

    def verify_inference(self):
        """Universal inference function verification"""
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

    def run_mmlu_eval(self) -> float:
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
        self.launch_server(enable_multimodal=True)
        self.verify_inference()
        TestEnableMultimodalNonMlm.score_with_param = self.run_mmlu_eval()

    def test_02_disable_multimodal(self):
        self.launch_server(enable_multimodal=False)
        self.verify_inference()
        TestEnableMultimodalNonMlm.score_without_param = self.run_mmlu_eval()

    def test_03_assert_score(self):
        print("---------------------------------------res---------------------------------------------")
        print(f"MMLU score with parameter: {TestEnableMultimodalNonMlm.score_with_param}")
        print(f"MMLU score without parameter: {TestEnableMultimodalNonMlm.score_without_param}")
        ALLOWED_ERROR = 0.015
        score_diff = TestEnableMultimodalNonMlm.score_with_param - TestEnableMultimodalNonMlm.score_without_param
        abs_score_diff = abs(score_diff)
        self.assertLessEqual(
            abs_score_diff,
            ALLOWED_ERROR,
            f"MMLU score absolute difference ({abs_score_diff}) exceeds allowed error ({ALLOWED_ERROR})"
        )



if __name__ == "__main__":
    unittest.main(verbosity=2)
