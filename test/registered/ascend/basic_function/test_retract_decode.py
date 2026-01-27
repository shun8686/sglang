import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestRetractDecode(CustomTestCase):
    """Test class for Llama-3.1-8B-Instruct with retract decode enabled.

    Tests MMLU dataset accuracy with retract decode feature:
    - mmlu: MMLU dataset accuracy verification (score ≥ 0.65)
    """
    @classmethod
    def setUpClass(cls):
        # Enable retract decode feature for test
        os.environ["SGLANG_TEST_RETRACT"] = "1"

        cls.model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
            if is_npu()
            else DEFAULT_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.8,
            ]
            if is_npu()
            else []
        )
        # Launch model server with retract decode enabled
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up model server process after test
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """Test MMLU dataset accuracy with retract decode enabled."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        # Run MMLU evaluation and verify accuracy threshold
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


class TestRetractDecodeChunkCache(CustomTestCase):
    """Test class for Llama-3.1-8B-Instruct with retract decode + chunk cache.

    Tests MMLU dataset accuracy with retract decode + chunked prefill:
    - chunk-cache: Retract decode with disabled radix cache + chunked prefill (size=128)
    """
    @classmethod
    def setUpClass(cls):
        # Enable retract decode feature for test
        os.environ["SGLANG_TEST_RETRACT"] = "1"

        # Set model path (NPU uses ModelScope path, others use default test model)
        cls.model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
            if is_npu()
            else DEFAULT_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Configure server arguments with chunked prefill (disable radix cache)
        other_args = (
            [
                "--disable-radix-cache",
                "--chunked-prefill-size",
                128,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.8,
            ]
            if is_npu()
            else ["--disable-radix-cache", "--chunked-prefill-size", 128]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )


if __name__ == "__main__":
    unittest.main()
