import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestRetractDecode(CustomTestCase):
    """Testcase：Verify MMLU dataset accuracy of Llama-3.1-8B-Instruct model with retract decode feature enabled

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT
    """
    @classmethod
    def setUpClass(cls):
        # Enable retract decode feature for test
        os.environ["SGLANG_TEST_RETRACT"] = "1"

        cls.model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
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

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)



if __name__ == "__main__":
    unittest.main()
