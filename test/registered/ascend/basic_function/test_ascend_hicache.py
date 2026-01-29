import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import Llama_3_1_8B_Instruct_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_is_hip = is_hip()

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestHiCache(CustomTestCase):
    """Testcase：Verify the correctness of --enable-hierarchical-cache (HiCache) and MMLU dataset accuracy meets the requirement (score ≥ 0.65).

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache
    """

    @classmethod
    def setUpClass(cls):
        # Test class initialization: Launch the service with HiCache enabled and related NPU/HIP configurations
        cls.model = Llama_3_1_8B_Instruct_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.7,
                "--hicache-size",
                100 if not _is_hip else 200,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        # Verify MMLU dataset evaluation accuracy meets the minimum requirement (score ≥ 0.65) with HiCache enabled
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
