import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestLlama4LoRA(CustomTestCase):
    """
    Testcase：Verify the successful launch and operation of Llama-4 model when LoRA function is enabled.

    [Test Category] Parameter
    [Test Target] --enable-lora, --max-lora-rank 64, --lora-target-modules all
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--lora-target-modules",
                "all",
                "--tp-size",
                8,
                "--context-length",
                "262144",
                "--attention-backend",
                "fa3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_bringup(self):
        self.assertNotEqual(self.process, None)
        self.assertEqual(self.process.poll(), None)


if __name__ == "__main__":
    unittest.main()
