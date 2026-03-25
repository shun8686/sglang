"""
Test case 4.6: Verify dynamic batch tokenizer works correctly under
Tensor Parallelism (--tp-size 2).

The AsyncDynamicbatchTokenizer lives inside TokenizerManager, which runs on
the front-end (scheduler) process.  TP worker processes handle the forward
pass independently; the tokenizer layer is TP-transparent.  Enabling TP
should therefore not affect dynamic batch tokenizer behavior.

Requires: 2 NPU cards (suite nightly-2-npu-a3).
"""
import unittest

import os

# ============ [Local path override - for local debugging only] ============
LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"
import sglang.test.ascend.test_ascend_utils as _utils
_utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B-Instruct"
)
# =========================================================================


from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    send_concurrent_requests,
    verify_process_terminated,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-2-npu-a3", nightly=True)

NUM_REQUESTS = 20
NUM_CONCURRENT = 4


class TestDynamicBatchTokenizerTP(CustomTestCase):
    """Testcase: Verify dynamic batch tokenizer works correctly when
    tensor parallelism (--tp-size 2) is enabled.  Requires 2 NPU cards.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --tp-size
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--tp-size",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_concurrent_requests_with_tp(self):
        # TokenizerManager (and AsyncDynamicbatchTokenizer) runs on the front-end
        # process and is unaware of the TP split.  Verify all requests succeed.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=NUM_REQUESTS,
            num_concurrent=NUM_CONCURRENT,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            NUM_REQUESTS,
            f"Expected {NUM_REQUESTS} successful requests under tp-size=2, "
            f"but only {success_count} succeeded.",
        )
        for result in results:
            self.assertIn(
                "Paris",
                result["text"],
                f"Task {result['task_id']}: inference result does not contain 'Paris'.",
            )


if __name__ == "__main__":
    unittest.main()
