"""
Test cases 1.2 + 1.3 (merged):
  1.2 -- Verify dynamic batch tokenizer handles concurrent requests correctly.
  1.3 -- Verify compatibility with --disable-radix-cache (merged into 1.2).

How the feature works:
  AsyncDynamicbatchTokenizer maintains an async queue.  When concurrent requests
  arrive, the background task _dynamic_batch_loop accumulates them until
  max_batch_size is reached OR batch_wait_timeout_s expires, then calls the
  tokenizer once for the whole batch (if all kwargs are identical).
  --disable-radix-cache disables the prefix KV-cache but must not break the
  dynamic tokenizer pipeline that lives upstream in the TokenizerManager.
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

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)

NUM_REQUESTS = 20
NUM_CONCURRENT = 8


class TestDynamicBatchTokenizerConcurrent(CustomTestCase):
    """Testcase: Verify dynamic batch tokenizer handles concurrent requests
    correctly and is compatible with --disable-radix-cache.

    NUM_REQUESTS concurrent requests are sent with up to NUM_CONCURRENT
    in-flight at once.  The tokenizer batches them and all must return HTTP
    200 with the expected inference result.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --disable-radix-cache
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
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_concurrent_requests_with_radix_cache_disabled(self):
        # All NUM_REQUESTS concurrent requests must succeed and produce a
        # correct result even when the radix cache is disabled.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=NUM_REQUESTS,
            num_concurrent=NUM_CONCURRENT,
        )

        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            NUM_REQUESTS,
            f"Expected {NUM_REQUESTS} successful requests, "
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
