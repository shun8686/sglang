"""
Test cases 3.1 + 3.2 + 3.3 (merged):
  3.4 -- batch_wait_timeout = 0       s  (no waiting, batch_size stays at 1)
  3.3 -- batch_wait_timeout = 0.001 s  (very short, near-zero waiting)
  3.1 -- batch_wait_timeout = 0.01  s  (10 x default of 0.002 s)
  3.2 -- batch_wait_timeout = 0.1   s  (50 x default, long accumulation window)

Internal behavior (developer notes):
  _dynamic_batch_loop uses asyncio.wait_for to wait up to batch_wait_timeout_s
  for the *next* request to arrive.

  - Zero timeout (0 s):
      The wait exits immediately without waiting for additional requests.
      Debug logs should show batch_size=1 throughout (per developer discussion).
      This is the strictest test of the non-batching fallback path.

  - Near-zero timeout (0.001 s):
      The wait exits almost immediately after the first request is dequeued.
      Under concurrency the effective tokenization batch_size stays close to 1
      (visible in debug logs).  Behavior is similar to timeout=0 described
      in the developer discussion.

  - Long timeout (0.1 s):
      The loop waits 100 ms for more requests to join the batch before flushing,
      yielding larger batch_size values in debug logs and higher tokenization
      throughput at the cost of slightly higher per-request first-token latency.

  - Default (0.002 s per the official parameter table): not covered here;
      covered implicitly by all other test files that omit the arg.

Pattern:
  TestDynamicBatchTokenizerTimeout001 is the concrete base.
  Subclasses override only the `batch_timeout` class attribute.
  setUpClass reads cls.batch_timeout so each subclass starts its own server
  instance with the correct value.  The single test method is inherited
  unchanged -- only the server configuration differs between classes.
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

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

NUM_REQUESTS = 20
NUM_CONCURRENT = 8


class TestDynamicBatchTokenizerTimeout0(CustomTestCase):
    """Testcase: Verify dynamic batch tokenizer with batch_wait_timeout=0.
    At timeout=0 the background loop does not wait for more requests to
    accumulate; debug logs should show batch_size=1 throughout.
    All concurrent requests must still complete correctly.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-timeout
    """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    batch_timeout = 0

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
                "--dynamic-batch-tokenizer-batch-timeout",
                str(cls.batch_timeout),
                "--log-level",
                "debug",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_requests_succeed(self):
        # Verify all NUM_REQUESTS concurrent requests complete with the correct
        # result at the configured batch_timeout.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=NUM_REQUESTS,
            num_concurrent=NUM_CONCURRENT,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            NUM_REQUESTS,
            f"[timeout={self.batch_timeout}] Expected {NUM_REQUESTS} successful "
            f"requests, but only {success_count} succeeded.",
        )
        for result in results:
            self.assertIn(
                "Paris",
                result["text"],
                f"[timeout={self.batch_timeout}] Task {result['task_id']}: "
                f"result does not contain 'Paris'.",
            )


class TestDynamicBatchTokenizerTimeout001(TestDynamicBatchTokenizerTimeout0):
    """Testcase: Verify dynamic batch tokenizer with batch_wait_timeout=0.001 s.
    At near-zero timeout the background loop does not wait for more requests
    to accumulate (but may allow minimal accumulation window); the effective
    tokenization batch_size in debug logs should stay close to 1.
    All concurrent requests must still complete correctly.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-timeout
    """
    batch_timeout = 0.001


class TestDynamicBatchTokenizerTimeout01(TestDynamicBatchTokenizerTimeout0):
    """Testcase: Verify dynamic batch tokenizer with batch_wait_timeout=0.01 s.
    10x larger than the 0.001 s case; under concurrency the loop can accumulate
    several requests before flushing, improving tokenization throughput.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-timeout
    """
    batch_timeout = 0.01

class TestDynamicBatchTokenizerTimeout1(TestDynamicBatchTokenizerTimeout0):
    """Testcase: Verify dynamic batch tokenizer with batch_wait_timeout=0.1 s.
    50x the default (0.002 s).  Long accumulation window maximizes batch_size
    visible in debug logs; accuracy and correctness must be unaffected.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-timeout
    """
    batch_timeout = 0.1

if __name__ == "__main__":
    unittest.main()
