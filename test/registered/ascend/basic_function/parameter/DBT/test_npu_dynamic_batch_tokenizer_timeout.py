import os
import unittest



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
