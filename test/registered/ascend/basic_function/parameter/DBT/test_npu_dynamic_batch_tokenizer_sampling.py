import os
import threading
import unittest




import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
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

# Four distinct sampling configurations whose kwargs all differ from each other.
# Requests carrying these configs cannot be grouped into a single tokenizer call.
SAMPLING_CONFIGS = [
    {"temperature": 0.0, "max_new_tokens": 32},
    {"temperature": 0.7, "max_new_tokens": 32},
    {"temperature": 1.0, "max_new_tokens": 32},
    {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 32},
]


class TestDynamicBatchTokenizerSamplingParams(CustomTestCase):
    """Testcase: Verify concurrent requests with different sampling parameters
    all complete successfully when dynamic batch tokenizer is enabled.
    Requests whose kwargs differ are processed individually by the tokenizer
    (not grouped into a shared batch call); all must still return HTTP 200.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size;
                  --dynamic-batch-tokenizer-batch-timeout
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
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_different_sampling_params(self):
        # Send 5 copies of each sampling config concurrently (20 requests total).
        # Because kwargs differ across configs, the tokenizer cannot batch them
        # together and must fall back to individual processing.
        # All 20 requests must return HTTP 200.
        request_payloads = SAMPLING_CONFIGS * 5
        results = []
        lock = threading.Lock()

        def _send(sampling_params):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": sampling_params,
                },
                timeout=60,
            )
            with lock:
                results.append({"status_code": response.status_code})

        threads = [
            threading.Thread(target=_send, args=(p,)) for p in request_payloads
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = len(request_payloads)
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            expected,
            f"Expected {expected} successful requests, "
            f"but only {success_count} succeeded.",
        )


if __name__ == "__main__":
    unittest.main()
