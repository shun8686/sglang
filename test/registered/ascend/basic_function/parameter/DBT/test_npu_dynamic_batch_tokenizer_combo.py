"""
Test cases 4.1 + 4.2 + 4.3 + 4.4 (merged into one test class):
  One server started with all three dynamic batch tokenizer parameters set.
  The three test methods cover:
    test_high_concurrency    -- 4.1 / 4.2: all params + 100 concurrent requests
    test_mixed_text_lengths  -- 4.3: short / medium / long prompts concurrently
    test_streaming_requests  -- 4.4: SSE streaming mode

Streaming note:
  The "stream" flag belongs at the TOP LEVEL of the JSON request body, NOT
  inside "sampling_params".  The requests library must also be opened with
  stream=True so that iter_lines() works correctly.

Server configuration:
  --dynamic-batch-tokenizer-batch-size 16  (forces ~7 tokenization batch cycles
    for 100 concurrent requests, validating multi-cycle correctness)
  --dynamic-batch-tokenizer-batch-timeout 0.01  (10 ms wait; under 100-request
    concurrency the loop will accumulate requests within the window)
  --disable-radix-cache  (confirms compatibility)
"""
import json
import threading
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


import requests

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

# Prompts of varying lengths used in test_mixed_text_lengths.
SHORT_PROMPTS = ["Hi", "OK", "Yes"]
MEDIUM_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is.",
    "Describe the water cycle briefly.",
]
LONG_PROMPTS = [
    "Describe the history of the Roman Empire and its influence on modern culture " * 3,
    "Explain how large language models are trained, evaluated, and deployed " * 3,
]


class TestDynamicBatchTokenizerCombo(CustomTestCase):
    """Testcase: Verify dynamic batch tokenizer with all parameters combined,
    covering high-concurrency traffic, mixed input lengths, and streaming.
    All three test methods share a single server launched with batch_size=16,
    batch_timeout=0.01, and --disable-radix-cache.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size;
                  --dynamic-batch-tokenizer-batch-timeout; --disable-radix-cache
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
                "--dynamic-batch-tokenizer-batch-size",
                "16",
                "--dynamic-batch-tokenizer-batch-timeout",
                "0.01",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_high_concurrency(self):
        # Send 100 concurrent requests; batch_size=16 means ~7 tokenization
        # cycles are formed.  No request must be lost across cycles.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=100,
            num_concurrent=20,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            100,
            f"Expected 100 successful requests, but only {success_count} succeeded.",
        )

    def test_mixed_text_lengths(self):
        # Concurrent requests with short / medium / long prompts.
        # The tokenizer batches them when all kwargs match even if token counts differ.
        all_prompts = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS
        results = []
        lock = threading.Lock()

        def _send(prompt):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
                timeout=60,
            )
            with lock:
                results.append({"status_code": response.status_code})

        threads = [threading.Thread(target=_send, args=(p,)) for p in all_prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            len(all_prompts),
            f"Expected {len(all_prompts)} successful requests, "
            f"but only {success_count} succeeded.",
        )

    def test_streaming_requests(self):
        # Streaming (SSE) mode must work correctly alongside dynamic batch tokenizer.
        # "stream": True is a TOP-LEVEL field in the request body, not in sampling_params.
        # The requests library must also use stream=True to iterate SSE lines.
        results = []
        lock = threading.Lock()

        def _send_stream(prompt):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )
            collected = []
            for line in response.iter_lines():
                if line and line.startswith(b"data: ") and line[6:] != b"[DONE]":
                    chunk = json.loads(line[6:])
                    if "text" in chunk:
                        collected.append(chunk["text"])
            with lock:
                results.append(
                    {
                        "status_code": response.status_code,
                        "has_content": len(collected) > 0,
                    }
                )

        prompts = [
            "The capital of France is",
            "The largest planet in the solar system is",
            "The speed of light is approximately",
        ]
        threads = [
            threading.Thread(target=_send_stream, args=(p,)) for p in prompts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            self.assertEqual(
                r["status_code"],
                200,
                "Streaming request returned non-200 status code.",
            )
            self.assertTrue(
                r["has_content"],
                "Streaming response returned no SSE content chunks.",
            )


if __name__ == "__main__":
    unittest.main()
