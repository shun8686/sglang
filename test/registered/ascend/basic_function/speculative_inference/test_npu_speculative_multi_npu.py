import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

_ASCEND_BACKEND = "ascend"


_COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    _ASCEND_BACKEND,
    "--quantization",
    "modelslim",
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-attention-mode",
    "decode",
    "--mem-fraction-static",
    "0.7",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

# --speculative-draft-load-format dummy: initializes draft weights with random values.
#   Valid options: auto (default), dummy.
#   dummy is used for profiling; output quality is not guaranteed.
# --speculative-draft-model-revision main: specifies a branch/tag/commit for the
#   draft model. Valid range: any valid git ref string; None means default branch.
_SERVER_ARGS = _COMMON_ARGS + [
    "--tp-size",
    "2",
    "--speculative-draft-load-format",
    "dummy",
    "--speculative-draft-model-revision",
    "main",
]


class TestNpuSpeculativeDraftParams(CustomTestCase):
    """Test --speculative-draft-load-format and --speculative-draft-model-revision
    with 2-card TP.

    [Test Category] Parameter & Multi-NPU
    [Test Target]
        --tp-size 2;
        --speculative-draft-load-format dummy;
        --speculative-draft-model-revision main
    [Model]
        Target: aleoyang/Qwen3-32B-w8a8-MindIE
        Draft: Qwen/Qwen3-32B-Eagle3 (dummy weights, revision=main)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )
        cls.process = popen_launch_server(
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_dummy_format_with_tp2(self):
        """Verify --speculative-draft-load-format dummy: server starts and responds.

        Test steps:
          1. Send an inference request to a server whose draft weights are randomly
             initialized (dummy format) with 2-card tensor parallelism.
          2. Verify HTTP 200 and non-empty response content.
          Note: Output accuracy is not checked because draft weights are random.
        """
        # Step 1: Send inference request
        prompt = "What is the capital of France?"
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=300,
        )

        # Step 2: Verify HTTP 200 and non-empty response
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed with status {response.status_code}: {response.text}",
        )

        result = response.json()
        self.assertIn("choices", result, "Response missing 'choices' field")
        self.assertGreater(len(result["choices"]), 0, "No choices in response")

        content = result["choices"][0]["message"]["content"]
        self.assertGreater(len(content.strip()), 0, "Generated content is empty")

        print(f"Q: {prompt}")
        print(f"A: {content}")

    def test_draft_model_revision_main(self):
        """Verify --speculative-draft-model-revision main: parameter accepted without error.

        Test steps:
          1. Verify server health endpoint returns 200 (server started with revision param).
          2. Send an inference request and verify HTTP 200 and non-empty response.
        """
        # Step 1: Check server health to confirm revision param did not block startup
        health_resp = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(health_resp.status_code, 200)

        # Step 2: Send inference request
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
                "messages": [
                    {"role": "user", "content": "Name one programming language."}
                ],
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=300,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed with status {response.status_code}: {response.text}",
        )
        result = response.json()
        self.assertIn("choices", result, "Response missing 'choices' field")
        content = result["choices"][0]["message"]["content"]
        self.assertGreater(len(content.strip()), 0, "Generated content is empty")

        print(f"[draft model revision=main] response: {content}")


if __name__ == "__main__":
    unittest.main()
