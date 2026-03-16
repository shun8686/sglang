
"""
Verify NEXTN speculative decoding basic functionality on a single Ascend NPU card.

NEXTN (also referred to as MTP in SGLang) uses an independent smaller draft model
from the same model family to predict the next N tokens. Unlike EAGLE3, no
specially trained draft head is required; any smaller model that shares the same
tokenizer and vocabulary as the target model is compatible.

Qwen3-0.6B (draft) and Qwen3-30B-A3B-W8A8 (target) share the Qwen3 tokenizer,
satisfying the vocabulary alignment requirement for NEXTN speculation.

Server configuration follows the reference performance test configuration:
  target model : Qwen/Qwen3-30B-A3B-w8a8  (W8A8 MoE, active 3B params)
  draft model  : Qwen/Qwen3-0.6B           (full precision)
  tp-size      : 1  (single NPU card)

[Test Category] Parameter
[Test Target] --speculative-algorithm; --speculative-draft-model-path;
              --speculative-num-steps; --speculative-eagle-topk;
              --speculative-num-draft-tokens; --speculative-attention-mode;
              --attention-backend
[Model] Qwen/Qwen3-30B-A3B-w8a8; Qwen/Qwen3-0.6B
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
    assert_spec_decoding_active,
    send_inference_request,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_ASCEND_BACKEND = "ascend"

_SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend", _ASCEND_BACKEND,
    "--quantization", "modelslim",
    "--disable-radix-cache",
    # --speculative-algorithm NEXTN: use an independent smaller draft LLM.
    "--speculative-algorithm", "NEXTN",
    "--speculative-draft-model-path", QWEN3_0_6B_WEIGHTS_PATH,
    # --speculative-num-steps 4: draft model runs 4 auto-regressive steps per iteration.
    "--speculative-num-steps", "4",
    # --speculative-eagle-topk 2: retain 2 candidate paths per step.
    "--speculative-eagle-topk", "2",
    # --speculative-num-draft-tokens 7: maximum draft tokens submitted for verification.
    "--speculative-num-draft-tokens", "7",
    # --speculative-attention-mode decode: draft attention in single-token decode mode.
    "--speculative-attention-mode", "decode",
    "--tp-size", "1",
    "--mem-fraction-static", "0.7",
    "--disable-cuda-graph",
    "--dtype", "bfloat16",
]


class TestNpuNextnBasic(CustomTestCase):
    """
    [Test Category] Parameter
    [Test Target] --speculative-algorithm; --speculative-draft-model-path;
                  --speculative-num-steps; --speculative-eagle-topk;
                  --speculative-num-draft-tokens; --speculative-attention-mode;
                  --attention-backend
    [Model] Qwen/Qwen3-30B-A3B-w8a8; Qwen/Qwen3-0.6B
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=os.environ.copy(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_nextn_basic_inference(self):
        """
        Test steps:
          1. Send a single inference request to the NEXTN-enabled server.
          2. Assert the response payload is structurally valid.
          3. Assert avg_spec_accept_length > 1.0 (multi-token acceptance confirmed).
        """
        response = send_inference_request(
            self.base_url, QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
            "List 3 programming languages and their primary use cases.",
        )

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertGreater(
            len(response["choices"][0]["message"]["content"].strip()), 0
        )

        # avg_spec_accept_length > 1.0: Qwen3-0.6B draft tokens accepted by
        # Qwen3-30B-A3B beyond the trivial single-token case, validating the pipeline.
        assert_spec_decoding_active(self, self.base_url, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
