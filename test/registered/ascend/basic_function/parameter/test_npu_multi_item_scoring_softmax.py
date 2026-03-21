"""
Testcase: Verify apply_softmax parameter behavior in multi-item scoring mode.

This file covers:
- Case 4 (Apply Softmax): contrast apply_softmax=True and apply_softmax=False behavior.
- Case 6 (Tokenized input, merged): the apply_softmax=False test uses pre-tokenized
  query and items (List[int] / List[List[int]]) to verify that the server correctly
  skips re-tokenization and produces the same scoring logic.

Background on apply_softmax:
  apply_softmax=True:
    scores[i][j] = softmax(logprobs)[j]
    => sum(scores[i]) == 1.0 exactly, values in (0, 1).
  apply_softmax=False:
    scores[i][j] = exp(logprob[j]) if logprob is finite, else 0.0
    => logprob <= 0 for any token, so exp(logprob) in (0, 1].
    => sum(scores[i]) is NOT guaranteed to be 1.0.
  The two modes produce different numerical values for the same input;
  this difference is the key behavioral distinction being tested.

[Test Category] Parameter
[Test Target] apply_softmax; --multi-item-scoring-delimiter
[Model] Qwen/Qwen3-32B
"""

import logging
import os
import unittest

LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"

import sglang.test.ascend.test_ascend_utils as utils
utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
utils.QWEN3_32B_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B"

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH, send_score_request
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

# Token ID for <|endoftext|> in Qwen3 vocabulary used as multi-item delimiter.
_DELIMITER_TOKEN_ID = 151643

# Qwen3 token IDs for Yes (9454) and No (2753).
# Two label tokens are chosen deliberately so that softmax produces a non-trivial
# distribution (neither token collapses to probability 1.0), making the
# softmax=True vs softmax=False comparison meaningful.
_LABEL_TOKEN_IDS = [9454, 2753]

# Mandatory server arguments for multi-item scoring mode.
_SERVER_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--tp-size",
    "4",
    "--disable-radix-cache",
    "--chunked-prefill-size",
    "-1",
    "--multi-item-scoring-delimiter",
    str(_DELIMITER_TOKEN_ID),
]

# Test data shared across both test methods.
_QUERY = "Is this the correct result of 1 plus 2? "
_ITEMS = ["It is 3", "It is 4", "It is 5"]


class TestMultiItemScoringSoftmax(CustomTestCase):
    """
    Verify apply_softmax=True and apply_softmax=False produce correct and distinct outputs
    in multi-item scoring mode, using both text and pre-tokenized input formats.

    A single server is started with --multi-item-scoring-delimiter. Two test methods share
    the server: one sends text input with softmax=True, the other sends tokenized input
    with softmax=False and additionally cross-checks that the two modes differ.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Load tokenizer for pre-tokenized input (Case 6 merged).
        # Only tokenizer metadata is loaded here (JSON files), not model weights,
        # so this is fast and does not conflict with server startup.
        cls.tokenizer = get_tokenizer(QWEN3_32B_WEIGHTS_PATH)
        logger.info("Tokenizer loaded from %s.", QWEN3_32B_WEIGHTS_PATH)

        cls.process = popen_launch_server(
            model=QWEN3_32B_WEIGHTS_PATH,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_SERVER_ARGS,
        )
        logger.info("Server started with multi-item scoring delimiter enabled.")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        logger.info("Server process terminated.")

    def test_apply_softmax_true_text_input(self):
        """
        Verify apply_softmax=True with text query and items:

        For each item, the server computes softmax over the logprobs of label_token_ids.
        By definition: softmax(x)[i] = exp(x_i) / sum_j(exp(x_j)), so sum == 1.0.
        This property must hold regardless of the absolute logprob magnitudes.

        Assertions:
          - HTTP 200.
          - len(scores) == len(items).
          - For each item: sum(score_list) ~= 1.0 (tolerance 1e-5 for float precision).
        """
        response = send_score_request(
            base_url=self.base_url,
            query=_QUERY,
            items=_ITEMS,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected HTTP 200, got {response.status_code}: {response.text}",
        )
        result = response.json()
        self.assertIn("scores", result)
        scores = result["scores"]

        self.assertEqual(
            len(scores),
            len(_ITEMS),
            f"Expected {len(_ITEMS)} score lists, got {len(scores)}.",
        )
        for idx, score_list in enumerate(scores):
            total = sum(score_list)
            self.assertAlmostEqual(
                total,
                1.0,
                places=5,
                msg=(
                    f"scores[{idx}] must sum to 1.0 with apply_softmax=True "
                    f"(actual sum: {total:.8f})."
                ),
            )
        logger.info("apply_softmax=True verified: all item score lists sum to 1.0.")

    def test_apply_softmax_false_tokenized_input(self):
        """
        Verify apply_softmax=False with pre-tokenized query and items (Case 6 merged):

        Pre-tokenized input bypasses the server's internal tokenization step.
        The score formula is: score = exp(logprob) if finite, else 0.0.
        Since logprob <= 0 for any token in a valid distribution, exp(logprob) in (0, 1].

        This test also cross-checks that apply_softmax=True and apply_softmax=False
        produce different numerical values for the same input, confirming the parameter
        switches the normalization path in _convert_logprobs_to_scores.

        Assertions:
          - HTTP 200 for both requests.
          - Each score value: 0.0 <= score <= 1.0.
          - scores from softmax=False != scores from softmax=True (modes differ).
        """
        # Build pre-tokenized input (Case 6: tokenized input format).
        # add_special_tokens=False avoids prepending BOS tokens that would alter
        # the token sequence compared to what the server constructs internally.
        query_tokens = self.tokenizer.encode(_QUERY, add_special_tokens=False)
        items_tokens = [
            self.tokenizer.encode(item, add_special_tokens=False) for item in _ITEMS
        ]
        logger.info(
            "Tokenized query length: %d, item lengths: %s",
            len(query_tokens),
            [len(t) for t in items_tokens],
        )

        # Request with apply_softmax=False
        response_false = send_score_request(
            base_url=self.base_url,
            query=query_tokens,
            items=items_tokens,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=False,
        )
        self.assertEqual(
            response_false.status_code,
            200,
            f"apply_softmax=False: Expected HTTP 200, got {response_false.status_code}.",
        )
        scores_false = response_false.json()["scores"]

        # Verify each raw probability value is within [0, 1].
        # exp(logprob) with logprob <= 0 => value in (0, 1].
        # Value of exactly 0.0 is possible only when the token is absent from the
        # server's top-k logprob results (mapped to -inf, then exp(-inf) = 0.0).
        for idx, score_list in enumerate(scores_false):
            for j, score in enumerate(score_list):
                self.assertIsInstance(
                    score, float, f"scores_false[{idx}][{j}] must be a float."
                )
                self.assertGreaterEqual(
                    score,
                    0.0,
                    f"scores_false[{idx}][{j}] must be >= 0.0 (exp of logprob).",
                )
                self.assertLessEqual(
                    score,
                    1.0,
                    f"scores_false[{idx}][{j}] must be <= 1.0 (exp(logprob) <= exp(0)=1).",
                )

        logger.info("scores (apply_softmax=False): %s", scores_false)

        # Cross-check: the same input with apply_softmax=True must yield different values.
        # If both modes returned identical values, one of the two code paths would be broken.
        response_true = send_score_request(
            base_url=self.base_url,
            query=query_tokens,
            items=items_tokens,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.assertEqual(
            response_true.status_code,
            200,
            f"apply_softmax=True: Expected HTTP 200, got {response_true.status_code}.",
        )
        scores_true = response_true.json()["scores"]

        self.assertNotEqual(
            scores_false,
            scores_true,
            "apply_softmax=True and apply_softmax=False should produce different score "
            "values for the same input (softmax rescales the distribution).",
        )
        logger.info("scores (apply_softmax=True): %s", scores_true)
        logger.info("apply_softmax=False vs apply_softmax=True distinction verified.")


if __name__ == "__main__":
    unittest.main()
