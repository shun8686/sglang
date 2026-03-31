import logging
import os
import unittest



from sglang.srt.utils import kill_process_tree
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

# Token ID for <|endoftext|> in Qwen3 vocabulary, used as the delimiter between
# query and each item in the combined sequence. A special token is chosen here
# because (1) it is guaranteed to be within vocabulary bounds, and (2) it
# serves as an unambiguous separator that does not conflict with regular text tokens.
_DELIMITER_TOKEN_ID = 151643

# Qwen3 vocabulary token IDs corresponding to affirmative ("Yes", token 9454)
# and negative ("No", token 2753) responses. These are used as label_token_ids
# to measure how strongly the model endorses each item as the correct answer.
_LABEL_TOKEN_IDS = [9454, 2753]

# Mandatory server arguments for multi-item scoring mode.
# Constraints from the feature specification:
#   --disable-radix-cache: required; radix cache conflicts with delimiter-based
#                          single-sequence multi-item batching.
#   --chunked-prefill-size -1: required; disables chunked prefill so the full
#                               combined sequence is processed in one pass.
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

# Test data: simple arithmetic correctness query.
# "It is 3" is the only correct answer (1+2=3); "It is 4" and "It is 5" are wrong.
# Using a factual correctness question ensures the model's Yes/No token probabilities
# reflect actual semantic reasoning, making the assertion direction predictable.
_QUERY = "Is this the correct result of 1 plus 2? "
_ITEMS = ["It is 3", "It is 4", "It is 5"]


class TestMultiItemScoringBasic(CustomTestCase):
    """
    Verify multi-item scoring basic functionality with text input and apply_softmax=True.

    A single server is started with --multi-item-scoring-delimiter enabled.
    The test sends a text-format score request and validates:
      - Response structure (status, shape, types).
      - Softmax normalization guarantee (sum to 1.0).
      - Semantic score ordering (correct item scores Yes > No; wrong items score No > Yes).
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
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

    def test_structure_and_semantic_correctness_with_softmax(self):
        """
        Send a text-input scoring request with apply_softmax=True and verify:

        1. HTTP 200 response.
        2. 'scores' field present and contains len(items)=3 sub-lists.
        3. Each sub-list has length == len(label_token_ids)=2.
        4. All values are floats.
        5. Each sub-list sums to 1.0: guaranteed by softmax(x)[i] = exp(x_i)/sum(exp(x_j)).
        6. Semantic correctness validated by Yes/No token probability ordering:
             - "It is 3" (correct): score[0][0] (Yes) > score[0][1] (No).
             - "It is 4" (wrong):   score[1][0] (Yes) < score[1][1] (No).
             - "It is 5" (wrong):   score[2][0] (Yes) < score[2][1] (No).
           This ordering demonstrates that the delimiter-based extraction correctly
           captures per-item prediction quality from a single forward pass.
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


        self.assertIn("scores", result, "Response must contain 'scores' field.")
        scores = result["scores"]
        self.assertEqual(
            len(scores),
            len(_ITEMS),
            f"scores count ({len(scores)}) must equal number of items ({len(_ITEMS)}).",
        )


        for idx, score_list in enumerate(scores):
            self.assertEqual(
                len(score_list),
                len(_LABEL_TOKEN_IDS),
                f"scores[{idx}] length ({len(score_list)}) must equal "
                f"len(label_token_ids) ({len(_LABEL_TOKEN_IDS)}).",
            )
            self.assertTrue(
                all(isinstance(v, float) for v in score_list),
                f"scores[{idx}] contains non-float values: {score_list}",
            )
            # apply_softmax=True: softmax over label_token_ids sums exactly to 1.0
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=5,
                msg=(
                    f"scores[{idx}] should sum to 1.0 with apply_softmax=True "
                    f"(got {sum(score_list):.8f})."
                ),
            )

        logger.info("Scores returned: %s", scores)


        self.assertGreater(
            scores[0][0],
            scores[0][1],
            "Correct item 'It is 3': Yes-token probability (scores[0][0]) "
            "should exceed No-token probability (scores[0][1]).",
        )

        self.assertLess(
            scores[1][0],
            scores[1][1],
            "Wrong item 'It is 4': Yes-token probability (scores[1][0]) "
            "should be less than No-token probability (scores[1][1]).",
        )

        self.assertLess(
            scores[2][0],
            scores[2][1],
            "Wrong item 'It is 5': Yes-token probability (scores[2][0]) "
            "should be less than No-token probability (scores[2][1]).",
        )
        logger.info("Semantic correctness verified: correct item ranked highest for Yes token.")


if __name__ == "__main__":
    unittest.main()