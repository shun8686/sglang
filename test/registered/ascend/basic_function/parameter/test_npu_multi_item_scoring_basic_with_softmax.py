import logging
import unittest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# CI Registration
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

# Test Constants
_DELIMITER_TOKEN_ID = 151643  # Qwen3 vocab
_LABEL_TOKEN_IDS = [9454, 2753]  # Yes/No tokens
_QUERY = "Is this the correct result of 1 plus 2? "
_ITEMS = ["It is 3", "It is 4", "It is 5"]


def send_score_request(
        base_url,
        query,
        items,
        label_token_ids,
        apply_softmax=False,
        item_first=False,
        timeout=180,
):
    """Send scoring request to the server."""
    return requests.post(
        url=f"{base_url}/v1/score",
        json={
            "query": query,
            "items": items,
            "label_token_ids": label_token_ids,
            "apply_softmax": apply_softmax,
            "item_first": item_first,
        },
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


class TestMultiItemScoringBasic(CustomTestCase):
    """Test multi-item scoring with various parameters including softmax and item_first."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Launch server with inline args (no separate _SERVER_ARGS list)
        cls.process = popen_launch_server(
            model=QWEN3_32B_WEIGHTS_PATH,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static", "0.8",
                "--attention-backend", "ascend",
                "--disable-cuda-graph",
                "--tp-size", "4",
                "--disable-radix-cache",
                "--chunked-prefill-size", "-1",
                "--multi-item-scoring-delimiter", str(_DELIMITER_TOKEN_ID),
            ],
        )
        logger.info("Server started.")

        # Initialize tokenizer for tokenized tests
        cls.tokenizer = get_tokenizer(QWEN3_32B_WEIGHTS_PATH)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        logger.info("Server terminated.")

    def test_semantic_correctness_with_softmax(self):
        """Verify that Yes/No probabilities reflect correctness of each item."""
        response = send_score_request(
            base_url=self.base_url,
            query=_QUERY,
            items=_ITEMS,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.assertEqual(response.status_code, 200)
        scores = response.json()["scores"]

        # Structure checks
        self.assertEqual(len(scores), len(_ITEMS))
        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(_LABEL_TOKEN_IDS))
            self.assertTrue(all(isinstance(v, float) for v in score_list))

        # Semantic ordering
        self.assertGreater(scores[0][0], scores[0][1], "Correct item: Yes > No")
        self.assertLess(scores[1][0], scores[1][1], "Wrong item 1: Yes < No")
        self.assertLess(scores[2][0], scores[2][1], "Wrong item 2: Yes < No")
        logger.info("Semantic correctness verified.")

    def test_softmax_true_text_input(self):
        """apply_softmax=True => each item's scores sum to 1.0."""
        response = send_score_request(
            base_url=self.base_url,
            query=_QUERY,
            items=_ITEMS,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.assertEqual(response.status_code, 200)
        scores = response.json()["scores"]
        for i, score_list in enumerate(scores):
            self.assertAlmostEqual(sum(score_list), 1.0, places=5,
                                   msg=f"Item {i} scores do not sum to 1.0")
        logger.info("Softmax=True normalization verified.")

    def test_softmax_false_tokenized_input(self):
        """apply_softmax=False with tokenized input => values in [0,1] and differs from softmax=True."""
        query_tokens = self.tokenizer.encode(_QUERY, add_special_tokens=False)
        items_tokens = [self.tokenizer.encode(item, add_special_tokens=False) for item in _ITEMS]

        # Request with apply_softmax=False
        response_false = send_score_request(
            base_url=self.base_url,
            query=query_tokens,
            items=items_tokens,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=False,
        )
        self.assertEqual(response_false.status_code, 200)
        scores_false = response_false.json()["scores"]
        for i, score_list in enumerate(scores_false):
            for j, val in enumerate(score_list):
                self.assertIsInstance(val, float)
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)
        logger.info("scores (apply_softmax=False): %s", scores_false)

        # Request with apply_softmax=True (same tokenized input)
        response_true = send_score_request(
            base_url=self.base_url,
            query=query_tokens,
            items=items_tokens,
            label_token_ids=_LABEL_TOKEN_IDS,
            apply_softmax=True,
        )
        self.assertEqual(response_true.status_code, 200)
        scores_true = response_true.json()["scores"]
        logger.info("scores (apply_softmax=True): %s", scores_true)

        # The two modes must produce different numerical values
        self.assertNotEqual(
            scores_false, scores_true,
            "apply_softmax=True and apply_softmax=False should produce different scores."
        )
        logger.info("Softmax=False vs Softmax=True distinction verified.")


    def test_item_first_flag(self):
        """Verify that item_first=True does not cause server error."""
        response = send_score_request(
            base_url=self.base_url,
            query=_QUERY,
            items=_ITEMS,
            label_token_ids=_LABEL_TOKEN_IDS,
            item_first=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("scores", response.json())
        logger.info("item_first=True accepted by server.")


if __name__ == "__main__":
    unittest.main()