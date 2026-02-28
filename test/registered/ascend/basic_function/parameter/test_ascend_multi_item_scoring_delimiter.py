import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Initialize logging configuration (replace print)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)

# Common configuration extraction
COMMON_CONFIG = {
    "model": QWEN3_32B_WEIGHTS_PATH,
    "base_args": [
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
    ],
    "request_timeout": 120,
}

# Common score request data (reused in both test classes)
COMMON_SCORE_REQUEST = {
    "query": "Is this the correct result of 1 plus 2? ",
    "items": ["It is 3", "It is 4", "It is 5"],
    "label_token_ids": [9454, 2753],
    "apply_softmax": True,
    "item_first": False,
}


class TestScoreWithDelimiter(CustomTestCase):
    """Testcase: Verify score logic when --multi-item-scoring-delimiter is enabled.

    [Test Category] Parameter
    [Test Target] --multi-item-scoring-delimiter
    """

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Class-level initialization: Start server with delimiter enabled."""
        logger.info("\n=== Initializing test environment [delimiter enabled] ===")
        # Construct complete server startup arguments
        server_args = COMMON_CONFIG["base_args"] + [
            "--multi-item-scoring-delimiter",
            "151643",
        ]
        # Trust popen_launch_server's readiness logic, no sleep needed
        cls.server_process = popen_launch_server(
            model=COMMON_CONFIG["model"],
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )
        logger.info(
            f"✅ Server started successfully (with delimiter), args: {server_args}"
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.server_process.pid)

    def test_score_logic_with_delimiter(self):
        """Verify score comparison logic when --multi-item-scoring-delimiter is enabled."""
        logger.info("\n=== Testing: --multi-item-scoring-delimiter enabled ===")
        # Call score API
        response = requests.post(
            url=f"{DEFAULT_URL_FOR_TEST}/v1/score",
            json=COMMON_SCORE_REQUEST,
            headers={"Content-Type": "application/json"},
            timeout=COMMON_CONFIG["request_timeout"],
        )
        # Verify response status code
        self.assertEqual(
            response.status_code,
            200,
            f"❌ API returned wrong status code: expected 200, actual {response.status_code}",
        )

        # Parse result and verify score logic
        result = response.json()
        scores = result["scores"]
        logger.info(f"📝 API returned scores: {scores}")

        # Core logic assertions
        self.assertTrue(
            scores[0][0] > scores[0][1],
            "❌ Score logic error for correct item (It is 3): score[0] should be greater than score[1]",
        )
        self.assertTrue(
            scores[1][0] < scores[1][1],
            "❌ Score logic error for wrong item (It is 4): score[0] should be less than score[1]",
        )
        self.assertTrue(
            scores[2][0] < scores[2][1],
            "❌ Score logic error for wrong item (It is 5): score[0] should be less than score[1]",
        )
        logger.info("✅ Delimiter enabled: Score logic verification passed!")


class TestScoreWithoutDelimiter(CustomTestCase):
    """Testcase: Verify score logic when --multi-item-scoring-delimiter is disabled.

    [Test Category] Parameter
    [Test Target] --multi-item-scoring-delimiter
    """

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Class-level initialization: Start server with delimiter disabled."""
        logger.info("\n=== Initializing test environment [delimiter disabled] ===")
        # Use base args only (no delimiter parameter)
        server_args = COMMON_CONFIG["base_args"].copy()
        cls.server_process = popen_launch_server(
            model=COMMON_CONFIG["model"],
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )
        logger.info(
            f"✅ Server started successfully (without delimiter), args: {server_args}"
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.server_process.pid)

    def test_score_logic_without_delimiter(self):
        logger.info("\n=== Testing: --multi-item-scoring-delimiter disabled ===")
        # Call score API (same request data as delimiter enabled)
        response = requests.post(
            url=f"{DEFAULT_URL_FOR_TEST}/v1/score",
            json=COMMON_SCORE_REQUEST,
            headers={"Content-Type": "application/json"},
            timeout=COMMON_CONFIG["request_timeout"],
        )
        self.assertEqual(
            response.status_code,
            200,
            f"❌ API returned wrong status code: expected 200, actual {response.status_code}",
        )

        # Parse result and verify score logic
        result = response.json()
        scores = result["scores"]
        logger.info(f"📝 API returned scores: {scores}")

        self.assertTrue(
            scores[0][0] > scores[0][1],
            "❌ Score logic error for correct item (It is 3): score[0] should be greater than score[1]",
        )
        self.assertTrue(
            scores[1][0] > scores[1][1],
            "❌ Score logic error for wrong item (It is 4): score[0] should be greater than score[1]",
        )
        self.assertTrue(
            scores[2][0] > scores[2][1],
            "❌ Score logic error for wrong item (It is 5): score[0] should be greater than score[1]",
        )
        logger.info("✅ Delimiter disabled: Score logic verification passed!")


if __name__ == "__main__":
    # Run both independent test classes with detailed output
    unittest.main(verbosity=2)
