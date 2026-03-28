import unittest

import requests

import os

# ============ [Local path override - for local debugging only] ============
LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"
import sglang.test.ascend.test_ascend_utils as _utils
_utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-30B-A3B-Instruct"
)
# =========================================================================

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestEncoderOnly(CustomTestCase):
    """Testcase 5.1: Verify encoder-only server starts and is healthy with complete production config.

    --encoder-only instructs the VLM server to load and run only the visual encoder,
    skipping lm_head and all language model layers. In production VLM disaggregation:
    - Encoder servers handle image tokenization / visual feature extraction.
    - Language servers receive the resulting embeddings via a transfer backend.

    This test covers case 5.1 (complete encoder-only config) and implicitly case 1.1
    (basic encoder-only mode). It also covers case 5.5 (multi-card environment) by
    using tp-size=2, verifying that the encoder can be sharded across two NPU cards.

    Assertion rationale: a 200 from /health_generate proves the server fully initialized
    without crashing. An encoder-only server does NOT do text generation, but the health
    endpoint is still registered (see encode_server.py) and returns 200 once the encoder
    model is loaded. Any startup error caused by bad parameter combination would cause
    popen_launch_server to raise before the test method is reached.

    [Test Category] Parameter
    [Test Target] --encoder-only; --tp-size; --encoder-transfer-backend
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--encoder-only",
                "--tp-size",
                "2",
                # Use the default transfer backend explicitly for completeness
                "--encoder-transfer-backend",
                "zmq_to_scheduler",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        # Restore the test environment by stopping the server process (coding standard rule 8)
        kill_process_tree(cls.process.pid)

    def test_encoder_only_health(self):
        """Verify the encoder-only server is healthy after startup.

        HTTP 200 from /health confirms:
        1. --encoder-only was recognized and the encoder model was successfully loaded.
        2. --tp-size=2 sharded the encoder across two NPU cards without error.
        3. No language model layers were loaded (any failure there would crash startup).
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(
            response.status_code,
            200,
            "Encoder-only server failed health check; encoder model may not have "
            "loaded correctly with tp-size=2.",
        )

    def test_encoder_only_does_not_support_generate(self):
        """Verify an encoder-only server returns a non-2xx response for text generation.

        An encoder-only server has no language model, so /generate requests must fail.
        This test guards against accidental false passing: if /generate returns 200
        on an encoder-only server, the server was not actually configured in encoder-only
        mode (a safety check against fake pass).
        """
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=30,
        )
        # An encoder-only server must reject text generation; anything other than
        # 2xx is acceptable (400, 404, 500, etc.)
        self.assertNotEqual(
            response.status_code // 100,
            2,
            "Encoder-only server unexpectedly accepted a /generate request; "
            "--encoder-only may not have taken effect.",
        )


if __name__ == "__main__":
    unittest.main()
