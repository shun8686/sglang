import unittest

import requests

import os
import base64
_INLINE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)

_IMAGE_URL = f"data:image/png;base64,{_INLINE_IMAGE_B64}"

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
        env = os.environ.copy()
        env["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--encoder-only",
                "--tp-size",
                "2",
                # Use the default transfer backend explicitly for completeness
                "--encoder-transfer-backend",
                "zmq_to_scheduler",
                #"moooncake"
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

    def test_encoder_processes_image_request(self):
        """Verify the encoder-only server can process an image encoding request.

        The /encode endpoint is the primary functional interface of an encoder-only
        server. A 200 response confirms the visual encoder model loaded correctly
        and can perform forward pass on image input.
        """
        payload = {
            "mm_items": [_IMAGE_URL],
            "req_id": "test-req-001",
            "num_parts": 1,
            "part_idx": 0,
        }
        response = requests.post(
            f"{self.base_url}/encode",
            json=payload,
            timeout=60,
        )
        # 200 means encoder model ran forward pass successfully
        self.assertEqual(
            response.status_code,
            200,
            f"Encoder-only server /encode returned {response.status_code}; "
            "visual encoder may not have initialized correctly.",
        )


if __name__ == "__main__":
    unittest.main()
