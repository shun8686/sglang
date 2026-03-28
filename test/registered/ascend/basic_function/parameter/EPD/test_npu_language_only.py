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


class TestLanguageOnly(CustomTestCase):
    """Testcase 5.2: Verify language-only server starts and handles text generation correctly.

    --language-only instructs the VLM server to load only the language model weights,
    skipping the visual encoder. In production VLM disaggregation:
    - Language servers receive pre-computed visual embeddings from encoder servers.
    - For text-only prompts, the language server runs inference independently.

    This test covers case 5.2 (complete language-only config) and case 2.1 (basic mode).
    It also covers case 5.5 (multi-card environment) via tp-size=2.

    Why text generation is a stronger assertion than health-check alone:
    A language-only server DOES support text generation (/generate endpoint). Verifying
    that a sensible answer is returned end-to-end proves:
    1. The language model weights were loaded correctly.
    2. The forward pass ran on the NPU without error.
    3. The --language-only flag did not corrupt language model initialization.

    [Test Category] Parameter
    [Test Target] --language-only; --tp-size
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
                "--language-only",
                "--tp-size",
                "2",
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

    def test_language_only_health(self):
        """Verify the language-only server is healthy after startup.

        HTTP 200 from /health_generate confirms the language model finished loading
        and is ready to accept inference requests. Visual encoder weights are absent,
        so initialization is faster than a full VLM server.
        """
        response = requests.get(f"{self.base_url}/health_generate", timeout=10)
        self.assertEqual(
            response.status_code,
            200,
            "Language-only server failed health check; language model may not have "
            "loaded correctly with tp-size=2.",
        )

    def test_language_only_text_generation(self):
        """Verify the language-only server correctly handles a text-only inference request.

        A language-only VLM server must process text prompts without a visual encoder.
        The expected answer ('Paris') is stable and unambiguous, making it a reliable
        correctness signal. Temperature=0 ensures deterministic output.

        This assertion rules out scenarios where the server starts but the language
        model forward pass silently fails or produces garbage output.
        """
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        generated_text = response.json().get("text", "")
        self.assertIn(
            "Paris",
            generated_text,
            f"Language-only server returned unexpected output: '{generated_text}'. "
            "Expected 'Paris' for the capital of France.",
        )


if __name__ == "__main__":
    unittest.main()
