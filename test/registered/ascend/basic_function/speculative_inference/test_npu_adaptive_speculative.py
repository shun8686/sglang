"""Test Adaptive Speculative Decoding on NPU.

[Test Category] Speculative Decoding
[Test Target] --speculative-algorithm=EAGLE3; --speculative-adaptive;
--speculative-adaptive-config; --speculative-num-steps (dynamic);
--speculative-eagle-topk; --speculative-num-draft-tokens
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] Ported from GPU test: sgl-project/sglang test/test_adaptive_speculative.py
  Class: TestAdaptiveSpeculativeServer

Porting notes:
  - attention-backend: triton -> ascend
  - model: Qwen2.5-1.5B-Instruct -> Qwen3-8B (NPU CI pre-installed)
  - mem-fraction-static: 0.85 -> 0.7 (NPU standard)
  - GSM8K threshold: 0.20 -> 0.69 (stricter, consistent with NPU spec tests)
  - Added NPU env vars (SGLANG_ENABLE_SPEC_V2, etc.)
  - register_cuda_ci -> register_npu_ci
  - print() -> logger.info()
  - Added --disable-cuda-graph (NPU doesn't support CUDA Graph)
  - Added --sampling-backend ascend
  - TestAdaptiveZeroStepBatchSizeServer NOT ported (depends on GPU routing logic)

Key adaptation from GPU version:
  The GPU test uses /set_args to manually drive upshift/downshift. The NPU
  image does not have /set_args, so this test relies on the natural adaptive
  behavior: --speculative-adaptive automatically adjusts speculative_num_steps
  based on observed acceptance lengths. We verify the feature is enabled via
  /server_info and that inference + GSM8K still works correctly.
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


NPU_ENV = {
    **os.environ,
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}


class TestNPUAdaptiveSpeculativeServer(CustomTestCase):
    """Test Adaptive Speculative Decoding on NPU.

    Ported from GPU: sgl-project/sglang test/test_adaptive_speculative.py
    Class: TestAdaptiveSpeculativeServer

    This test verifies that the adaptive speculative decoding system:
    1. Starts with --speculative-adaptive enabled
    2. /server_info reflects speculative_adaptive=True
    3. Can handle inference requests without errors
    4. Maintains GSM8K accuracy with adaptive enabled

    Note: Unlike the GPU version which uses /set_args to manually drive
    upshift/downshift, this test relies on the natural adaptive behavior.
    The /set_args endpoint is not available in the NPU CI image.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
            "--tp-size",
            "1",
            "--sampling-backend",
            "ascend",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "8",
            "--speculative-adaptive",
        ]

        logger.info("Starting Adaptive Speculative server on NPU...")
        logger.info("Model: %s", cls.model)
        logger.info("Draft model: %s", cls.draft_model)
        logger.info("speculative-adaptive=True, initial num_steps=1")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
            env=NPU_ENV,
        )
        logger.info("Adaptive server started successfully.")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _get_server_info(self):
        resp = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _send_one_prompt(self):
        """Send one prompt via send_one_prompt with correct BenchArgs."""
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        args = BenchArgs(host=parsed.hostname, port=parsed.port)
        send_one_prompt(args, print_output=False)

    def test_a_adaptive_enabled(self):
        """Verify --speculative-adaptive is enabled in server info.

        The adaptive feature may be silently disabled by the framework if
        the server args are incompatible (e.g. dp_attention, topk>1). We
        verify it is actually enabled.
        """
        info = self._get_server_info()
        adaptive_enabled = info.get("speculative_adaptive", False)
        logger.info("speculative_adaptive in /server_info: %s", adaptive_enabled)
        self.assertTrue(
            adaptive_enabled,
            "speculative_adaptive should be True in /server_info. "
            "If False, the framework may have silently disabled it due to "
            f"unsupported config. Full info keys: {list(info.keys())}",
        )

        # Verify speculative_num_steps is present
        num_steps = info.get("speculative_num_steps")
        logger.info("Initial speculative_num_steps: %s", num_steps)
        self.assertIsNotNone(num_steps, "speculative_num_steps should be present")

    def test_b_inference_triggers_adaptive(self):
        """Send prompts to trigger natural adaptive behavior.

        The adaptive algorithm monitors acceptance lengths and adjusts
        speculative_num_steps automatically. We send a batch of prompts
        to exercise the adaptive logic, then check that speculative_num_steps
        is still valid (present and a positive integer).

        We do NOT assert a specific value because the natural adaptive
        behavior depends on the model, the prompts, and the accept rate,
        which are non-deterministic.
        """
        logger.info("=== Sending prompts to trigger adaptive behavior ===")

        # Send a batch of prompts to exercise the adaptive logic
        for i in range(10):
            self._send_one_prompt()
            logger.info("Sent prompt %d/10", i + 1)

        # Check that speculative_num_steps is still valid after inference
        info = self._get_server_info()
        num_steps = info.get("speculative_num_steps")
        logger.info("speculative_num_steps after prompts: %s", num_steps)
        self.assertIsNotNone(num_steps, "speculative_num_steps should be present")
        self.assertIsInstance(
            num_steps,
            int,
            f"speculative_num_steps should be int, got {type(num_steps)}",
        )
        self.assertGreater(
            num_steps, 0, f"speculative_num_steps should be > 0, got {num_steps}"
        )

        # Verify adaptive is still enabled
        adaptive_enabled = info.get("speculative_adaptive", False)
        self.assertTrue(adaptive_enabled, "speculative_adaptive should still be True")
        logger.info("Adaptive behavior exercised. speculative_num_steps=%s", num_steps)

    def test_c_gsm8k(self):
        """Verify GSM8K accuracy with adaptive speculative decoding enabled."""
        requests.get(self.base_url + "/flush_cache", timeout=30)

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        logger.info("GSM8K metrics (adaptive speculative): %s", metrics)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Adaptive Speculative on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
            )

        # NPU uses stricter threshold (0.69) than GPU (0.20)
        self.assertGreater(
            metrics["score"],
            0.69,
            "GSM8K score should be > 0.69 with adaptive speculative",
        )


if __name__ == "__main__":
    unittest.main()
