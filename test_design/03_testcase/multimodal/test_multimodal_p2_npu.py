"""
NPU multimodal P2 test cases.

This file implements edge-case P2 multimodal interaction tests for the NPU
platform as defined in the multimodal interaction test analysis report
(Section 2.3).

Implemented test cases:
  - P2-002: Context Parallelism + multi-image -> long context correct
  - P2-004: Quantization (w8a8) + image -> accuracy acceptable
  - P2-006: Pipeline Parallelism + image -> pipeline correct
  - P2-008: EPLB + image -> elastic load balancing correct
  - P2-009: Reasoning Parser + image -> think tags preserved
  - P2-010: Full model DP + image -> multi-replica inference correct
  - P2-011: Quantization + chunked prefill + image -> accuracy persists
  - P2-013: DP-attention + NPU Graph + image -> DP graph replay correct
  - P2-014: Multistream MoE + image -> dual-stream routing correct

Each test class is independent with its own setUpClass/tearDownClass, running
a fresh server on a dedicated port.

"""

import unittest

import openai
import requests
from utils import (
    QWEN3_5_9B_PATH,
    QWEN3_5_35B_A3B_PATH,
    QWEN3_5_35B_A3B_W8A8_MTP_PATH,
    Color,
    Shape,
    assert_color_and_shape,
    assert_text_contains,
    chat,
    create_test_image,
    get_port,
    image_content,
    launch_server,
    text_content,
)

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# CI registration
#   P2-009 uses 1 NPU with a 4B model.
#   All other tests use 2 NPU with 35B-A3B or 4B models.
# ---------------------------------------------------------------------------
register_npu_ci(est_time=1800, suite="nightly-2-npu-a3", nightly=True)


# ============================================
# Shared constants for 35B-A3B tests
# ============================================

# The 35B-A3B MoE model needs TP=2 on 64 GB NPU cards.
# See test_multimodal_p0_advanced_npu.py for the established pattern.
_MOE_COMMON_ARGS = [
    "--tp-size",
    "2",
    "--mem-fraction-static",
    "0.7",
    "--disable-radix-cache",
]

# Long prefix (~5K tokens) for tests that need chunked prefill to trigger.
# Placing an image after this prefix forces image tokens to cross chunk
# boundaries (chunk_size=512), exercising cross-chunk state management.
_LONG_PREFIX = (
    "This paragraph is used to generate a long text prefix that triggers "
    "chunked prefill in the SGLang multimodal inference pipeline. When a "
    "large language model processes a sequence longer than the configured "
    "chunk size, the prefill phase splits the sequence into multiple chunks "
    "and computes attention on each chunk sequentially. This text is designed "
    "to produce approximately five thousand tokens with the Qwen tokenizer, "
    "ensuring that any image tokens appended after it will land on a later "
    "chunk rather than the first one. The vocabulary is intentionally varied "
    "to avoid degenerate repetition patterns that could affect model output. "
    "Chunk boundaries are a known source of subtle bugs in multimodal "
    "inference because visual features computed by the ViT encoder must be "
    "preserved and correctly reloaded when the next chunk is processed. "
)


# ===================================================================
# Shared teardown helper
# ===================================================================


def _teardown_server(cls, attr="process"):
    if hasattr(cls, attr) and getattr(cls, attr) is not None:
        try:
            kill_process_tree(getattr(cls, attr).pid)
        except Exception:
            pass


# ============================================
# P2-002: Context Parallelism + multi-image -> long context correct
# ============================================


class TestP2002ContextParallelism(CustomTestCase):
    """P2-002: Verify context parallelism works with multi-image input.

    Deploy Qwen3-VL-4B with ``--tp-size 2 --attn-cp-size 2``, send
    two images with a comparison prompt, and verify the output
    independently references **both** images.  Also confirms via
    ``/server_info`` that CP is active.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(30)

    @classmethod
    def setUpClass(cls):
        _, cls._image1_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.RED,
            shape=Shape.ELLIPSE,
        )
        _, cls._image2_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.BLUE,
            shape=Shape.RECTANGLE,
        )

    def test_context_parallelism_multi_image(self):
        """Send 2 images with CP=2, verify both are independently described."""
        process, url = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--tp-size",
                "2",
                "--attn-cp-size",
                "2",
                "--mem-fraction-static",
                "0.3",
            ],
        )
        try:
            text = chat(
                url,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content(self._image1_b64),
                            image_content(self._image2_b64),
                            text_content(
                                "Describe both images. "
                                "What colors and shapes do you see?"
                            ),
                        ],
                    }
                ],
                max_tokens=256,
            )

            self.assertIsNotNone(text, "P2-002: Response is None")
            self.assertGreater(len(text), 0, "P2-002: Response is empty")

            # Both images must be described — each color mentioned
            text_lower = text.lower()
            self.assertIn(
                "red",
                text_lower,
                "P2-002: First image (red ellipse) not described — "
                "CP may have lost its tokens",
            )
            self.assertIn(
                "blue",
                text_lower,
                "P2-002: Second image (blue rectangle) not described — "
                "CP may have lost its tokens",
            )
            self.assertIn(
                "ellipse",
                text_lower,
                "P2-002: Ellipse shape from first image not mentioned",
            )
            self.assertIn(
                "rectangle",
                text_lower,
                "P2-002: Rectangle shape from second image not mentioned",
            )

            # Verify CP is actually active
            server_info = requests.get(url + "/server_info", timeout=10).json()
            self.assertEqual(
                server_info.get("attn_cp_size"),
                2,
                f"P2-002: attn_cp_size={server_info.get('attn_cp_size')} != 2",
            )

            print(f"  [P2-002] output_len={len(text)}")
        finally:
            kill_process_tree(process.pid)


# ============================================
# P2-004: Quantization (w8a8) + image -> accuracy acceptable
# ============================================


class TestP2004Quantization(CustomTestCase):
    """P2-004: Verify w8a8 quantization does not degrade image understanding.

    Deploy Qwen3.5-35B-A3B-w8a8-mtp (weight-only INT8 quantization),
    send an image request, and verify the output references visual content.
    The quantized model should produce semantically equivalent output to
    the FP16 baseline.
    """

    _model = QWEN3_5_35B_A3B_W8A8_MTP_PATH
    _port = get_port(23)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.GREEN,
            shape=Shape.ELLIPSE,
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=_MOE_COMMON_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        _teardown_server(cls)

    def test_quantization_image(self):
        """Send image to w8a8 quantized model, verify output."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(text, "P2-004: Response is None")
        self.assertGreater(len(text), 0, "P2-004: Response is empty")

        # Quantized model must correctly identify the expected color and shape
        assert_color_and_shape(
            self,
            text,
            "green",
            "ellipse",
            prefix="P2-004: ",
        )

        print(f"  [P2-004] output_len={len(text)}")


# ============================================
# P2-006: Pipeline Parallelism + image -> pipeline correct
# ============================================


class TestP2006PipelineParallelism(CustomTestCase):
    """P2-006: Verify pipeline parallelism with image input.

    Deploy Qwen3-VL-4B with ``--pp-size 2``, send an image request,
    and verify the output correctly identifies the expected color and
    shape.  ViT runs on stage 0 — image embeddings cross the P2P
    boundary to stage 1.  Also confirms ``/server_info`` reports PP
    is active.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(31)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.PURPLE,
            shape=Shape.ELLIPSE,
        )

    def test_pipeline_parallelism_image(self):
        """Send image with PP=2, verify output and feature gating."""
        process, url = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--pp-size",
                "2",
                "--mem-fraction-static",
                "0.3",
            ],
        )
        try:
            text = chat(
                url,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content(self._image_b64),
                            text_content("Describe the image"),
                        ],
                    }
                ],
                max_tokens=128,
            )

            self.assertIsNotNone(text, "P2-006: Response is None")
            self.assertGreater(len(text), 0, "P2-006: Response is empty")

            assert_color_and_shape(
                self,
                text,
                "purple",
                "ellipse",
                prefix="P2-006: ",
            )

            # Verify PP is actually active
            server_info = requests.get(url + "/server_info", timeout=10).json()
            self.assertEqual(
                server_info.get("pp_size"),
                2,
                f"P2-006: pp_size={server_info.get('pp_size')} != 2",
            )

            print(f"  [P2-006] output_len={len(text)}")
        finally:
            kill_process_tree(process.pid)


# ============================================
# P2-008: EPLB + image -> elastic load balancing correct
# ============================================


class TestP2008EPLB(CustomTestCase):
    """P2-008: Verify EPLB does not disrupt image inference.

    Deploy Qwen3.5-35B-A3B with ``--enable-eplb``, send an image
    request, and verify the output correctly identifies the expected
    color and shape.  EPLB is the only test combining expert load
    balancing with multimodal input.

    Pattern modelled after ``test/manual/ep/test_eplb.py``.
    """

    _model = QWEN3_5_35B_A3B_PATH
    _port = get_port(24)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.PURPLE,
            shape=Shape.RECTANGLE,
        )
        cls.base_url = f"http://127.0.0.1:{cls._port}"
        with (
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
            envs.SGLANG_EXPERT_LOCATION_UPDATER_CANARY.override(True),
        ):
            cls.process = popen_launch_server(
                cls._model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--device",
                    "npu",
                    "--attention-backend",
                    "ascend",
                    "--trust-remote-code",
                    "--enable-multimodal",
                    "--mm-attention-backend",
                    "ascend_attn",
                    "--tp-size",
                    "2",
                    "--dp-size",
                    "2",
                    "--enable-dp-attention",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",
                    "--enable-eplb",
                    "--ep-num-redundant-experts",
                    "4",
                    "--eplb-rebalance-num-iterations",
                    "50",
                    "--expert-distribution-recorder-buffer-size",
                    "50",
                    "--expert-distribution-recorder-mode",
                    "stat",
                    "--ep-dispatch-algorithm",
                    "static",
                ],
                env={
                    "HCCL_BUFFSIZE": "1024",
                },
            )

    @classmethod
    def tearDownClass(cls):
        _teardown_server(cls)

    def test_eplb_image(self):
        """Send image with EPLB enabled, verify output."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(text, "P2-008: Response is None")
        self.assertGreater(len(text), 0, "P2-008: Response is empty")

        assert_color_and_shape(
            self,
            text,
            "purple",
            "rectangle",
            prefix="P2-008: ",
        )

        server_info = requests.get(self.base_url + "/server_info", timeout=10).json()
        self.assertTrue(
            server_info.get("enable_eplb"),
            "P2-008: enable_eplb is not True — EPLB was silently disabled",
        )

        print(f"  [P2-008] output_len={len(text)}")


# ============================================
# P2-009: Reasoning Parser + image -> think tags preserved
# ============================================


class TestP2009ReasoningParser(CustomTestCase):
    """P2-009: Verify reasoning parser with image input.

    +--------------------+------------------------------------------------+
    | separate_reasoning | Expected behaviour                             |
    +--------------------+------------------------------------------------+
    | True               | content split: think tags stripped, tags only  |
    |                    | in reasoning_content                           |
    +--------------------+------------------------------------------------+
    | False              | think tags preserved in message.content,       |
    |                    | reasoning_content is None                      |
    +--------------------+------------------------------------------------+

    Both ``separate_reasoning`` values are tested against a single server
    launched with ``--reasoning-parser qwen3``.

    Note: The model must support the reasoning parser natively. Qwen3.5-9B
    produces garbled output when ``enable_thinking`` is turned on, so the
    35B-A3B MoE model is used instead.
    """

    _model = QWEN3_5_35B_A3B_PATH
    _port_parser = get_port(25)

    _BASE_ARGS = [
        "--tp-size",
        "2",
        "--mem-fraction-static",
        "0.7",
        "--disable-radix-cache",
        "--disable-cuda-graph",
    ]

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.GREEN,
            shape=Shape.ELLIPSE,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reasoning_request(client, image_b64, separate_reasoning):
        return client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(image_b64),
                        text_content(
                            "What color is the shape and what shape is drawn "
                            "in this image? Answer with the color and shape only."
                        ),
                    ],
                }
            ],
            temperature=0,
            max_tokens=512,
            extra_body={
                "enable_thinking": True,
                "separate_reasoning": separate_reasoning,
            },
        )

    def _launch_with_parser(self, port):
        return launch_server(
            self._model,
            port,
            extra_args=["--reasoning-parser", "qwen3"] + self._BASE_ARGS,
        )

    @staticmethod
    def _verify_image_content(test, text, prefix):
        assert_color_and_shape(
            test,
            text,
            "green",
            "ellipse",
            prefix=prefix,
        )

    # ------------------------------------------------------------------
    # Test: both separate_reasoning values with qwen3 parser
    # ------------------------------------------------------------------

    def test_separate_reasoning_modes(self):
        """Test both separate_reasoning values with --reasoning-parser qwen3."""
        process, url = self._launch_with_parser(self._port_parser)
        try:
            client = openai.Client(
                api_key="sk-123456",
                base_url=f"{url}/v1",
            )

            # --- separate_reasoning=True ---
            response = self._reasoning_request(
                client,
                self._image_b64,
                separate_reasoning=True,
            )
            msg = response.choices[0].message
            content = msg.content
            reasoning = msg.reasoning_content

            print(f"  [P2-009-sep_true] content={repr(content[:200])}")
            print(
                f"  [P2-009-sep_true] reasoning_content={repr(reasoning[:500] if reasoning else None)}"
            )

            self.assertIsNotNone(
                reasoning, "P2-009-sep_true: reasoning_content is None"
            )
            self.assertGreater(
                len(reasoning),
                0,
                "P2-009-sep_true: reasoning_content is empty",
            )
            if content:
                self._verify_image_content(self, content, "P2-009-sep_true: ")
            assert_color_and_shape(
                self,
                reasoning,
                "green",
                "ellipse",
                prefix="P2-009-sep_true: ",
            )
            print(
                f"  [P2-009-sep_true] content_len={len(content)}, reasoning_len={len(reasoning)}, split=OK"
            )

            # --- separate_reasoning=False ---
            response = self._reasoning_request(
                client,
                self._image_b64,
                separate_reasoning=False,
            )
            msg = response.choices[0].message
            text = msg.content
            print(f"  [P2-009-sep_false] content={repr(text[:300])}")
            self.assertIsNone(
                msg.reasoning_content,
                "P2-009-sep_false: reasoning_content should be None when "
                "separate_reasoning=False with qwen3 parser — reasoning "
                "is merged into content",
            )
            self._verify_image_content(self, text, "P2-009-sep_false: ")
            print(f"  [P2-009-sep_false] output_len={len(text)}")
        finally:
            kill_process_tree(process.pid)


# ============================================
# P2-010: Full model DP + image -> multi-replica inference correct
# ============================================


class TestP2010FullModelDP(CustomTestCase):
    """P2-010: Verify full-model data parallelism (2 replicas) with image input.

    Deploy Qwen3-VL-4B with ``--dp-size 2``, send an image request, and
    verify the output references visual content.  Also confirms via
    ``/server_info`` that DP is active.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(26)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            color=Color.PURPLE,
            shape=Shape.RECTANGLE,
        )

    def test_full_model_dp_image(self):
        """Send image request with DP=2, verify output and feature gating."""
        process, url = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--dp-size",
                "2",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        try:
            text = chat(
                url,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content(self._image_b64),
                            text_content("Describe the image"),
                        ],
                    }
                ],
                max_tokens=128,
            )

            self.assertIsNotNone(text, "P2-010: Response is None")
            self.assertGreater(len(text), 0, "P2-010: Response is empty")

            assert_color_and_shape(
                self,
                text,
                "purple",
                "rectangle",
                prefix="P2-010: ",
            )

            server_info = requests.get(url + "/server_info", timeout=10).json()
            self.assertEqual(
                server_info.get("dp_size"),
                2,
                f"P2-010: dp_size={server_info.get('dp_size')} != 2",
            )

            print(f"  [P2-010] output_len={len(text)}")
        finally:
            kill_process_tree(process.pid)


# ============================================
# P2-011: Quantization + chunked prefill + image -> accuracy persists
# ============================================


class TestP2011QuantizationChunked(CustomTestCase):
    """P2-011: Verify quantized accuracy survives chunked prefill boundaries.

    Deploy Qwen3.5-35B-A3B-w8a8-mtp with chunked prefill, send a long
    text prefix + image request, and verify the image content is
    recognized despite quantization + chunk boundary crossing.
    """

    _model = QWEN3_5_35B_A3B_W8A8_MTP_PATH
    _port = get_port(27)
    _long_text = _LONG_PREFIX * 25  # ~5.7K tokens

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            color=Color.TEAL,
            shape=Shape.ELLIPSE,
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=_MOE_COMMON_ARGS
            + [
                "--chunked-prefill-size",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        _teardown_server(cls)

    def test_quantization_chunked_image(self):
        """Send long prefix + image with quantized chunked prefill, verify output."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        text_content(self._long_text),
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(text, "P2-011: Response is None")
        self.assertGreater(len(text), 0, "P2-011: Response is empty")

        assert_color_and_shape(
            self,
            text,
            "teal",
            "ellipse",
            prefix="P2-011: ",
        )

        print(f"  [P2-011] output_len={len(text)}")


# ============================================
# P2-013: DP-attention + NPU Graph + image -> DP graph replay correct
# ============================================


class TestP2013DPAttentionNPUGraph(CustomTestCase):
    """P2-013: Verify DP-attention works with NPU Graph enabled for images.

    Deploy Qwen3-VL-4B with ``--enable-dp-attention --dp-size 2`` and
    **without** ``--disable-cuda-graph`` (NPU Graph is ON by default),
    send an image request, and verify output correctness.  Also confirms
    via ``/server_info`` that DP-attention is active.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(28)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            color=Color.TEAL,
            shape=Shape.ELLIPSE,
        )

    def test_dp_attention_npu_graph_image(self):
        """Send image request with DP-attention + NPU Graph, verify output."""
        process, url = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--enable-dp-attention",
                "--dp-size",
                "2",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        try:
            text = chat(
                url,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content(self._image_b64),
                            text_content("Describe the image"),
                        ],
                    }
                ],
                max_tokens=128,
            )

            self.assertIsNotNone(text, "P2-013: Response is None")
            self.assertGreater(len(text), 0, "P2-013: Response is empty")

            assert_color_and_shape(
                self,
                text,
                "teal",
                "ellipse",
                prefix="P2-013: ",
            )

            server_info = requests.get(url + "/server_info", timeout=10).json()
            self.assertTrue(
                server_info.get("enable_dp_attention"),
                "P2-013: enable_dp_attention is not True in server info — "
                "DP-attention was silently disabled",
            )
            self.assertEqual(
                server_info.get("dp_size"),
                2,
                f"P2-013: dp_size={server_info.get('dp_size')} != 2",
            )

            print(f"  [P2-013] output_len={len(text)}")
        finally:
            kill_process_tree(process.pid)


# ============================================
# P2-014: Multistream MoE + image -> dual-stream routing correct
# ============================================


class TestP2014MultistreamMoE(CustomTestCase):
    """P2-014: Verify dual-stream MoE execution does not break image routing.

    Deploy Qwen3.5-35B-A3B with ``SGLANG_NPU_USE_MULTI_STREAM=1``, send
    an image request, and verify expert routing remains correct under
    dual-stream execution.
    """

    _model = QWEN3_5_35B_A3B_PATH
    _port = get_port(29)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            color=Color.RED,
            shape=Shape.ELLIPSE,
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=_MOE_COMMON_ARGS,
            extra_env={"SGLANG_NPU_USE_MULTI_STREAM": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        _teardown_server(cls)

    def test_multistream_moe_image(self):
        """Send image with dual-stream MoE enabled, verify output."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(text, "P2-014: Response is None")
        self.assertGreater(len(text), 0, "P2-014: Response is empty")

        assert_color_and_shape(
            self,
            text,
            "red",
            "ellipse",
            prefix="P2-014: ",
        )

        print(f"  [P2-014] output_len={len(text)}")


if __name__ == "__main__":
    unittest.main()
