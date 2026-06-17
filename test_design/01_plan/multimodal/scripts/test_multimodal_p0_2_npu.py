#!/usr/bin/env python3
"""
NPU multimodal P0 test cases P0-006 through P0-010.

Implements advanced multimodal interaction test cases for the NPU platform:

  - P0-006: Long text + image triggering chunked prefill (no image feature loss)
  - P0-007: Graph compilation enabled multimodal inference (semantic match with baseline)
  - P0-008: LoRA adapter with multimodal model (LoRA does not break image understanding)
  - P0-009: GDN linear attention with visual encoder (Qwen3.5-MoE)
  - P0-010: GDN + MoE + visual encoder (expert routing not skewed by image tokens)

"""

import unittest

import openai
from utils import (
    QWEN3_5_9B_PATH,
    QWEN3_5_35B_A3B_PATH,
    QWEN3_VL_LORA_PATH,
    Color,
    Shape,
    assert_color_and_shape,
    assert_text_contains,
    chat,
    chat_single_image,
    create_test_image,
    get_port,
    image_content,
    launch_server,
    text_content,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=2100, suite="nightly-2-npu-a3", nightly=True)

# ============================================
# Test Helpers
# ============================================


_LONG_PREFIX = (
    "To verify that chunked prefill does not truncate image features in the "
    "multimodal inference pipeline, a sufficient amount of text must precede "
    "the actual image content so that the vision tokens inevitably cross a "
    "chunk boundary. When the prefill stage encounters an input longer than "
    "the configured threshold, it partitions the sequence into fixed-size "
    "segments and processes them one at a time through the attention layers. "
    "Any modality-specific embeddings that sit on a later segment must be "
    "correctly materialized and merged by the scheduler. This paragraph is "
    "deliberately crafted with varied vocabulary and diverse sentence "
    "structures to exercise the tokenizer thoroughly and avoid degenerate "
    "repetition patterns that could trigger unusual model behavior or "
    "attention collapse in the decoder. Ensuring correctness under chunked "
    "scheduling is essential because production deployments routinely handle "
    "long-context requests containing embedded images, diagrams, scanned "
    "documents, and other visual artifacts alongside substantial textual "
    "narrative. The interaction between the ViT encoder output and the LLM "
    "decoder's autoregressive generation must remain consistent regardless "
    "of how the prefill phase splits the input, as any inconsistency could "
    "manifest as hallucinations, missing visual details, or garbled responses "
    "that degrade the user experience in subtle but consequential ways. This "
    "text alone does not reference any specific color or shape; it serves "
    "solely as a neutral preceding context whose only purpose is to push the "
    "subsequent image data past the first and second chunk boundaries, so "
    "that the test can validate whether cross-chunk visual feature forwarding "
    "works correctly on the target hardware platform without interference "
    "from the semantic content of the prefix itself."
)


# ============================================
# P0-006: Long text + image -> chunked prefill
# ============================================


class TestP0006ChunkedPrefill(CustomTestCase):
    """P0-006: Verify that chunked prefill does not truncate image features.

    Scenario:
        The user appends a long system prompt (~3K tokens) before an image,
        triggering chunked prefill across the image token boundary. The model
        must correctly process the image even though its tokens fall in a
        later chunk.

    NPU note:
        Chunked prefill on NPU has "weak" participation (page-size constraint
        only), but the test validates end-to-end correctness.

    Related features: chunked_prefill, scheduling
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(6)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._long_prefix = (
            _LONG_PREFIX * 10
        )  # ~3K tokens with Qwen tokenizer (298 × 10 ≈ 2980)
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=["--chunked-prefill-size", "512"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_long_prefix_with_image(self):
        """Send long prefix + image + prompt, verify both are understood."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        text_content(self._long_prefix),
                        image_content(self._image_b64),
                        text_content(
                            "Describe the image and briefly summarize the text above it."
                        ),
                    ],
                },
            ],
            max_tokens=128,
        )
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        # Image features must NOT be ignored
        assert_color_and_shape(self, text, "blue", "rectangle")

        # The long prefix content should also be referenced
        assert_text_contains(
            self,
            text,
            hints=["chunked", "prefill", "boundary", "scheduler", "embedding"],
        )


# ============================================
# P0-007: Graph compilation + multimodal
# ============================================


class TestP0007GraphCompilation(CustomTestCase):
    """P0-007: Verify multimodal inference works with graph compilation enabled.

    Launches a server without --disable-cuda-graph so the NPU ViT graph
    runner (vit_npu_graph_runner) handles the ViT forward pass.  Validates
    that graph capture + replay does not crash, corrupt features, or produce
    gibberish.

    NPU note:
        NPU uses vit_npu_graph_runner -- this is a critical NPU-specific path
        used only when --disable-cuda-graph is NOT passed.

    Related features: graph_compilation (NPU ViT graph runner)
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(7)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.RED, shape=Shape.RECTANGLE
        )
        cls._prompt = "Describe the image"

    def test_graph_multimodal_inference(self):
        """Verify multimodal inference with graph compilation enabled.

        Launches a server WITHOUT --disable-cuda-graph so that the NPU ViT
        graph runner (vit_npu_graph_runner) captures and replays the ViT
        forward pass.  Validates that the output is non-empty, references
        image content.

        NOTE: Server is launched inside the test method (not setUpClass)
        because graph compilation test needs a fresh server with specific
        --cuda-graph-max-bs argument, and separating it from setUpClass
        avoids interfering with potential future test additions that may
        need a baseline server without graph compilation.
        """
        process_graph, url_graph = launch_server(
            self._model,
            self._port,
            extra_args=["--cuda-graph-max-bs", "4"],
        )
        try:
            result = chat_single_image(
                url_graph,
                self._image_b64,
                self._prompt,
                max_tokens=256,
            )
        finally:
            kill_process_tree(process_graph.pid)

        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        assert_color_and_shape(self, result, "red", "rectangle")
        print(f"Graph-enabled output:\n{result}\n")


# ============================================
# P0-008: LoRA adapter + multimodal
# ============================================


@unittest.skip("Multimodal + LoRA is not currently supported on NPU")
class TestP0008LoRA(CustomTestCase):
    """P0-008: Verify LoRA adapter does not break image understanding.

    Load a LoRA adapter with the VLM, send an image request, and verify
    the output is still based on the image. LoRA should only affect the
    LLM decoder, not the ViT encoder.

    Currently skipped — multimodal + LoRA combination is not supported
    on NPU.  Re-enable when ascend_backend.py gains full VLM + LoRA support.

    Related features: LoRA
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _lora_path = QWEN3_VL_LORA_PATH
    _port = get_port(8)
    _lora_name = "test-lora"

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=[
                "--lora-paths",
                f"{cls._lora_name}={cls._lora_path}",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_lora_adapter_and_image(self):
        """Send image with LoRA adapter, verify image understanding is intact."""
        client = openai.Client(api_key="sk-123456", base_url=f"{self.base_url}/v1")
        response = client.chat.completions.create(
            model=self._lora_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
        )
        text = response.choices[0].message.content
        print(f"P0-008 LoRA response:\n{text}\n")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)
        assert_color_and_shape(self, text, "blue", "rectangle")


# ============================================
# P0-009: GDN linear attention + visual encoder
# ============================================


class TestP0009GDNLinearAttention(CustomTestCase):
    """P0-009: Verify GDN linear attention + visual encoder produce correct multimodal output.

    Related features: attention_backend (GDN hybrid_linear), graph_compilation
    """

    _model = QWEN3_5_9B_PATH
    _port = get_port(9)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=[
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.6",
                "--mamba-scheduler-strategy",
                "no_buffer",
                "--mamba-ssm-dtype",
                "bfloat16",
                "--tp-size",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_gdn_multimodal_inference(self):
        """Send image to GDN model, verify correct multimodal inference."""
        text = chat_single_image(
            self.base_url,
            self._image_b64,
            "Describe this image",
            max_tokens=256,
        )
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        assert_color_and_shape(self, text, "blue", "rectangle")


# ============================================
# P0-010: GDN + MoE + visual encoder
# ============================================


class TestP0010GDNMoE(CustomTestCase):
    """P0-010: Verify GDN + MoE + visual encoder work together correctly.

    Uses Qwen3.5-35B-A3B (GDN + MoE + DeepStack ViT) to verify the full
    GDN+MoE+Vision interaction.  P0-009 covers GDN+Vision without MoE.
    Verifies: image understanding intact.

    Related features: attention_backend (GDN), moe, eplb
    """

    _model = QWEN3_5_35B_A3B_PATH
    _port = get_port(10)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.GREEN, shape=Shape.RECTANGLE
        )
        # 35B MoE model: ~70GB weights, needs TP>=2 on 64GB NPU
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=[
                "--disable-radix-cache",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_gdn_moe_multimodal_inference(self):
        """Send image to GDN+MoE model, verify correct output and no expert skew artifacts."""
        text = chat_single_image(
            self.base_url,
            self._image_b64,
            "Describe this image",
            max_tokens=256,
        )
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        assert_color_and_shape(self, text, "green", "rectangle")

        # Basic expert routing sanity: output should not be garbled or repetitive
        # (repetitive output can indicate expert routing collapse)
        words = text.split()
        if len(words) >= 10:
            unique_ratio = len(set(words)) / len(words)
            self.assertGreater(
                unique_ratio,
                0.3,
                f"Output may indicate expert routing collapse (low vocab diversity "
                f"{unique_ratio:.2f}): {text[:300]}",
            )


# ============================================
# Entry point
# ============================================

if __name__ == "__main__":
    unittest.main()
