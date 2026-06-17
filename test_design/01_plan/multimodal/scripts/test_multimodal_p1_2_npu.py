"""
NPU multimodal P1 test cases P1-006 through P1-010.

This file implements P1 multimodal interaction test cases for the NPU platform
as defined in the multimodal interaction test analysis report (Section 2.2).

Implemented test cases:
  - P1-006: Structured output + image -> JSON Schema constraint
  - P1-007: Tool call + image -> image param in function arguments
  - P1-008: CPU offloading + image -> inference correctness
  - P1-009: DP-attention + DP LM Head + image -> LM head sharding
  - P1-010: Overlap Schedule + speculative decoding + image -> multimodal drafts

Each test class is independent with its own setUpClass/tearDownClass, running
a fresh server on a dedicated port.

Usage:
    python3 -m unittest \\
        test_design/03_testcase/multimodal/test_multimodal_p1_part2_npu.py
"""

import json
import time
import unittest

import openai
import requests
from PIL import Image, ImageDraw
from utils import (
    QWEN3_VL_8B_EAGLE3_PATH,
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

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# CI registration — nightly suite, 2 NPU
# P1-009 needs --dp-size 2 (DP LM Head requires 2 DP ranks).
# Each test class starts its own server sequentially; total runtime ~800 s.
# ---------------------------------------------------------------------------
register_npu_ci(est_time=900, suite="nightly-2-npu-a3", nightly=True)

# ---------------------------------------------------------------------------
# Environment & Constants
# ---------------------------------------------------------------------------

# Ports: _port = get_port(N)  (N = per-class offset)

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _generate_long_prefix_5k():
    """Generate a long text prefix (~5K tokens) for chunked prefill + offloading.

    English text is approximately 4 characters per token. This produces
    roughly 5000 tokens to ensure robust chunk boundary crossing.
    """
    paragraph = (
        "This is a paragraph used for testing chunked prefill and CPU offloading "
        "functionality in the SGLang multimodal inference pipeline. It contains "
        "multiple sentences with varied vocabulary to ensure the token count reaches "
        "approximately five thousand tokens when processed by the Qwen tokenizer. "
        "The purpose is to verify that image features are not truncated or lost "
        "when they span across chunk boundaries during the prefill phase, and "
        "that CPU offloading correctly preserves and restores image embeddings "
        "when model weights are swapped between GPU and CPU memory. "
        "Chunked prefill splits long sequences into multiple chunks and computes "
        "attention on each chunk sequentially. If image features overlap with "
        "a chunk boundary, the offloaded embeddings must be correctly reloaded "
        "when the next chunk is processed. "
        "CPU offloading moves infrequently accessed model weights from GPU memory "
        "to CPU memory, reducing GPU memory usage at the cost of increased "
        "latency for transferring weights back when needed. "
    )
    # ~340 characters per paragraph * 60 = ~20400 chars ≈ 5100 tokens
    return paragraph * 60


# ============================================
# P1-006: Structured output + image -> JSON Schema constraint
# ============================================


class TestP1006StructuredOutput(CustomTestCase):
    """P1-006: Verify structured output (JSON Schema) with image input.

    Scenario:
        The user provides an image of a coloured shape and requests extraction
        of color and shape in JSON format via the OpenAI-compatible
        ``response_format`` parameter with a JSON schema constraint.

    Related features: structured_outputs (xgrammar)

    NPU note:
        xgrammar is ``platform_agnostic`` (CPU-side grammar engine). ViT
        encoding is unaffected by the grammar constraint. Simple, low-risk test.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(16)

    @classmethod
    def setUpClass(cls):
        cls._expected_color = Color.BLUE.name.lower()
        cls._expected_shape = Shape.RECTANGLE.value
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=["--mem-fraction-static", "0.4"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_structured_output_json_schema(self):
        """Send image with JSON schema constraint, verify valid structured output."""
        client = openai.Client(
            api_key="sk-123456",
            base_url=f"{self.base_url}/v1",
        )

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "image_description",
                "schema": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string"},
                        "shape": {"type": "string"},
                    },
                    "required": ["color", "shape"],
                },
            },
        }

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content(
                            "Extract the color and shape from this image "
                            "as JSON fields: color and shape."
                        ),
                    ],
                },
            ],
            temperature=0,
            max_tokens=128,
            response_format=schema,
        )

        output = response.choices[0].message.content
        print(f"[P1-006] Raw output:\n{output}\n")

        # The output must be parseable JSON
        self.assertIsNotNone(output, "P1-006: Response is None")
        self.assertGreater(len(output), 0, "P1-006: Response is empty")

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            self.fail(
                f"P1-006: Output is not valid JSON: {e}\n" f"Raw output: {output}"
            )

        # Verify required keys
        self.assertIn(
            "color",
            parsed,
            f"P1-006: Missing 'color' key in parsed JSON: {parsed}",
        )
        self.assertIn(
            "shape",
            parsed,
            f"P1-006: Missing 'shape' key in parsed JSON: {parsed}",
        )

        # Values must be non-empty strings
        self.assertIsInstance(
            parsed["color"],
            str,
            f"P1-006: 'color' should be a string, got {type(parsed['color'])}",
        )
        self.assertIsInstance(
            parsed["shape"],
            str,
            f"P1-006: 'shape' should be a string, got {type(parsed['shape'])}",
        )
        self.assertGreater(
            len(parsed["color"]),
            0,
            f"P1-006: 'color' value is empty",
        )
        self.assertGreater(
            len(parsed["shape"]),
            0,
            f"P1-006: 'shape' value is empty",
        )

        # The extracted information should match the created image
        color_lower = parsed["color"].lower()
        shape_lower = parsed["shape"].lower()
        self.assertEqual(
            color_lower,
            self._expected_color,
            f"P1-006: Extracted color '{parsed['color']}' doesn't match "
            f"expected '{self._expected_color}'",
        )
        self.assertEqual(
            shape_lower,
            self._expected_shape,
            f"P1-006: Extracted shape '{parsed['shape']}' doesn't match "
            f"expected '{self._expected_shape}'",
        )

        # No crash: usage fields present
        self.assertGreater(
            response.usage.prompt_tokens,
            0,
            "P1-006: prompt_tokens should be > 0",
        )

        print(f"  [P1-006] Parsed: color={parsed['color']}, shape={parsed['shape']}")


# ============================================
# P1-007: Tool call + image -> image param in function arguments
# ============================================


class TestP1007ToolCall(CustomTestCase):
    """P1-007: Verify tool calling works with image input.

    Scenario:
        The user provides an image and asks the model to analyse it by calling
        a registered ``analyze_image`` function. The tool_parser is CPU-side
        (platform_agnostic), so risk is low.

    Related features: tool_parser

    NPU note:
        tool_parser is ``platform_agnostic`` (CPU-side text parser). ViT
        encoding is unaffected. Simple, low-risk test.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(17)

    @classmethod
    def setUpClass(cls):
        cls._expected_color = Color.BLUE.name.lower()
        cls._expected_shape = Shape.RECTANGLE.value
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=["--mem-fraction-static", "0.4", "--tool-call-parser", "qwen"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_tool_call_with_image(self):
        """Send image with a registered tool, verify correct tool call format."""
        client = openai.Client(
            api_key="sk-123456",
            base_url=f"{self.base_url}/v1",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "Analyze the content of an image and record a description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the image content",
                            },
                        },
                        "required": ["description"],
                    },
                },
            },
        ]

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content(
                            "Analyze this image and call the analyze_image "
                            "function with a description of what you see."
                        ),
                    ],
                },
            ],
            temperature=0,
            max_tokens=256,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # The model must have produced a tool call
        self.assertIsNotNone(
            message.tool_calls,
            "P1-007: Model did not return any tool calls. "
            f"Response content: '{message.content}'",
        )
        self.assertGreater(
            len(message.tool_calls),
            0,
            "P1-007: Empty tool_calls list",
        )

        tool_call = message.tool_calls[0]
        self.assertEqual(
            tool_call.function.name,
            "analyze_image",
            f"P1-007: Expected function name 'analyze_image', "
            f"got '{tool_call.function.name}'",
        )

        # Arguments must be valid JSON with a "description" field
        self.assertIsNotNone(
            tool_call.function.arguments,
            "P1-007: Tool call arguments are None",
        )
        self.assertGreater(
            len(tool_call.function.arguments),
            0,
            "P1-007: Tool call arguments are empty",
        )

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            self.fail(
                f"P1-007: Tool call arguments are not valid JSON: {e}\n"
                f"Arguments raw: {tool_call.function.arguments}"
            )

        self.assertIn(
            "description",
            args,
            f"P1-007: Missing 'description' in tool call arguments: {args}",
        )
        self.assertIsInstance(
            args["description"],
            str,
            f"P1-007: 'description' should be a string, got {type(args['description'])}",
        )
        self.assertGreater(
            len(args["description"]),
            0,
            "P1-007: 'description' value is empty",
        )

        # The description should reference image content
        assert_text_contains(self, args["description"])

        # Precisely verify the expected color and shape are mentioned
        assert_color_and_shape(
            self,
            args["description"],
            self._expected_color,
            self._expected_shape,
            prefix="P1-007: ",
        )

        # No crash verification
        self.assertGreater(
            response.usage.prompt_tokens,
            0,
            "P1-007: prompt_tokens should be > 0",
        )

        print(
            f"  [P1-007] Tool: {tool_call.function.name}, "
            f"description length: {len(args['description'])}"
        )


# ============================================
# P1-008: Offloading + image -> inference correctness
# ============================================


class TestP1008Offload(CustomTestCase):
    """P1-008: Verify image inference is correct when weights are offloaded to CPU.

    ``--cpu-offload-gb 4`` moves 4 GB of model weights from NPU to CPU memory.
    On each forward pass those weights are temporarily moved back to NPU for
    computation.  This test verifies that the offload hook wrapping does not
    break inference correctness for multimodal input.

    The test sends a request with a long text prefix plus an image.  The long
    prefix naturally triggers chunked prefill, but that is an incidental
    side-effect — the test target is offloading, not chunked prefill.

    Related features: offloading
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(18)

    @classmethod
    def setUpClass(cls):
        cls._expected_color = Color.BLUE.name.lower()
        cls._expected_shape = Shape.RECTANGLE.value
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._long_prefix = _generate_long_prefix_5k()
        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=[
                "--mem-fraction-static",
                "0.4",
                "--chunked-prefill-size",
                "512",
                "--cpu-offload-gb",
                "4",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_offload_image_inference(self):
        """Send image + long prefix request with offloaded weights, verify output."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        text_content(self._long_prefix),
                        image_content(self._image_b64),
                        text_content("Describe the image content in detail"),
                    ],
                },
            ],
            max_tokens=256,
        )

        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        # Expected color and shape must be mentioned
        assert_color_and_shape(
            self,
            text,
            self._expected_color,
            self._expected_shape,
            prefix="P1-008: ",
        )

        print(f"  [P1-008] output_len={len(text)}")


# ============================================
# P1-009: DP-attention + DP LM Head + image -> LM head sharding
# ============================================


class TestP1009DpLmHead(CustomTestCase):
    """P1-009: Verify DP LM head sharding does not affect image token projection.

    Deploy Qwen3-VL-4B with ``--dp-size 2 --enable-dp-attention
    --enable-dp-lm-head``, send an image request, and verify the output
    references visual content.  Also verifies via ``/server_info`` that
    the feature was not silently disabled.

    Related features: dp_attention, hardware_backend
    NPU note: Needs **2 NPU chips** (``--dp-size 2``).
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    _port = get_port(19)

    @classmethod
    def setUpClass(cls):
        cls._expected_color = Color.BLUE.name.lower()
        cls._expected_shape = Shape.RECTANGLE.value
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._messages = [
            {
                "role": "user",
                "content": [
                    image_content(cls._image_b64),
                    text_content("Describe the image"),
                ],
            },
        ]

    def test_dp_lm_head_image(self):
        """Send image request with DP LM Head, verify output and feature gating."""
        process, url = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--mem-fraction-static",
                "0.4",
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
            ],
        )
        try:
            output = chat(url, self._messages, max_tokens=128, seed=42)

            self.assertIsNotNone(
                output,
                "P1-009: DP LM Head returned None",
            )
            self.assertGreater(
                len(output),
                0,
                "P1-009: DP LM Head output is empty",
            )

            # ---- Verify DP LM Head is actually active ----
            server_info = requests.get(url + "/server_info", timeout=10).json()
            self.assertTrue(
                server_info.get("enable_dp_lm_head"),
                "P1-009: enable_dp_lm_head is not True in server info — "
                "DP LM Head was silently disabled",
            )

            # Precisely verify the expected color and shape are mentioned
            assert_color_and_shape(
                self,
                output,
                self._expected_color,
                self._expected_shape,
                prefix="P1-009: ",
            )

        finally:
            kill_process_tree(process.pid)

        print(f"  [P1-009] Output len={len(output)}")


# ============================================
# P1-010: Overlap Schedule + speculative decoding + image
# ============================================


class TestP1010OverlapSchedule(CustomTestCase):
    """P1-010: Verify overlap schedule does not break multimodal draft generation.

    Scenario:
        Deploy Qwen3-VL-8B-Instruct with EAGLE3 speculative decoding +
        ``SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`` +
        ``SGLANG_ENABLE_SPEC_V2=1``, send an image request, and verify
        the output is non-empty and semantically plausible.

    Notes on environment variables:
        * ``SGLANG_ENABLE_OVERLAP_PLAN_STREAM``: only needed for
          EAGLE2/EAGLE3 to enable overlap scheduling.  NEXTN always
          uses overlap by default and is not affected by this variable.
        * ``SGLANG_ENABLE_SPEC_V2``: only takes effect for EAGLE/EAGLE3;
          it has no effect on NEXTN.

    Known limitation:
        Qwen3-VL / Qwen3.5-VL multimodal models currently **do not**
        support EAGLE3 speculative decoding.  The
        ``capture_aux_hidden_states`` path in ``qwen3_vl.py`` is
        incompatible with the multimodal forward pass and causes a
        ``ValueError`` at runtime.
        Only pure-text Qwen3-* models (e.g. Qwen3-8B) can work with
        EAGLE3 at the moment.

    Related features: speculative_decoding

    NPU note:
        Uses Qwen3-VL-8B-Instruct as the target model and
        Qwen3-VL-8B-Instruct-Eagle3 as the draft model.  If the EAGLE3
        draft weights are not available, the test is skipped gracefully.
    """

    _model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    _draft_model = QWEN3_VL_8B_EAGLE3_PATH
    _port = get_port(21)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._messages = [
            {
                "role": "user",
                "content": [
                    image_content(cls._image_b64),
                    text_content("Describe the image"),
                ],
            },
        ]

        env_overlap = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        common_args = [
            "--mem-fraction-static",
            0.6,
            "--cuda-graph-bs",
            1,
            "--disable-radix-cache",
            "--tp-size",
            2,
        ]
        overlap_args = [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls._draft_model,
            "--speculative-num-steps",
            3,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            4,
            "--speculative-draft-model-quantization",
            "unquant",
        ]

        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=common_args + overlap_args,
            env=env_overlap,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_overlap_schedule_with_image(self):
        """Send image request with overlap schedule + EAGLE3 spec decoding."""
        output = chat(self.base_url, self._messages, max_tokens=64)

        self.assertIsNotNone(
            output,
            "P1-010: Overlap schedule output is None",
        )
        self.assertGreater(
            len(output),
            0,
            "P1-010: Overlap schedule output is empty",
        )
        assert_text_contains(self, output)

        print(f"  [P1-010] Output len={len(output)}")


# ============================================
# Entry point
# ============================================

if __name__ == "__main__":
    unittest.main()
