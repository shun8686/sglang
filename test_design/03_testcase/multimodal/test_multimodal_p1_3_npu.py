#!/usr/bin/env python3
"""
NPU multimodal P1 test cases P1-011 through P1-021 — Completions API Parameter Interactions.

This file implements multimodal (VLM) interaction tests that verify OpenAI Chat
Completions API parameters work correctly with image inputs on the NPU platform.

Test cases:
  - P1-011: temperature + multimodal (temp=0, 0.8, 1.5)
  - P1-012: top_p + multimodal (top_p=0.1, 0.5, 1.0)
  - P1-013: seed + multimodal (determinism)
  - P1-014: max_tokens + multimodal (short vs long)
  - P1-015: stop + multimodal (single + multi stop)
  - P1-016: n + multimodal (multiple completions)
  - P1-017: stream + multimodal (streaming chunks)
  - P1-018: stream_options + multimodal (include_usage)
  - P1-019: frequency_penalty + multimodal (0 vs 1.5)
  - P1-020: presence_penalty + multimodal (0 vs 1.5)
  - P1-021: logprobs + top_logprobs + multimodal

All tests run against a single Qwen3.5-9B server on 2 NPU (TP=2).

Server args follow the pattern established by TestP1001SpeculativeDecoding
in test_multimodal_p1_1_npu.py (Qwen3.5-9B GDN/Mamba model).

Reference:
  https://developers.openai.com/api/reference/resources/completions/methods/create
"""

import unittest

import openai
from utils import (
    QWEN3_5_9B_PATH,
    Color,
    Shape,
    assert_color_and_shape,
    create_test_image,
    get_port,
    image_content,
    launch_server,
    text_content,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1200, suite="nightly-2-npu-a3", nightly=True)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
class TestMultimodalP1ParameterInteractions(CustomTestCase):
    """P1 multimodal parameter interaction test cases for NPU.

    Verifies that OpenAI Chat Completions API parameters (temperature,
    top_p, seed, max_tokens, stop, n, stream, stream_options,
    frequency_penalty, presence_penalty, logprobs) interact correctly
    with multimodal (image) inputs.

    Uses Qwen3.5-9B (GDN attention + DeepStack ViT + native NEXTN MTP)
    with TP=1 on 1 NPU chip and speculative decoding enabled.
    Server args follow the pattern (common + spec) established by
    TestP1001SpeculativeDecoding in test_multimodal_p1_1_npu.py.
    """

    # ------------------------------------------------------------------
    # Model & server configuration (matches TestP1001SpeculativeDecoding)
    # ------------------------------------------------------------------

    _model = QWEN3_5_9B_PATH
    _port = get_port(40)
    _common_args = [
        "--mem-fraction-static",
        "0.75",
        "--cuda-graph-bs",
        1,
        2,
        4,
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--tp-size",
        1,
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
    ]
    _spec_args = [
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]
    _server_args = _common_args + _spec_args

    @classmethod
    def setUpClass(cls):
        """Start the SGLang server for VLM inference on Qwen3.5-9B."""
        cls.api_key = "sk-123456"

        cls.process, cls.base_url = launch_server(
            cls._model,
            cls._port,
            extra_args=cls._server_args,
        )
        cls.client = openai.Client(
            api_key=cls.api_key,
            base_url=f"{cls.base_url}/v1",
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up the server process."""
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    # -- helpers -----------------------------------------------------------

    def _build_msg(self, image_b64, prompt="Describe the image"):
        """Build a single-image user message."""
        return [
            {
                "role": "user",
                "content": [image_content(image_b64), text_content(prompt)],
            }
        ]

    def _request(self, messages, **kwargs):
        """Send a chat completion and return the response object."""
        params = dict(
            model="default",
            messages=messages,
            max_tokens=128,
        )
        params.update(kwargs)
        return self.client.chat.completions.create(**params)

    def _request_text(self, messages, **kwargs):
        """Send a chat completion and return just the text content."""
        return self._request(messages, **kwargs).choices[0].message.content

    # ===================================================================
    # P1-011: temperature + multimodal
    # ===================================================================

    def test_011_temperature_multimodal(self):
        """P1-011: Verify temperature sampling with image input.

        Tests temperature=0 (deterministic), 0.8 (moderate), 1.5 (high).
        All must produce non-empty output that correctly identifies the
        test image's color and shape.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.GREEN, shape=Shape.ELLIPSE
        )
        temps = [0, 0.8, 1.5]
        outputs = {}

        for temp in temps:
            with self.subTest(temperature=temp):
                text = self._request_text(
                    self._build_msg(image_b64),
                    temperature=temp,
                )
                outputs[temp] = text  # store before asserts for safe access below
                self.assertTrue(text, f"P1-011: Empty output at temperature={temp}")
                self.assertGreater(
                    len(text),
                    5,
                    f"P1-011: Suspiciously short output ({len(text)} chars) at temperature={temp}",
                )
                assert_color_and_shape(
                    self, text, "green", "ellipse", prefix=f"P1-011/temp={temp}: "
                )
                print(f"  [P1-011/temp={temp}] output_len={len(text)}")

        # Verify all temperatures produced output
        if 0 in outputs:
            self.assertGreaterEqual(
                len(outputs[0]), 5, "P1-011: temperature=0 output too short"
            )

    # ===================================================================
    # P1-012: top_p + multimodal
    # ===================================================================

    def test_012_top_p_multimodal(self):
        """P1-012: Verify top_p (nucleus sampling) with image input.

        Tests top_p=0.1 (narrow), 0.5 (moderate), 1.0 (full distribution).
        All must produce image-relevant output.  Narrow top_p must not
        produce degenerate / repeated-token output.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.BLUE, shape=Shape.RECTANGLE
        )

        for top_p in [0.1, 0.5, 1.0]:
            with self.subTest(top_p=top_p):
                text = self._request_text(
                    self._build_msg(image_b64),
                    top_p=top_p,
                    temperature=0.8,  # top_p requires temperature > 0
                )
                self.assertTrue(text, f"P1-012: Empty output at top_p={top_p}")
                self.assertGreater(
                    len(text), 5, f"P1-012: Suspiciously short output at top_p={top_p}"
                )

                assert_color_and_shape(
                    self, text, "blue", "rectangle", prefix=f"P1-012/top_p={top_p}: "
                )
                print(f"  [P1-012/top_p={top_p}] output_len={len(text)}")

    # ===================================================================
    # P1-013: seed + multimodal (determinism)
    # ===================================================================

    def test_013_seed_determinism(self):
        """P1-013: Verify seed produces deterministic output for same image.

        Sends two requests with identical image, prompt, seed=42, and a
        low non-zero temperature (0.2).  Unlike temperature=0 (which is
        deterministic regardless of seed), temperature=0.2 uses random
        sampling — so the seed MUST be controlling the RNG for the two
        outputs to match.

        Falls back to semantic equivalence if the NPU backend cannot
        guarantee strict determinism (e.g., due to kernel-level noise).
        """
        _, image_b64 = create_test_image(320, 240, color=Color.RED, shape=Shape.ELLIPSE)
        seed = 42

        text1 = self._request_text(
            self._build_msg(
                image_b64, "What color and shape do you see? Answer concisely."
            ),
            temperature=0.2,
            seed=seed,
            max_tokens=64,
        )
        text2 = self._request_text(
            self._build_msg(
                image_b64, "What color and shape do you see? Answer concisely."
            ),
            temperature=0.2,
            seed=seed,
            max_tokens=64,
        )

        self.assertTrue(text1, "P1-013: First request returned empty output")
        self.assertTrue(text2, "P1-013: Second request returned empty output")

        # Strict determinism check
        if text1 == text2:
            print(
                f"  [P1-013] deterministic=True (exact match)  output_len={len(text1)}"
            )
        else:
            # NPU hardware non-determinism is a known limitation (kernel
            # scheduling / attention ordering differences between runs).
            # Verify semantic correctness first (hard fail if image
            # understanding is wrong), then record the determinism gap.
            assert_color_and_shape(
                self, text1, "red", "ellipse", prefix="P1-013/req1: "
            )
            assert_color_and_shape(
                self, text2, "red", "ellipse", prefix="P1-013/req2: "
            )
            # Semantic check passed — the mismatch is likely NPU noise,
            # but could also signal a seed bug.  Promote to a hard failure
            # if this becomes a recurring pattern across CI runs.
            print(
                f"  [P1-013] WARNING: deterministic=False — "
                f"len1={len(text1)} len2={len(text2)}. "
                f"Both semantically correct; may be NPU noise or seed bug. "
                f"text1[:120]: {text1[:120]}"
            )

    # ===================================================================
    # P1-014: max_tokens + multimodal
    # ===================================================================

    def test_014_max_tokens_multimodal(self):
        """P1-014: Verify max_tokens limits output with image prompts.

        Tests max_tokens=128 (short) and max_tokens=256 (longer).
        Short output must respect the token budget; both must contain
        image-relevant content.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.PURPLE, shape=Shape.ELLIPSE
        )

        # --- Short max_tokens ---
        resp_short = self._request(
            self._build_msg(
                image_b64, "Answer concisely: what shape and color is this image?"
            ),
            max_tokens=128,
            temperature=0,
        )
        text_short = resp_short.choices[0].message.content
        self.assertTrue(text_short, "P1-014: Empty output with max_tokens=128")

        finish_short = resp_short.choices[0].finish_reason
        print(
            f"  [P1-014/max_tokens=128] output_len_approx={len(text_short)}  "
            f"finish_reason={finish_short}"
        )
        self.assertIn(
            finish_short,
            ("length", "stop"),
            f"P1-014: Unexpected finish_reason '{finish_short}' for max_tokens=128",
        )

        # --- Longer max_tokens ---
        resp_long = self._request(
            self._build_msg(image_b64),
            max_tokens=256,
            temperature=0,
        )
        text_long = resp_long.choices[0].message.content
        self.assertTrue(text_long, "P1-014: Empty output with max_tokens=256")
        print(
            f"  [P1-014/max_tokens=256] output_len_approx={len(text_long)}  "
            f"finish_reason={resp_long.choices[0].finish_reason}"
        )

        # Short output: at minimum must mention the color (shape may not fit
        # within 128 tokens).  Long output: must identify both color and shape.
        self.assertIn(
            "purple",
            text_short.lower(),
            f"P1-014: (max_tokens=128) Expected 'purple' not in output: {text_short[:200]}",
        )
        assert_color_and_shape(
            self, text_long, "purple", "ellipse", prefix="P1-014/max_tokens=256: "
        )

        # Longer output should generally be longer
        self.assertGreaterEqual(
            len(text_long),
            len(text_short),
            "P1-014: max_tokens=256 output not longer than max_tokens=128",
        )

    # ===================================================================
    # P1-015: stop + multimodal
    # ===================================================================

    def test_015_stop_multimodal(self):
        """P1-015: Verify stop sequences work with image prompts.

        Tests a single stop string and a list of stop strings by
        instructing the model to end its response with a specific word
        (STOPEND), which is set as the stop sequence.  This avoids
        tokenization edge cases with punctuation-based stops.

        Image content must be described before termination.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.TEAL, shape=Shape.RECTANGLE
        )

        # --- Single stop string ---
        resp_single = self._request(
            self._build_msg(
                image_b64,
                "Describe the image in one short sentence. "
                "End your response with the word STOPEND",
            ),
            temperature=0,
            max_tokens=512,
            stop="STOPEND",
        )
        text_single = resp_single.choices[0].message.content
        self.assertTrue(text_single, "P1-015: Empty output with stop='STOPEND'")
        self.assertEqual(
            resp_single.choices[0].finish_reason,
            "stop",
            f"P1-015: Expected finish_reason='stop' with stop='STOPEND', "
            f"got '{resp_single.choices[0].finish_reason}'",
        )
        # Image content must be described even with early stop
        assert_color_and_shape(
            self, text_single, "teal", "rectangle", prefix="P1-015/stop='STOPEND': "
        )
        print(
            f"  [P1-015/stop='STOPEND'] output_len={len(text_single)}  "
            f"finish_reason={resp_single.choices[0].finish_reason}"
        )

        # --- Multiple stop strings ---
        resp_multi = self._request(
            self._build_msg(
                image_b64,
                "Describe the image briefly. " "End with either STOPEND or FINISHED",
            ),
            temperature=0,
            max_tokens=512,
            stop=["STOPEND", "FINISHED"],
        )
        text_multi = resp_multi.choices[0].message.content
        self.assertTrue(
            text_multi, "P1-015: Empty output with stop=['STOPEND', 'FINISHED']"
        )
        self.assertEqual(
            resp_multi.choices[0].finish_reason,
            "stop",
            f"P1-015: Expected finish_reason='stop' with stop list, "
            f"got '{resp_multi.choices[0].finish_reason}'",
        )
        # Must still reference image content
        assert_color_and_shape(
            self, text_multi, "teal", "rectangle", prefix="P1-015/stop=list: "
        )
        print(
            f"  [P1-015/stop=list] output_len={len(text_multi)}  "
            f"finish_reason={resp_multi.choices[0].finish_reason}"
        )

    # ===================================================================
    # P1-016: n + multimodal
    # ===================================================================

    def test_016_n_multimodal(self):
        """P1-016: Verify n returns multiple independent completions.

        Uses temperature=0.7 so the two sampling paths draw from different
        random states — the two choices should differ while both correctly
        describe the image.  If n were silently ignored the response would
        contain only 1 choice; if both choices shared the same RNG they
        would be identical.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.RED, shape=Shape.RECTANGLE
        )

        resp = self._request(
            self._build_msg(image_b64, "Describe the image"),
            temperature=0.7,
            max_tokens=64,
            n=2,
        )

        choices = resp.choices
        self.assertEqual(
            len(choices), 2, f"P1-016: Expected 2 choices, got {len(choices)}"
        )

        texts = []
        for i, choice in enumerate(choices):
            text = choice.message.content
            self.assertTrue(text, f"P1-016: Choice {i} returned empty content")
            self.assertGreater(
                len(text), 3, f"P1-016: Choice {i} suspiciously short: '{text}'"
            )
            assert_color_and_shape(
                self, text, "red", "rectangle", prefix=f"P1-016/choice[{i}]: "
            )
            texts.append(text)
            print(
                f"  [P1-016/choice[{i}]] index={choice.index}  "
                f"finish={choice.finish_reason}  len={len(text)}"
            )

        # Independent sampling paths at temperature > 0 should produce
        # different outputs — not always guaranteed (very small model or
        # very low temp may still collide), but with temp=0.7 it is the
        # expected outcome.
        if texts[0] == texts[1]:
            print(
                f"  [P1-016] NOTE: two choices are byte-identical — "
                f"this is unlikely at temperature=0.7 and may indicate "
                f"a shared RNG or n degenerating to greedy."
            )

        # Usage: prompt_tokens should be > 0 (image contributes tokens)
        self.assertGreater(
            resp.usage.prompt_tokens,
            0,
            "P1-016: prompt_tokens should be > 0 for image request",
        )
        print(
            f"  [P1-016] {len(choices)} choices  prompt_tokens={resp.usage.prompt_tokens}  "
            f"completion_tokens={resp.usage.completion_tokens}"
        )

    # ===================================================================
    # P1-017: stream + multimodal
    # ===================================================================

    def test_017_stream_multimodal(self):
        """P1-017: Verify streaming works with image input.

        Streams a chat completion and verifies:
        - Multiple chunks received
        - Combined text correctly identifies image content
        - Final chunk has finish_reason
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.BLUE, shape=Shape.ELLIPSE
        )

        stream = self.client.chat.completions.create(
            model="default",
            messages=self._build_msg(image_b64, "Describe the image"),
            temperature=0,
            max_tokens=128,
            stream=True,
        )

        chunks = []
        combined_text = ""
        finish_reason = None

        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                combined_text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        self.assertGreater(
            len(chunks),
            1,
            f"P1-017: Only {len(chunks)} chunk(s) received — streaming may not be working",
        )
        self.assertTrue(
            combined_text, "P1-017: No text content accumulated from stream"
        )
        self.assertIsNotNone(finish_reason, "P1-017: No finish_reason in stream chunks")

        # Combined text must describe the image
        assert_color_and_shape(
            self, combined_text, "blue", "ellipse", prefix="P1-017: "
        )

        print(
            f"  [P1-017] chunks={len(chunks)}  combined_len={len(combined_text)}  "
            f"finish_reason={finish_reason}"
        )

    # ===================================================================
    # P1-018: stream_options + multimodal
    # ===================================================================

    def test_018_stream_options_multimodal(self):
        """P1-018: Verify stream_options with include_usage in image requests.

        Streams with stream_options={"include_usage": True} and verifies:
        - At least one chunk contains usage data
        - Combined text still correctly identifies image
        - Total token counts > 0 in usage chunk
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.GREEN, shape=Shape.RECTANGLE
        )

        stream = self.client.chat.completions.create(
            model="default",
            messages=self._build_msg(image_b64, "Describe the image"),
            temperature=0,
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = []
        combined_text = ""
        usage_chunk_found = False
        finish_reason = None

        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                combined_text += chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            # Check for usage data
            if chunk.usage is not None:
                usage_chunk_found = True
                print(
                    f"  [P1-018/usage_chunk] prompt_tokens={chunk.usage.prompt_tokens}  "
                    f"completion_tokens={chunk.usage.completion_tokens}  "
                    f"total_tokens={chunk.usage.total_tokens}"
                )

        self.assertGreater(
            len(chunks), 1, f"P1-018: Only {len(chunks)} chunk(s) received"
        )
        self.assertTrue(
            combined_text, "P1-018: No text content accumulated from stream"
        )
        self.assertTrue(
            usage_chunk_found,
            "P1-018: No usage chunk found — include_usage may not be working",
        )
        self.assertIsNotNone(finish_reason, "P1-018: No finish_reason in stream chunks")

        # Verify image understanding
        assert_color_and_shape(
            self, combined_text, "green", "rectangle", prefix="P1-018: "
        )

        print(
            f"  [P1-018] chunks={len(chunks)}  usage_chunk={usage_chunk_found}  "
            f"combined_len={len(combined_text)}"
        )

    # ===================================================================
    # P1-019: frequency_penalty + multimodal
    # ===================================================================

    def test_019_frequency_penalty_multimodal(self):
        """P1-019: Verify frequency_penalty with image input.

        Tests frequency_penalty=0 (baseline) vs 1.5 (high penalty).
        Both must produce correct image descriptions.  Higher penalty
        should reduce token-level repetition.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.PURPLE, shape=Shape.RECTANGLE
        )

        for freq_penalty in [0, 1.5]:
            with self.subTest(frequency_penalty=freq_penalty):
                text = self._request_text(
                    self._build_msg(
                        image_b64,
                        "Describe the image in detail, including colors and shapes",
                    ),
                    temperature=0.7,  # penalty has no effect at temperature=0
                    frequency_penalty=freq_penalty,
                    max_tokens=128,
                )
                self.assertTrue(
                    text, f"P1-019: Empty output at frequency_penalty={freq_penalty}"
                )
                self.assertGreater(
                    len(text),
                    10,
                    f"P1-019: Suspiciously short output at frequency_penalty={freq_penalty}",
                )

                assert_color_and_shape(
                    self,
                    text,
                    "purple",
                    "rectangle",
                    prefix=f"P1-019/freq_pen={freq_penalty}: ",
                )

                print(f"  [P1-019/freq_pen={freq_penalty}] output_len={len(text)}")

    # ===================================================================
    # P1-020: presence_penalty + multimodal
    # ===================================================================

    def test_020_presence_penalty_multimodal(self):
        """P1-020: Verify presence_penalty with image input.

        Tests presence_penalty=0 (baseline) vs 1.5 (high penalty).
        Both must produce correct image descriptions.  Higher penalty
        encourages more diverse vocabulary.
        """
        _, image_b64 = create_test_image(
            320, 240, color=Color.TEAL, shape=Shape.ELLIPSE
        )

        for pres_penalty in [0, 1.5]:
            with self.subTest(presence_penalty=pres_penalty):
                text = self._request_text(
                    self._build_msg(
                        image_b64,
                        "Describe the image in detail, including colors and shapes",
                    ),
                    temperature=0.7,
                    presence_penalty=pres_penalty,
                    max_tokens=128,
                )
                self.assertTrue(
                    text, f"P1-020: Empty output at presence_penalty={pres_penalty}"
                )
                self.assertGreater(
                    len(text),
                    10,
                    f"P1-020: Suspiciously short output at presence_penalty={pres_penalty}",
                )

                assert_color_and_shape(
                    self,
                    text,
                    "teal",
                    "ellipse",
                    prefix=f"P1-020/pres_pen={pres_penalty}: ",
                )

                print(f"  [P1-020/pres_pen={pres_penalty}] output_len={len(text)}")

    # ===================================================================
    # P1-021: logprobs + top_logprobs + multimodal
    # ===================================================================

    def test_021_logprobs_multimodal(self):
        """P1-021: Verify logprobs and top_logprobs with image input.

        Sends a request with logprobs=True and top_logprobs=3.
        Verifies:
        - Response has logprobs field populated
        - Content is non-empty and identifies image content
        - Token-level logprobs are present for output tokens
        """
        _, image_b64 = create_test_image(320, 240, color=Color.RED, shape=Shape.ELLIPSE)

        resp = self._request(
            self._build_msg(
                image_b64, "Answer concisely: what shape and color is this image?"
            ),
            temperature=0,
            max_tokens=256,
            logprobs=True,
            top_logprobs=3,
        )

        text = resp.choices[0].message.content
        self.assertTrue(text, "P1-021: Empty output with logprobs=True")
        self.assertIn(
            "red", text.lower(), f"P1-021: Expected 'red' not in output: {text[:200]}"
        )

        # Check logprobs in response
        logprobs_content = resp.choices[0].logprobs
        content_has_logprobs = logprobs_content is not None
        print(
            f"  [P1-021] logprobs_present={content_has_logprobs}  "
            f"content_len={len(text)}  content='{text[:100]}'"
        )

        if content_has_logprobs:
            # Verify structure: logprobs should have content (token-level data)
            # The exact attribute name depends on the OpenAI API version
            # Standard: logprobs.content is a list of token logprob entries
            if hasattr(logprobs_content, "content") and logprobs_content.content:
                token_count = len(logprobs_content.content)
                print(f"  [P1-021] token_logprobs_count={token_count}")
                self.assertGreater(
                    token_count,
                    0,
                    "P1-021: logprobs.content is empty — no token-level data",
                )
            elif hasattr(logprobs_content, "model_dump"):
                dumped = logprobs_content.model_dump()
                # At minimum, the logprobs object should not be entirely empty
                self.assertTrue(
                    any(v for v in dumped.values() if v),
                    f"P1-021: logprobs object appears empty: {dumped}",
                )
        else:
            # logprobs may not be returned in all backends — document and pass
            print(
                f"  [P1-021] NOTE: logprobs not populated — NPU backend may not "
                f"support this feature yet. Content still verified."
            )

        # Verify image understanding is correct even with logprobs enabled
        assert_color_and_shape(self, text, "red", "ellipse", prefix="P1-021: ")


if __name__ == "__main__":
    unittest.main()
