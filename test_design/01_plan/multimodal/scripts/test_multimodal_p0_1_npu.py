"""
NPU multimodal P0 basic test cases (P0-001 through P0-005).

This file implements P0 test cases for multimodal (VLM) interaction testing on NPU
as defined in the multimodal interaction test analysis report.

Implemented test cases:
  - P0-001: Single image + text -> describe image content (Smoke Test)
  - P0-002: Same image twice -> cache hit (Radix Cache prefix caching)
  - P0-003: Concurrent text + image requests -> isolation
  - P0-004: Multi-image -> compare two images
  - P0-005: Variable size images -> different resolutions

All tests run against a single Qwen3-VL-4B-Instruct server on 1 NPU.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from utils import (
    Color,
    Shape,
    assert_color_and_shape,
    chat,
    create_test_image,
    get_port,
    image_content,
    launch_server,
    text_content,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
class TestMultimodalP0Basic(CustomTestCase):
    """P0 multimodal basic test cases for NPU.

    Implements P0-001 through P0-005 from the multimodal interaction test plan.
    A single Qwen3-VL-4B-Instruct server is started once per test class and
    shared across all test methods.
    """

    @classmethod
    def setUpClass(cls):
        """Start the SGLang server for VLM inference."""
        cls.model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
        cls.api_key = "sk-123456"

        cls.process, cls.base_url = launch_server(
            cls.model,
            port=get_port(0),
            extra_args=[
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.25",
                "--tp-size",
                "1",
            ],
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

    def _build_msg(self, image_b64, prompt):
        return [
            {
                "role": "user",
                "content": [image_content(image_b64), text_content(prompt)],
            }
        ]

    # -- test cases --------------------------------------------------------

    def test_001_single_image_smoke(self):
        """P0-001: Single image + text -> describe image content (Smoke Test).

        Sends one test image with a description prompt and checks:
        - Non-empty, semantically relevant output
        """
        _, image_b64 = create_test_image(640, 480, color=Color.RED, shape=Shape.ELLIPSE)

        output = chat(
            self.base_url,
            self._build_msg(image_b64, "Please describe this image"),
            max_tokens=128,
        )

        self.assertTrue(
            output,
            "P0-001: Model returned empty response for single-image description",
        )
        self.assertGreater(
            len(output),
            5,
            f"P0-001: Output suspiciously short: '{output}'",
        )

        assert_color_and_shape(self, output, "red", "ellipse", prefix="P0-001: ")

        print(f"  [P0-001] output_len={len(output)}")

    def test_002_same_image_cache_hit(self):
        """P0-002: Same image twice -> cache hit.

        Sends two requests with an identical image but different text prompts.
        The second request should benefit from Radix Cache prefix matching
        on the image-token prefix, resulting in a lower TTFT.
        """
        _, image_b64 = create_test_image(
            640, 480, color=Color.BLUE, shape=Shape.ELLIPSE
        )

        # --- Request 1 (cache miss) ---
        output1 = chat(
            self.base_url,
            self._build_msg(image_b64, "Describe the image"),
            max_tokens=32,
        )
        self.assertTrue(
            output1,
            "P0-002: First request returned empty response",
        )

        # --- Request 2 (same image, different text -> prefix cache hit) ---
        output2 = chat(
            self.base_url,
            self._build_msg(
                image_b64, "Describe the shape and color of the object in the image"
            ),
            max_tokens=32,
        )
        self.assertTrue(
            output2,
            "P0-002: Second request returned empty response",
        )

        assert_color_and_shape(self, output1, "blue", "ellipse", prefix="P0-002/req1: ")
        assert_color_and_shape(self, output2, "blue", "ellipse", prefix="P0-002/req2: ")

        print(f"  [P0-002] output1_len={len(output1)}  output2_len={len(output2)}")

    def test_003_concurrent_text_image_isolation(self):
        """P0-003: Concurrent text + image requests -> isolation.

        Sends 10 text-only "hello" requests and 5 image-description requests
        concurrently. Verifies that all 15 complete successfully and that
        text-only outputs are not polluted by image-token artefacts.
        """
        text_prompt = "hello"
        _, image_b64 = create_test_image(
            320, 240, color=Color.GREEN, shape=Shape.ELLIPSE
        )

        results = []

        def _send_text(idx):
            try:
                resp = self.client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": text_prompt}],
                    temperature=0,
                    max_tokens=16,
                )
                return ("text", idx, resp.choices[0].message.content)
            except Exception as e:
                return ("text", idx, f"__ERROR__:{e}")

        def _send_image(idx):
            try:
                resp = self.client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                image_content(image_b64),
                                text_content("Describe the image"),
                            ],
                        }
                    ],
                    temperature=0,
                    max_tokens=32,
                )
                return ("image", idx, resp.choices[0].message.content)
            except Exception as e:
                return ("image", idx, f"__ERROR__:{e}")

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for i in range(10):
                futures.append(executor.submit(_send_text, i))
            for i in range(5):
                futures.append(executor.submit(_send_image, i))

            for future in as_completed(futures):
                results.append(future.result())

        error_results = [
            r for r in results if isinstance(r[2], str) and r[2].startswith("__ERROR__")
        ]

        self.assertEqual(
            len(error_results),
            0,
            f"P0-003: {len(error_results)} request(s) failed: {error_results[:5]}",
        )

        # Verify counts
        text_results = [r for r in results if r[0] == "text"]
        image_results = [r for r in results if r[0] == "image"]
        self.assertEqual(
            len(text_results),
            10,
            f"P0-003: Expected 10 text results, got {len(text_results)}",
        )
        self.assertEqual(
            len(image_results),
            5,
            f"P0-003: Expected 5 image results, got {len(image_results)}",
        )

        # Every text response must be a short, plausible text string (not
        # polluted by image-token artefacts or base64 garbage).
        for req_type, idx, content in results:
            self.assertIsInstance(
                content, str, f"P0-003: {req_type}[{idx}] content is not a string"
            )
            if req_type == "text":
                # "hello" usually yields "Hello!" or a short greeting
                self.assertGreater(
                    len(content),
                    0,
                    f"P0-003: text[{idx}] is empty",
                )
                # Check it looks like natural language, not base64 garbage
                self.assertFalse(
                    content.startswith("data:image"),
                    f"P0-003: text[{idx}] appears contaminated with image data: '{content[:50]}'",
                )
                # Simple heuristic: text-only responses should be short
                self.assertLess(
                    len(content),
                    200,
                    f"P0-003: text[{idx}] suspiciously long ({len(content)} chars): '{content[:80]}'",
                )
            elif req_type == "image":
                self.assertGreater(
                    len(content),
                    0,
                    f"P0-003: image[{idx}] is empty",
                )

        print(
            f"  [P0-003] {len(text_results)} text + {len(image_results)} image "
            "all succeeded — no cross-pollution detected"
        )

    def test_004_multi_image_compare(self):
        """P0-004: Multi-image -> compare two images.

        Sends two different images with a comparison prompt. Verifies that
        the model produces a coherent output referencing both images, and
        that no OOM / crash occurs.
        """
        # Two visually distinct images
        _, img1_b64 = create_test_image(320, 240, color=Color.RED, shape=Shape.ELLIPSE)
        _, img2_b64 = create_test_image(320, 240, color=Color.BLUE, shape=Shape.ELLIPSE)

        # Multi-image request: two image_url blocks + comparison prompt
        response = self.client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(img1_b64),
                        image_content(img2_b64),
                        text_content("Please compare these two images"),
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
        )

        output = response.choices[0].message.content

        self.assertTrue(
            output,
            "P0-004: Model returned empty response for multi-image comparison",
        )
        self.assertGreaterEqual(
            len(output),
            10,
            f"P0-004: Output too short for a meaningful comparison: '{output}'",
        )

        # The response should reference both images
        assert_color_and_shape(self, output, "red", "ellipse", prefix="P0-004: ")
        assert_color_and_shape(self, output, "blue", "ellipse", prefix="P0-004: ")

        # Basic usage fields should be present
        self.assertGreater(
            response.usage.prompt_tokens,
            0,
            "P0-004: prompt_tokens should be > 0 for multi-image request",
        )
        self.assertGreater(
            response.usage.completion_tokens,
            0,
            "P0-004: completion_tokens should be > 0",
        )

        print(
            f"  [P0-004] output_len={len(output)}  prompt_tokens="
            f"{response.usage.prompt_tokens}"
        )

    def test_005_variable_size_images(self):
        """P0-005: Variable size images -> different resolutions.

        Verifies two distinct image processing paths from SGLang's own
        smart_resize() in srt/multimodal/processors/qwen_vl.py:

          MIN_PIXELS = 4*28*28 = 3136        → below triggers upscale
          MAX_PIXELS = 16384*28*28 ≈ 12.8M   → above triggers downscale
          (MAX_PIXELS is configurable via SGLANG_IMAGE_MAX_PIXELS env var)

        Test images:
          32x32 (1024 px)   → below MIN_PIXELS → upscale path
          1920x1080 (2.0M px) → within normal range (below 12.8M MAX_PIXELS)
                                  → does NOT trigger downscale, but stresses
                                  ViT memory allocation (large patch count)

        Note: --disable-cuda-graph is used; graph-compilation on shape change
        is tested separately in P0-007.
        """
        sizes = [
            (32, 32, Color.PURPLE, "below_min"),  # 1024 px → upscale
            (1920, 1080, Color.TEAL, "large"),  # 2M px → memory stress, not downscale
        ]

        for width, height, color, label in sizes:
            with self.subTest(size=f"{width}x{height}"):
                _, img_b64 = create_test_image(
                    width, height, color=color, shape=Shape.ELLIPSE
                )

                output = chat(
                    self.base_url,
                    self._build_msg(
                        img_b64,
                        "Describe the shape and color of the object in the image",
                    ),
                    max_tokens=64,
                )

                # All sizes: must produce non-empty output
                self.assertTrue(
                    output,
                    f"P0-005: Empty output for {label} image ({width}x{height})",
                )

                # All sizes: must reference image content (not gibberish)
                assert_color_and_shape(
                    self,
                    output,
                    color.name.lower(),
                    "ellipse",
                    prefix=f"P0-005/{label}: ",
                )

                print(f"  [P0-005/{label} {width}x{height}] output_len={len(output)}")


if __name__ == "__main__":
    unittest.main()
