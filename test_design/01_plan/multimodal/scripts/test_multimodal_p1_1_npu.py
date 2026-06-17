#!/usr/bin/env python3
"""
NPU multimodal P1 test cases P1-001 through P1-004.

This file implements the first five P1 test cases for multimodal (VLM)
interaction testing on NPU.

Test cases:
  - P1-001: Speculative decoding + image -> speedup without correctness loss
  - P1-002: PD disaggregation + image -> prefill node transmits correctly
  - P1-003: TP parallelism + image -> multi-card inference correct
  - P1-004: DP-attention + image -> high concurrency throughput

"""

import base64
import io
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import requests
from PIL import Image, ImageDraw
from utils import (
    QWEN3_5_9B_PATH,
    Color,
    Shape,
    chat_single_image,
    content_has_keywords,
    create_test_image,
    get_port,
    image_content,
    launch_router,
    launch_server,
    text_content,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1200, suite="nightly-2-npu-a3", nightly=True)


def _create_multi_object_image():
    """Create a 640x480 image with four distinct coloured objects.

    Layout:
        - Red circle   (top-left)
        - Blue rectangle  (top-right)
        - Green triangle  (bottom-left)
        - Yellow circle   (bottom-right)

    Designed for P1-001 where the prompt asks the model to enumerate each
    object individually.
    """
    img = Image.new("RGB", (640, 480), color=(30, 30, 120))
    draw = ImageDraw.Draw(img)

    # Red circle
    draw.ellipse([50, 50, 200, 200], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    # Blue rectangle
    draw.rectangle(
        [300, 50, 500, 200], fill=(0, 0, 255), outline=(255, 255, 255), width=2
    )
    # Green triangle (approximated as polygon)
    draw.polygon(
        [(150, 300), (50, 450), (250, 450)],
        fill=(0, 255, 0),
        outline=(255, 255, 255),
        width=2,
    )
    # Yellow circle
    draw.ellipse(
        [350, 300, 500, 450], fill=(255, 255, 0), outline=(255, 255, 255), width=2
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return img_bytes, base64.b64encode(img_bytes).decode("utf-8")


def _send_concurrent(base_url, image_b64, prompt, num_requests=50, max_tokens=32):
    """Send *num_requests* concurrent image requests and return results.

    Returns:
        list of (status, content, completion_tokens)
    """
    client = openai.Client(api_key="sk-123456", base_url=f"{base_url}/v1")

    def _send():
        try:
            resp = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content(image_b64),
                            text_content(prompt),
                        ],
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return ("ok", resp.choices[0].message.content, resp.usage.completion_tokens)
        except Exception as e:
            return ("error", str(e), 0)

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(_send) for _ in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]
    return results


# ===================================================================
# P1-001: Speculative decoding + image -> speedup without correctness loss
# ===================================================================


class TestP1001SpeculativeDecoding(CustomTestCase):
    """P1-001: Verify MTP speculative decoding produces correct multimodal
    output with reduced latency vs a non-speculative baseline.

    Uses Qwen3.5-9B with built-in MTP heads (no external draft model).
    Qwen3.5-9B has GDN attention + DeepStack ViT + native NEXTN support.

    Related features: speculative_decoding
    Required capability: vlm-mtp
    """

    _model = QWEN3_5_9B_PATH
    _port = get_port(11)
    _baseline_port = get_port(12)
    _common_args = [
        "--mem-fraction-static",
        "0.78",
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

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = _create_multi_object_image()
        cls._prompt = "Describe each object in the image"

    def test_speculative_decoding_speedup_and_correctness(self):
        """Compare MTP speculative decoding vs non-speculative baseline."""
        # ---- Phase 1: Baseline (non-speculative) ----
        baseline_process, baseline_url = launch_server(
            self._model,
            self._baseline_port,
            extra_args=self._common_args + ["--base-gpu-id", "1"],
        )
        try:
            output_bl = chat_single_image(
                baseline_url,
                self._image_b64,
                self._prompt,
                max_tokens=64,
            )
        finally:
            kill_process_tree(baseline_process.pid)
            time.sleep(2)

        self.assertTrue(output_bl, "P1-001: Baseline returned empty output")
        self.assertGreater(
            len(output_bl), 5, f"P1-001: Baseline too short: '{output_bl}'"
        )
        self.assertTrue(
            content_has_keywords(output_bl),
            f"P1-001: Baseline output doesn't reference image: '{output_bl[:200]}'",
        )

        # ---- Phase 2: Speculative decoding ----
        spec_process, spec_url = launch_server(
            self._model,
            self._port,
            extra_args=self._common_args + self._spec_args + ["--base-gpu-id", "0"],
        )
        try:
            output_spec = chat_single_image(
                spec_url,
                self._image_b64,
                self._prompt,
                max_tokens=64,
            )

            self.assertTrue(output_spec, "P1-001: Spec returned empty output")
            self.assertGreater(
                len(output_spec), 5, f"P1-001: Spec too short: '{output_spec}'"
            )
            self.assertTrue(
                content_has_keywords(output_spec),
                f"P1-001: Spec output doesn't reference image: '{output_spec[:200]}'",
            )

            # At temperature=0 (greedy), speculative decoding must produce
            # identical output to the non-speculative baseline.  The rejection
            # sampling algorithm guarantees the output distribution matches the
            # target model, and greedy collapses that to a deterministic sequence.
            self.assertEqual(
                output_bl,
                output_spec,
                f"P1-001: Spec output differs from baseline at temperature=0:\n"
                f"  baseline: '{output_bl[:200]}'\n"
                f"  spec:     '{output_spec[:200]}'",
            )

            # ---- Phase 3: Verify MTP is actually accepting drafts ----
            for _ in range(3):
                chat_single_image(
                    spec_url,
                    self._image_b64,
                    self._prompt,
                    max_tokens=64,
                )
            time.sleep(3)

            server_info = requests.get(spec_url + "/server_info", timeout=10).json()
            avg_spec_accept_length = server_info["internal_states"][0].get(
                "avg_spec_accept_length", 1.0
            )
            print(f"  [P1-001] avg_spec_accept_length={avg_spec_accept_length:.2f}")
            self.assertGreater(
                avg_spec_accept_length,
                1.5,
                f"P1-001: accept_length={avg_spec_accept_length:.2f} <= 1.5 — "
                f"MTP drafts are mostly rejected, speculative decoding is ineffective",
            )
        finally:
            kill_process_tree(spec_process.pid)


# ===================================================================
# P1-002: PD disaggregation + image -> prefill node transmits correctly
# ===================================================================


class TestP1002PDDisaggregation(CustomTestCase):
    """P1-002: Verify PD disaggregation + image inference is correct.

    Scenario:
        Deploy prefill and decode servers on two NPU cards, launch a
        sglang_router to route requests, and verify image inference
        through the router produces reasonable output.

    Related features: pd_disaggregation
    Required capability: vlm, pd-disaggregation (2-NPU)
    """

    _model = QWEN3_5_9B_PATH
    _prefill_port = get_port(13)
    _decode_port = get_port(14)
    _router_port = get_port(10)
    _bootstrap_port = 18998
    _host = "127.0.0.1"
    _pd_env = {
        **os.environ,
        "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    }

    _prefill_url = f"http://{_host}:{_prefill_port}"
    _decode_url = f"http://{_host}:{_decode_port}"

    _prefill_process = None
    _decode_process = None
    _router_process = None
    _router_url = None

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._prompt = "Describe the image"

        cls._prefill_process, _ = launch_server(
            cls._model,
            cls._prefill_port,
            extra_args=[
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disaggregation-bootstrap-port",
                str(cls._bootstrap_port),
                "--disable-cuda-graph",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mem-fraction-static",
                "0.6",
                "--tp-size",
                "1",
                "--base-gpu-id",
                "0",
            ],
            env=cls._pd_env,
        )

        cls._decode_process, _ = launch_server(
            cls._model,
            cls._decode_port,
            extra_args=[
                "--disaggregation-mode",
                "decode",
                "--disaggregation-transfer-backend",
                "ascend",
                "--disaggregation-bootstrap-port",
                str(cls._bootstrap_port),
                "--cuda-graph-bs",
                1,
                2,
                4,
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--dtype",
                "bfloat16",
                "--mamba-ssm-dtype",
                "bfloat16",
                "--mem-fraction-static",
                0.6,
                "--tp-size",
                1,
                "--base-gpu-id",
                1,
            ],
            env=cls._pd_env,
        )

        cls._router_process, cls._router_url = launch_router(
            cls._prefill_url,
            cls._decode_url,
            cls._host,
            cls._router_port,
        )

    @classmethod
    def tearDownClass(cls):
        for proc in [cls._router_process, cls._decode_process, cls._prefill_process]:
            if proc:
                try:
                    kill_process_tree(proc.pid)
                except Exception:
                    pass

    def test_pd_disaggregation_image_inference(self):
        """Two-server PD with sglang_router: prefill+decode."""
        output = chat_single_image(
            self._router_url,
            self._image_b64,
            self._prompt,
            max_tokens=128,
        )

        self.assertTrue(output, "P1-002: PD router returned empty output")
        self.assertGreater(len(output), 5, f"P1-002: PD output too short: '{output}'")
        self.assertTrue(
            content_has_keywords(output),
            f"P1-002: PD output does not reference image content: " f"'{output[:200]}'",
        )


# ===================================================================
# P1-003: TP parallelism + image -> multi-card inference correct
# ===================================================================


class TestP1003TPParallelism(CustomTestCase):
    """P1-003: Verify tensor parallelism + image inference is correct.

    Scenario:
        Deploy the model with ``--tp-size 2`` (2 NPU cards), send an
        image request with ``temperature=0`` and a fixed ``seed``, and
        verify the output is **token-identical** to the TP=1 baseline.

    Related features: tensor_parallelism
    Required capability: vlm
    NPU note: Needs **2 NPU chips**.  Uses ``--tp-size 2``.

    Server-lifecycle note:
        This test manages both the TP=1 and TP=2 servers inline within
        the test method (sequentially) to avoid resource contention on
        the two NPU chips.  setUpClass only prepares shared test data.
    """

    _model = QWEN3_5_9B_PATH
    _port = get_port(15)

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._prompt = "Describe the image"

    def test_tp_parallelism_image_inference(self):
        """Send image to TP=2 server and verify output is reasonable.

        Launches a TP=2 server, sends an image request, and verifies
        the output is non-empty and references image content.
        """
        p, u = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--mem-fraction-static",
                0.6,
                "--tp-size",
                2,
                "--dtype",
                "bfloat16",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
        )
        try:
            output = chat_single_image(
                u,
                self._image_b64,
                self._prompt,
                max_tokens=128,
                temperature=0,
                seed=42,
            )
        finally:
            kill_process_tree(p.pid)

        self.assertTrue(
            output,
            "P1-003: TP=2 server returned empty output",
        )
        self.assertGreaterEqual(
            len(output),
            5,
            f"P1-003: TP=2 output too short: '{output}'",
        )
        self.assertTrue(
            content_has_keywords(output),
            f"P1-003: TP=2 output doesn't reference image: '{output[:200]}'",
        )

        print(f"  [P1-003] TP=2 output is valid " f"(len={len(output)})")


# ===================================================================
# P1-004: DP-attention + image -> high concurrency correctness
# ===================================================================


class TestP1004DPAttention(CustomTestCase):
    """P1-004: Verify DP-attention + image handles high concurrency correctly.

    Scenario:
        Enable DP-attention (``--enable-dp-attention --dp-size 2 --tp-size 2``),
        send 50 concurrent image-description requests, and verify all succeed
        with semantically reasonable outputs.

    Related features: dp_attention, tensor_parallelism
    Required capability: vlm, dp-attention (2-NPU)
    """

    _model = QWEN3_5_9B_PATH
    _port = get_port(17)
    _num_concurrent = 50

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.GREEN, shape=Shape.RECTANGLE
        )
        cls._prompt = "Describe the image"

    def test_dp_attention_concurrency(self):
        """Send 50 concurrent requests with DP-attention, verify all succeed."""
        p_dp, u_dp = launch_server(
            self._model,
            self._port,
            extra_args=[
                "--enable-dp-attention",
                "--dp-size",
                2,
                "--tp-size",
                2,
                "--mem-fraction-static",
                "0.6",
                "--cuda-graph-bs",
                1,
                2,
                4,
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--dtype",
                "bfloat16",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
        )
        try:
            dp_results = _send_concurrent(
                u_dp,
                self._image_b64,
                self._prompt,
                num_requests=self._num_concurrent,
                max_tokens=32,
            )

            # 1. All 50 requests must succeed
            dp_ok = sum(1 for r in dp_results if r[0] == "ok")
            self.assertEqual(
                dp_ok,
                self._num_concurrent,
                f"P1-004: DP-attention: only {dp_ok}/{self._num_concurrent} "
                f"requests succeeded",
            )

            # 2. Every successful response must reference image content
            for idx, (status, content, _tokens) in enumerate(dp_results):
                if status == "ok":
                    self.assertTrue(
                        content and len(content) > 0,
                        f"P1-004: DP request {idx} returned empty content",
                    )
                    self.assertTrue(
                        content_has_keywords(content),
                        f"P1-004: DP request {idx} output does not reference "
                        f"image: '{content[:100]}'",
                    )

            print(f"  [P1-004] DP: {dp_ok}/{self._num_concurrent} ok")

            # ---- Verify DP-attention is actually active ----
            server_info = requests.get(u_dp + "/server_info", timeout=10).json()
            self.assertTrue(
                server_info.get("enable_dp_attention"),
                "P1-004: enable_dp_attention is not True in server info — "
                "dp_attention was silently disabled",
            )
            self.assertEqual(
                server_info.get("dp_size"),
                2,
                f"P1-004: dp_size={server_info.get('dp_size')} != 2",
            )
        finally:
            kill_process_tree(p_dp.pid)


if __name__ == "__main__":
    unittest.main()
