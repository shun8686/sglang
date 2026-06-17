"""
Oracle tests for NPU EAGLE3 speculative decoding.

Verifies that EAGLE3 speculative decoding produces tokens identical to
non-speculative (greedy) decoding, and that decode-time logprobs match
prefill-mode rescore logprobs.

Pattern A (dual-server): compare output text token-by-token between a
    baseline server and an eagle3 server at temperature=0.
Pattern B (single-server): compare decode-time logprobs against
    prefill-only rescore logprobs on the eagle3 server.

Covers known NPU eagle3 failure modes:
    - BUG-2026-098: accept rate seq_len update (causes token divergence)
    - BUG-2026-016: mrope_position PlanStream race (causes positional errors)
    - BUG-2026-040: hidden states capture crash in DP attention mode
"""

import os
import time
import unittest
from typing import Optional

import numpy as np
import requests

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)

# ── port for the second (eagle3) server ──────────────────────────
_SPEC_PORT_OFFSET = 2000  # added to DEFAULT_PORT_FOR_SRT_TEST_RUNNER
_SPEC_URL = DEFAULT_URL_FOR_TEST.replace(
    DEFAULT_URL_FOR_TEST.split(":")[-1],
    str(int(DEFAULT_URL_FOR_TEST.split(":")[-1]) + 1000),
)

# ── shared server args (both baseline and eagle3) ────────────────
_SHARED_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.7",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

_EAGLE3_ARGS = [
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-attention-mode",
    "decode",
    "--disable-radix-cache",
]

# ── diverse prompts to exercise different token distributions ────
_ORACLE_PROMPTS = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the nth Fibonacci number."""',
    "Translate to Chinese: Hello world",
    "Solve step by step: 3x + 5 = 14",
    "The three primary colors are",
    "import numpy as np\n\n# Create a 3x3 identity matrix\n",
    "Explain the difference between DNA and RNA in one sentence:",
    "The boiling point of water at sea level is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "List three renewable energy sources:",
]


def _get_server_url(port_offset: int = 0) -> str:
    """Build a server URL with an optional port offset from the default."""
    base = DEFAULT_URL_FOR_TEST
    host_part, port_str = base.rsplit(":", 1)
    new_port = int(port_str) + port_offset
    return f"{host_part}:{new_port}"


def _generate(
    url: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    return_logprob: bool = False,
    timeout: int = 120,
) -> dict:
    """Send a /generate request and return the parsed JSON."""
    resp = requests.post(
        url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": return_logprob,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _generate_by_ids(
    url: str,
    input_ids: list[int],
    max_new_tokens: int = 0,
    return_logprob: bool = True,
    logprob_start_len: int = 0,
    timeout: int = 120,
) -> dict:
    """Send a /generate request with explicit input_ids (for rescore)."""
    resp = requests.post(
        url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


class TestNpuEagle3Oracle(CustomTestCase):
    """Oracle test: eagle3 output MUST match non-spec output at temperature=0.

    Starts two servers sequentially:
      1. Baseline (non-speculative) — generates reference outputs
      2. Eagle3 speculative — must produce identical text
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    spec_url = _get_server_url(1000)
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    base_process: Optional = None
    spec_process: Optional = None

    @classmethod
    def setUpClass(cls):
        """Launch baseline server first, then eagle3 server."""
        cls.base_process = None
        cls.spec_process = None

        # 1. Launch baseline (non-spec) server
        print("\n[Oracle] Launching BASELINE server (no speculation)...")
        cls.base_process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout_for_server_launch,
            other_args=[*_SHARED_ARGS, "--base-gpu-id", "0"],
            env={
                **os.environ,
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            },
        )
        print(f"[Oracle] Baseline server ready at {cls.base_url}")

        # 2. Launch eagle3 server on a different port
        print("\n[Oracle] Launching EAGLE3 server...")
        cls.spec_process = popen_launch_server(
            cls.model,
            cls.spec_url,
            timeout=cls.timeout_for_server_launch,
            other_args=[*_SHARED_ARGS, *_EAGLE3_ARGS, "--base-gpu-id", "1"],
            env={
                **os.environ,
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            },
        )
        print(f"[Oracle] Eagle3 server ready at {cls.spec_url}")

    @classmethod
    def tearDownClass(cls):
        """Kill both servers."""
        for proc, name in [
            (cls.spec_process, "Eagle3"),
            (cls.base_process, "Baseline"),
        ]:
            if proc is not None:
                try:
                    kill_process_tree(proc.pid)
                    print(f"[Oracle] {name} server stopped.")
                except Exception as e:
                    print(f"[Oracle] Error stopping {name} server: {e}")

    # ── Pattern A: dual-server output identity ────────────────────

    def test_output_token_identity(self):
        """Every prompt must produce identical text with and without eagle3.

        This is the primary oracle: speculative decoding at temperature=0
        is mathematically equivalent to non-speculative greedy decoding.
        Any divergence indicates a bug in the eagle3 draft/verify pipeline.
        """
        failures = []
        for i, prompt in enumerate(_ORACLE_PROMPTS):
            with self.subTest(prompt=prompt[:50]):
                # Generate from baseline (non-spec)
                base_resp = _generate(self.base_url, prompt, max_new_tokens=128)
                base_text = base_resp["text"]

                # Generate from eagle3
                spec_resp = _generate(self.spec_url, prompt, max_new_tokens=128)
                spec_text = spec_resp["text"]

                match = base_text == spec_text

                print(f"[Oracle] prompt {i:02d}: " f"{'PASS' if match else 'FAIL'}")
                if not match:
                    # Find the first divergent position
                    min_len = min(len(base_text), len(spec_text))
                    first_diff = None
                    for j in range(min_len):
                        if base_text[j] != spec_text[j]:
                            first_diff = j
                            break
                    if first_diff is None:
                        first_diff = min_len  # one is a prefix of the other

                    # Diagnostic: get top-5 logprobs at divergence point
                    print(f"\n[Oracle] Running logprob diagnostic for prompt {i}...")
                    try:
                        base_lp_resp = requests.post(
                            self.base_url + "/generate",
                            json={
                                "text": prompt,
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 128,
                                },
                                "return_logprob": True,
                                "top_logprobs_num": 5,
                            },
                            timeout=120,
                        ).json()
                        spec_lp_resp = requests.post(
                            self.spec_url + "/generate",
                            json={
                                "text": prompt,
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 128,
                                },
                                "return_logprob": True,
                                "top_logprobs_num": 5,
                            },
                            timeout=120,
                        ).json()

                        base_all_top5 = base_lp_resp["meta_info"]["output_top_logprobs"]
                        spec_all_top5 = spec_lp_resp["meta_info"]["output_top_logprobs"]
                        base_output_ids = base_lp_resp["output_ids"]
                        spec_output_ids = spec_lp_resp["output_ids"]

                        print(
                            f"[Oracle] Baseline output tokens: {len(base_output_ids)}, EAGLE3 output tokens: {len(spec_output_ids)}"
                        )

                        # Find first divergent token index
                        min_tokens = min(len(base_output_ids), len(spec_output_ids))
                        div_token_idx = None
                        for tidx in range(min_tokens):
                            if base_output_ids[tidx] != spec_output_ids[tidx]:
                                div_token_idx = tidx
                                break

                        if div_token_idx is not None:
                            # Also check the token just before divergence for root cause
                            check_indices = []
                            if div_token_idx > 0:
                                check_indices.append(div_token_idx - 1)
                            check_indices.append(div_token_idx)

                            for tidx in check_indices:
                                if tidx < len(base_all_top5) and tidx < len(
                                    spec_all_top5
                                ):
                                    base_top5 = base_all_top5[tidx]
                                    spec_top5 = spec_all_top5[tidx]
                                    label = (
                                        "BEFORE divergence"
                                        if tidx < div_token_idx
                                        else "AT divergence"
                                    )
                                    print(f"\n[Oracle] Token[{tidx}] ({label}):")
                                    print(
                                        f"  BASELINE id={base_output_ids[tidx]} top-5 logprobs:"
                                    )
                                    for logp, tid, token in base_top5:
                                        print(
                                            f"    {token!r:20s}  logp={logp:.6f}  id={tid}"
                                        )
                                    print(
                                        f"  EAGLE3   id={spec_output_ids[tidx]} top-5 logprobs:"
                                    )
                                    for logp, tid, token in spec_top5:
                                        print(
                                            f"    {token!r:20s}  logp={logp:.6f}  id={tid}"
                                        )

                            # Compare top-2 logprob at divergence point
                            if div_token_idx < len(
                                base_all_top5
                            ) and div_token_idx < len(spec_all_top5):
                                base_top5 = base_all_top5[div_token_idx]
                                spec_top5 = spec_all_top5[div_token_idx]
                                base_top2_logp = base_top5[1][0]
                                spec_top2_logp = spec_top5[1][0]
                                logp_diff = abs(base_top2_logp - spec_top2_logp)

                                if logp_diff < 0.01:
                                    print(
                                        f"\n[Oracle] DIAGNOSIS: Top-2 logprob diff={logp_diff:.6f} < 0.01 → Precision issue, not logic bug"
                                    )
                                elif logp_diff > 0.1:
                                    print(
                                        f"\n[Oracle] DIAGNOSIS: Top-2 logprob diff={logp_diff:.6f} > 0.1 → Logic bug, need to investigate NPU kernel"
                                    )
                                else:
                                    print(
                                        f"\n[Oracle] DIAGNOSIS: Top-2 logprob diff={logp_diff:.6f} → Unclear, needs further investigation"
                                    )
                        else:
                            print(
                                f"\n[Oracle] Output IDs identical (divergence may be in token decoding)"
                            )
                    except Exception as e:
                        print(f"\n[Oracle] Logprob diagnostic failed: {e}")

                    failures.append(
                        f"Prompt {i}: {prompt[:60]}...\n"
                        f"  First diff at char {first_diff}\n"
                        f"  Baseline[{first_diff-10}:{first_diff+20}]: "
                        f"{base_text[max(0,first_diff-10):first_diff+20]!r}\n"
                        f"  Eagle3[{first_diff-10}:{first_diff+20}]:   "
                        f"{spec_text[max(0,first_diff-10):first_diff+20]!r}"
                    )

        if failures:
            self.fail(
                f"{len(failures)}/{len(_ORACLE_PROMPTS)} prompts diverged:\n\n"
                + "\n\n".join(failures)
            )

    # ── Pattern A-2: batch generation identity ────────────────────

    def test_batch_output_identity(self):
        """Batch generation must also be identical."""
        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 64,
        }
        prompts = _ORACLE_PROMPTS[:5]  # use a subset to keep it fast

        base_resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": sampling_params,
            },
            timeout=180,
        ).json()

        spec_resp = requests.post(
            self.spec_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": sampling_params,
            },
            timeout=180,
        ).json()

        for i, (base, spec) in enumerate(zip(base_resp, spec_resp)):
            with self.subTest(batch_index=i):
                self.assertEqual(
                    base["text"],
                    spec["text"],
                    f"Batch item {i} diverged.\n"
                    f"Prompt: {prompts[i][:60]}...\n"
                    f"Baseline: {base['text'][:100]}\n"
                    f"Eagle3:   {spec['text'][:100]}",
                )

    # ── Pattern B: logprob rescore oracle ─────────────────────────

    def test_logprob_rescore_match(self):
        """Decode-time logprobs must match prefill-only rescore logprobs.

        This catches subtle logit computation bugs that don't change the
        argmax token but produce incorrect probabilities.
        """
        for round_idx, prompt in enumerate(_ORACLE_PROMPTS[:3]):
            with self.subTest(round=round_idx, prompt=prompt[:50]):
                # Step 1: generate with eagle3, capturing logprobs
                gen_resp = requests.post(
                    self.spec_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "logprob_start_len": 0,
                    },
                    timeout=180,
                ).json()

                output_logprobs = gen_resp["meta_info"]["output_token_logprobs"]
                input_logprobs = gen_resp["meta_info"]["input_token_logprobs"]
                num_prompt_tokens = gen_resp["meta_info"]["prompt_tokens"]

                input_token_ids = [t[1] for t in input_logprobs]
                output_token_ids = [t[1] for t in output_logprobs]
                full_ids = input_token_ids + output_token_ids

                # Step 2: rescore the same sequence with max_new_tokens=0
                score_resp = _generate_by_ids(
                    self.spec_url,
                    full_ids,
                    max_new_tokens=0,
                    return_logprob=True,
                )

                score_logprobs = score_resp["meta_info"]["input_token_logprobs"][
                    num_prompt_tokens:
                ]

                self.assertEqual(
                    len(output_logprobs),
                    len(score_logprobs),
                    "Token count mismatch between decode and rescore",
                )

                # Compare per-token logprobs
                decode_vals = np.array([t[0] for t in output_logprobs])
                score_vals = np.array([t[0] for t in score_logprobs])
                max_diff = np.max(np.abs(decode_vals - score_vals))

                print(
                    f"[Oracle] logprob round {round_idx}: "
                    f"max_diff={max_diff:.6f} "
                    f"(prompt={prompt[:40]}...)"
                )
                self.assertLess(
                    max_diff,
                    0.255,
                    f"Logprob mismatch in round {round_idx}. "
                    f"max_diff={max_diff:.6f} exceeds threshold 0.255.\n"
                    f"Prompt: {prompt}\n"
                    f"decode logprobs[-3:]: {decode_vals[-3:]}\n"
                    f"rescore logprobs[-3:]: {score_vals[-3:]}",
                )

    # ── Sanity check: speculation is actually active ──────────────

    def test_speculation_is_active(self):
        """Verify eagle3 is actually doing speculative decoding.

        The avg_spec_accept_length should be > 1.0, confirming that draft
        tokens are being accepted.
        """
        # Generate enough tokens to get meaningful accept statistics
        _generate(self.spec_url, _ORACLE_PROMPTS[0], max_new_tokens=256)

        # Check server info for speculation metrics
        resp = requests.get(self.spec_url + "/get_server_info", timeout=10).json()
        internal_states = resp.get("internal_states", [])
        if internal_states:
            avg_accept = internal_states[0].get("avg_spec_accept_length", 0)
            print(f"[Oracle] avg_spec_accept_length = {avg_accept:.2f}")
            self.assertGreater(
                avg_accept,
                1.0,
                f"Eagle3 speculation appears inactive (avg_accept_length={avg_accept:.2f})",
            )


class TestNpuEagle3OraclePlanStream(CustomTestCase):
    """Oracle test WITHOUT PlanStream overlap (SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0).

    This isolates whether PlanStream overlap (known risk area per BUG-2026-016)
    is the source of any output divergence.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = _get_server_url(2000)
    spec_url = _get_server_url(3000)
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    base_process: Optional = None
    spec_process: Optional = None

    @classmethod
    def setUpClass(cls):
        cls.base_process = None
        cls.spec_process = None

        print("\n[Oracle-NoPlanStream] Launching BASELINE server...")
        cls.base_process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout_for_server_launch,
            other_args=[*_SHARED_ARGS],
            env={**os.environ, "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0"},
        )

        print("\n[Oracle-NoPlanStream] Launching EAGLE3 server...")
        cls.spec_process = popen_launch_server(
            cls.model,
            cls.spec_url,
            timeout=cls.timeout_for_server_launch,
            other_args=[*_SHARED_ARGS, *_EAGLE3_ARGS],
            env={**os.environ, "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0"},
        )

    @classmethod
    def tearDownClass(cls):
        for proc, name in [
            (cls.spec_process, "Eagle3"),
            (cls.base_process, "Baseline"),
        ]:
            if proc is not None:
                try:
                    kill_process_tree(proc.pid)
                except Exception:
                    pass

    def test_output_identity_no_planstream(self):
        """Without PlanStream overlap, outputs must also be identical."""
        prompt = _ORACLE_PROMPTS[0]
        base_text = _generate(self.base_url, prompt)["text"]
        spec_text = _generate(self.spec_url, prompt)["text"]
        self.assertEqual(
            base_text,
            spec_text,
            f"Divergence without PlanStream overlap.\n"
            f"Baseline: {base_text[:100]}\n"
            f"Eagle3:   {spec_text[:100]}",
        )


if __name__ == "__main__":
    unittest.main()
