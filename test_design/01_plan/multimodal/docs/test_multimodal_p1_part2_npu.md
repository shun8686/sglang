# P1-006 to P1-010: Test Implementation Notes

## File

`test_design/03_testcase/multimodal/test_multimodal_p1_part2_npu.py`

## Overview

Implements five P1 multimodal interaction test cases for the NPU (Ascend) platform, targeting feature interactions that are important but not required to pass every commit. All tests are registered as `nightly-1-npu-a3` with `est_time=900`.

---

## P1-006: Structured output + image -> JSON Schema constraint

### Scenario
User provides an image (coloured ellipse on blue background) and asks the model to extract "color" and "shape" as JSON fields using the OpenAI `response_format` parameter with `json_schema`.

### Related features
`structured_outputs` (xgrammar)

### NPU assessment
- xgrammar is `platform_agnostic` (CPU-side grammar engine). No direct NPU interaction.
- Risk: **Low**. Grammar constraint is applied to logits after ViT encoding, no kernel-level interaction.

### Verification strategy
1. Parse the response as JSON (mandatory).
2. Verify "color" and "shape" keys exist.
3. Verify values are non-empty strings referencing expected image content (red/blue/white, circle/ellipse).
4. No crash (verify usage fields present).

### Implementation details
- `response_format` parameter with JSON schema.
- Schema requires `color` (string) and `shape` (string).
- Image: blue background, red ellipse with white outline.
- Prompt: "Extract the color and shape from this image as JSON fields: color and shape."

---

## P1-007: Tool call + image -> image param in function arguments

### Scenario
User provides an image and asks the model to analyse it by calling a registered `analyze_image` function. The model must output a valid tool call with a "description" parameter.

### Related features
`tool_parser`

### NPU assessment
- tool_parser is `platform_agnostic` (CPU-side text parser). No direct NPU interaction.
- Risk: **Low**. Parsing happens after LLM output.

### Verification strategy
1. Check `tool_calls` is not None.
2. Verify function name is "analyze_image".
3. Parse arguments as JSON, verify "description" key is present and non-empty.
4. Verify description references image content.
5. No crash.

### Implementation details
- OpenAI `tools` parameter with `tool_choice="auto"`.
- Tool: `analyze_image({"description": "string"})`.
- Prompt: "Analyze this image and call the analyze_image function with a description of what you see."

---

## P1-008: Chunked prefill + Offloading -> embeddings persist across chunks

### Scenario
A long text prefix (~5K tokens) is prepended to an image, triggering chunked prefill. CPU offloading (`--cpu-offload-gb 4`) is enabled to swap model weights between GPU and CPU memory. Image features must survive chunk boundaries under offload.

### Related features
- `offloading` (medium NPU participation)
- `chunked_prefill` (weak NPU participation — page-size constraint only)

### NPU assessment
- Offloading has medium NPU participation. The `--cpu-offload-gb` flag controls host-side memory allocation.
- Chunked prefill on NPU only enforces page-size alignment. Advanced chunking features may not be NPU-optimized.
- Combined risk: **Medium**. The offload + chunk + multimodal triple interaction has no dedicated NPU test coverage.

### Verification strategy
1. Output references image content (proving image features survived chunk boundaries under offload).
2. Output references long prefix content (proving the full sequence was processed).
3. No NaN in output (offload corruption indicator).

### Implementation details
- `--chunked-prefill-size 512 --cpu-offload-gb 4`
- Long prefix: ~5K tokens of English text about chunked prefill and offloading.
- Prompt: "Summarize the above" (covers both text prefix and image).

---

## P1-009: DP-attention + DP LM Head + image -> LM head sharding

### Scenario
Two servers are launched sequentially:
1. Baseline: `--enable-dp-attention` only
2. DP LM Head: `--enable-dp-attention --enable-dp-lm-head`
Both receive the same image + prompt. Outputs are compared for semantic equivalence.

### Related features
- `dp_attention` (weak NPU participation)
- `hardware_backend` (DP LM Head is NPU-specific)

### NPU assessment
- DP LM Head is a **pure NPU optimization** (GPU has no DP LM head concept).
- Compatibility matrix shows "" with EP, PD, Quantization, Chunked Prefill, NPU Graph.
- Risk: **Medium**. Exercise of an NPU-only optimization path that is known to have incomplete compatibility.

### Verification strategy
1. Both outputs reference image content.
2. No NaN in either output (LM head sharding corruption indicator).
3. Both outputs are non-empty and have reasonable length.
4. Soft semantic equivalence: at least one shared keyword between outputs.

### Implementation details
- Two sequential server launches within a single test method.
- Ports: baseline=_PORT_BASE+19, DP LM Head=_PORT_BASE+20.
- Server args: `--enable-dp-attention` vs `--enable-dp-attention --enable-dp-lm-head`.

---

## P1-010: Overlap Schedule + speculative decoding + image

### Scenario
Two servers are launched sequentially:
1. Baseline: no speculative decoding.
2. Overlap: `--speculative-algorithm EAGLE3` with Qwen3-8B_eagle3 draft model, plus `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` and `SGLANG_ENABLE_SPEC_V2=1`.
Both receive the same image + prompt. Outputs are compared and TTFT is measured.

### Related features
`speculative_decoding`

### NPU assessment
- Requires `vlm-mtp` capability (**TBC** for Qwen3-VL). Test is skipped gracefully if draft weights are unavailable.
- Uses EAGLE3 (not NEXTN) because Qwen3-VL models lack MTP heads in their config.
- Draft model: Qwen3-8B_eagle3 (hidden_size=4096) with target Qwen3-VL-8B (text hidden_size=4096). Matching dimensions but VL-specific hidden states differ from text-only model training data.
- Compatibility shows "" with Chunked Prefill, NPU Graph, Quantization.
- Risk: **High**. Multiple experimental features (overlap schedule, spec v2, VLM EAGLE3) combined.

### Verification strategy
1. Both outputs reference image content.
2. No NaN in overlap output (draft corruption indicator).
3. TTFT ratio < 3.0 (overlap not pathologically slower than baseline).
4. Semantic plausibility: both outputs share at least one visual keyword.

### Known limitations
1. **vlm-mtp TBC**: If Qwen3-VL models don't support EAGLE3 with a separate draft model, this test will consistently skip.
2. **Non-VL draft model**: Qwen3-8B_eagle3 is designed for a text-only model. Its performance as VLM draft is untested; hidden state distributions from image tokens may cause poor draft quality (but not crashes).
3. **TTFT measurement**: CI load fluctuations make strict TTFT comparisons unreliable. The test uses a soft upper bound (3x baseline) rather than asserting overlap TTFT < baseline TTFT.
4. **EAGLE3 vs NEXTN**: The design doc specifies `--speculative-algorithm NEXTN`, but Qwen3-VL models don't have MTP heads. EAGLE3 is used instead as the supported speculative path.

### Implementation details
- Target model: Qwen3-VL-8B-Instruct
- Draft model: Qwen3-8B_eagle3
- Env vars: `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`, `SGLANG_ENABLE_SPEC_V2=1`
- Spec args: `--speculative-algorithm EAGLE3 --speculative-num-steps 4 --speculative-eagle-topk 1`
- Server launch wrapped in try/except for graceful skip on incompatibility.
- TTFT measured via streaming response.

---

## CI registration

```python
register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)
```

### Rationale
- Total estimated runtime: ~770s (5 test classes, each launching 1-2 servers).
- Nightly-only: P1 tests exercise non-critical interactions and have longer runtimes.
- Single-NPU: all tests run on 1 NPU (P1-010 uses Qwen3-VL-8B which fits on 1 NPU with mem-fraction-static=0.4).

### Estimated breakdown

| Test class | Server launches | Est. time | Notes |
|---|---|---|---|
| P1-006 StructuredOutput | 1 | 80s | Simple, xgrammar CPU-side |
| P1-007 ToolCall | 1 | 80s | Simple, tool_parser CPU-side |
| P1-008 ChunkedPrefillOffload | 1 | 150s | Offloading adds overhead |
| P1-009 DpLmHead | 2 (sequential) | 160s | Two server launches |
| P1-010 OverlapSchedule | 2 (sequential) | 300s | EAGLE3 loading overhead |
| **Total** | **7** | **~770s** | |

### Port assignments
- P1-006: _PORT\_BASE + 16
- P1-007: _PORT\_BASE + 17
- P1-008: _PORT\_BASE + 18
- P1-009 baseline: _PORT\_BASE + 19, DP-LM-Head: _PORT\_BASE + 20
- P1-010 baseline: _PORT\_BASE + 21, overlap: _PORT\_BASE + 22

---

## Cross-reference

| Test case | Design doc | Reference pattern |
|---|---|---|
| P1-006 | SS2.2 P1-006 | `test_multimodal_p0_advanced_npu.py`, OpenAI response_format |
| P1-007 | SS2.2 P1-007 | `test_multimodal_p0_advanced_npu.py`, OpenAI tools API |
| P1-008 | SS2.2 P1-008 | `test_multimodal_p0_advanced_npu.py`, `_launch_server` pattern |
| P1-009 | SS2.2 P1-009 | `test_multimodal_p0_advanced_npu.py` P0-007 (in-method server launch) |
| P1-010 | SS2.2 P1-010 | `test_design/03_testcase/eagle3/test_npu_eagle3.py` (EAGLE3 patterns) |

---

## Review checklist compliance

- [x] `register_npu_ci` at module level, not class decorator
- [x] `CustomTestCase` used everywhere
- [x] Defensive `tearDownClass` with `hasattr` guard
- [x] Programmatic image generation via PIL
- [x] All prompts in English
- [x] `--device npu`, `--attention-backend ascend`, `--trust-remote-code`, `--enable-multimodal` present
- [x] No `--cuda-graph-max-bs` conflicts
- [x] No manual model existence checks (except P1-010 EAGLE3 weight check for skip)
- [x] Unique ports for all servers
- [x] TTFT measurement via streaming (P1-010)
- [x] NPU-unsupported features handled (P1-010 skip)
- [x] `--cpu-offload-gb` flag string (no equals sign)
- [x] Weak-NPU features annotated in docstrings
- [x] Env vars passed via `env` dict to `popen_launch_server`
- [x] Prompt text in English
- [x] `suite` and `est_time` as literal values (not variables)
