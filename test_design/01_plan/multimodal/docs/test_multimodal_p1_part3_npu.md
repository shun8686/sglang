# Multimodal P1 Part 3 NPU Tests — Completions API Parameter Interactions

## Overview

This file implements **P1-011 through P1-021** complementing the existing
P1 test cases (P1-001–P1-010 in parts 1 & 2).  These tests verify that
**OpenAI Chat Completions API parameters** interact correctly with
**multimodal (image) inputs** on the NPU platform.

Reference: [OpenAI Completions API Reference](https://developers.openai.com/api/reference/resources/completions/methods/create)

## Parameter Coverage Matrix

The Chat Completions API exposes the following parameters that are
relevant to multimodal inference.  The table shows which are covered by
this test file vs. already covered by existing tests.

| Parameter | Prior art | This file | Test ID | Rationale |
|-----------|-----------|-----------|---------|-----------|
| `temperature` | temp=0 in all P0/P1 | temp=0/0.8/1.5 | P1-011 | Core sampling; must work with image tokens |
| `top_p` | not tested | top_p=0.1/0.5/1.0 | P1-012 | Alternative to temperature; nucleus sampling with images |
| `seed` | not tested | seed=42 determinism | P1-013 | Reproducibility is critical for debugging |
| `max_tokens` | 16/32/64/128/256 | 16 vs 256 contrast | P1-014 | Token budget must be respected with image prompts |
| `stop` | not tested | single + multi stop | P1-015 | Stop sequences must work with embedded image tokens |
| `n` | not tested | n=2 | P1-016 | Multiple completions for same image+text prompt |
| `stream` | not tested | stream=True | P1-017 | Streaming must deliver image-relevant chunks |
| `stream_options` | not tested | include_usage=True | P1-018 | Usage chunk must appear in streamed response |
| `frequency_penalty` | not tested | 0 vs 1.5 | P1-019 | Penalty must not break image understanding |
| `presence_penalty` | not tested | 0 vs 1.5 | P1-020 | Penalty must not break image understanding |
| `logprobs` | not tested | logprobs=True | P1-021 | Log probabilities must be returned for image+tokens |

### Parameters NOT tested in this file (rationale)

| Parameter | Reason for omission |
|-----------|-------------------|
| `logit_bias` | Tokenizer-specific; requires GPT tokenizer IDs; limited multimodal relevance |
| `user` | Passthrough field with no server-side effect; no multimodal interaction |
| `max_completion_tokens` | Newer alias for max_tokens; same code path |
| `best_of` (completions) | Legacy completions endpoint; not tested with multimodal in SGLang |
| `echo` / `suffix` (completions) | Legacy completions-only params; multimodal uses chat completions |
| `max_dynamic_patch` / `min_dynamic_patch` | SGLang multimodal extensions; tested implicitly via image resolution tests (P0-005) |

## Test Case Details

### P1-011: temperature + multimodal

**Purpose**: Verify that `temperature` sampling interacts correctly with multimodal
inputs — all temperature values produce correct image descriptions.

**Sub-tests**:
1. `temperature=0` → deterministic, must correctly identify image color + shape
2. `temperature=0.8` → stochastic but must still identify image content
3. `temperature=1.5` → high randomness but must not produce gibberish about image

**Verification**:
- Non-empty output for all temperature values
- Each output correctly identifies expected color and shape
- temperature=0 output is shorter and more deterministic in style

### P1-012: top_p + multimodal

**Purpose**: Verify that `top_p` (nucleus sampling) interacts correctly with
multimodal inputs.

**Sub-tests**:
1. `top_p=0.1` → narrow sampling, must still describe image
2. `top_p=0.5` → moderate sampling
3. `top_p=1.0` → full distribution

**Verification**:
- All produce correct image descriptions
- Low top_p does not cause degenerate output

### P1-013: seed + multimodal (determinism)

**Purpose**: Verify that `seed` produces deterministic output for the same image
and prompt, even when temperature > 0 (stochastic sampling).

**Test**:
- Use `temperature=0.2` (non-zero, so RNG is active) + `seed=42`
- Send 2 requests with identical image + prompt + seed
- Compare outputs byte-for-byte
- Without seed, temperature=0.2 would produce different outputs

**Verification**:
- Both outputs are identical (if NPU hardware determinism permits)
- Both correctly identify image content
- Falls back to semantic equivalence if strict determinism isn't guaranteed

### P1-014: max_tokens + multimodal

**Purpose**: Verify that `max_tokens` correctly limits output length while
preserving image understanding.

**Sub-tests**:
1. `max_tokens=16` → very short output, finish_reason="length"
2. `max_tokens=256` → longer output, finish_reason="stop" or "length"

**Verification**:
- Short output has <= 16 tokens
- Both outputs reference image content
- finish_reason is correct

### P1-015: stop + multimodal

**Purpose**: Verify that `stop` sequences cause early termination with image prompts.

**Sub-tests**:
1. Single stop string (`"STOPEND"`) — model instructed to end response with this word
2. Array of stop strings (`["STOPEND", "FINISHED"]`) — model told to end with either

**Verification**:
- Response stops at the stop word
- finish_reason is "stop"
- Image content (color, shape) is described before the stop
- Using explicit word tokens avoids tokenization edge cases with punctuation

### P1-016: n + multimodal

**Purpose**: Verify that `n` returns multiple **independent** completions for
the same image prompt — not just duplicated copies of one result.

**Test**:
- Send request with `n=2, temperature=0.7`
- temperature=0.7 ensures the two sampling paths draw from different RNG
  states, so the two choices should differ

**Verification**:
- Response has exactly 2 choices
- Both choices contain image-relevant content (color/shape)
- Choices should differ at temperature=0.7 (if identical, log a NOTE —
  may indicate shared RNG or n degenerating to greedy)

### P1-017: stream + multimodal

**Purpose**: Verify that `stream=True` works correctly with image input.

**Test**:
- Stream a chat completion with image + text

**Verification**:
- Multiple chunks received
- Combined text correctly identifies image content
- Final chunk has finish_reason

### P1-018: stream_options + multimodal

**Purpose**: Verify that `stream_options={"include_usage": True}` includes a
usage chunk in the streamed response.

**Test**:
- Stream with `include_usage=True`

**Verification**:
- At least one chunk contains usage data
- Combined text identifies image content
- Total tokens > 0

### P1-019: frequency_penalty + multimodal

**Purpose**: Verify that `frequency_penalty` affects output repetition without
breaking image understanding.

**Sub-tests**:
1. `frequency_penalty=0` → baseline
2. `frequency_penalty=1.5` → high penalty against repetition

**Verification**:
- Both outputs correctly identify image
- Penalty=1.5 output may differ stylistically but still references color/shape

### P1-020: presence_penalty + multimodal

**Purpose**: Verify that `presence_penalty` affects vocabulary diversity without
breaking image understanding.

**Sub-tests**:
1. `presence_penalty=0` → baseline
2. `presence_penalty=1.5` → high penalty encourages new topics

**Verification**:
- Both outputs correctly identify image
- Penalty=1.5 output may have more diverse vocabulary but still references color/shape

### P1-021: logprobs + top_logprobs + multimodal

**Purpose**: Verify that `logprobs` and `top_logprobs` return probability data
when used with image input.

**Test**:
- Send request with `logprobs=True, top_logprobs=3`

**Verification**:
- Response has `logprobs` field populated
- Content field still contains image description
- Token-level logprobs are present for output tokens

## Model Selection

- **Primary model**: `Qwen/Qwen3.5-9B` (GDN attention + DeepStack ViT)
  - Path: `/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B`
  - Reason: Qwen3.5-9B is the GDN/Mamba architecture VLM; parameters are
    validated against this architecture to cover the mamba path
  - All 11 test cases run against a single server instance

## Server Configuration

```
--device npu
--attention-backend ascend
--trust-remote-code
--enable-multimodal
--mm-attention-backend ascend_attn
--mem-fraction-static 0.78
--cuda-graph-bs 1 2 4
--mamba-scheduler-strategy extra_buffer
--tp-size 2
--dtype bfloat16
--mamba-ssm-dtype bfloat16
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

- 2 NPU, TP=2 (Qwen3.5-9B requires 2 cards)
- CUDA graphs enabled with batch sizes 1, 2, 4
- Mamba scheduler strategy: extra_buffer
- dtype: bfloat16 (both model weights and SSM)
- Speculative decoding: NEXTN (MTP heads), 3 steps, topk=1, 4 draft tokens
  (full config matching TestP1001SpeculativeDecoding)

## CI Registration

- `register_npu_ci(est_time=600, suite="nightly-2-npu-a3", nightly=True)`
- Estimated 600 s total (server startup ~60 s + 11 tests × ~30-50 s each)
- Uses 2 NPU cards (TP=2), registered in nightly-2-npu-a3

## How to Run

```bash
cd /home/d00662834/test-debug/sglang

# All tests
python3 -m unittest \
  test_design/03_testcase/multimodal/test_multimodal_p1_3_npu.py -v

# Single test
python3 -m unittest \
  test_design/03_testcase/multimodal/test_multimodal_p1_3_npu.py \
  TestMultimodalP1ParameterInteractions.test_011_temperature_multimodal -v
```

## Expected Output (example)

```
[P1-011/temp=0] output_len=52
[P1-011/temp=0.8] output_len=78
[P1-011/temp=1.5] output_len=91
[P1-012/top_p=0.1] output_len=48
[P1-012/top_p=0.5] output_len=65
[P1-012/top_p=1.0] output_len=72
[P1-013] deterministic=True  output_len=52
[P1-014/max_tokens=16] output_len_approx=16  finish_reason=length
[P1-014/max_tokens=256] output_len=120  finish_reason=stop
[P1-015/stop="."] stopped_early=True  finish_reason=stop
[P1-015/stop=[".", "!"] stopped_early=True  finish_reason=stop
[P1-016] 2 choices returned, all reference image content
[P1-017] streamed 15 chunks, combined identifies image
[P1-018] stream+usage 18 chunks, usage_chunk=True
[P1-019/freq_pen=0] output_len=65  identified image
[P1-019/freq_pen=1.5] output_len=71  identified image
[P1-020/pres_pen=0] output_len=58  identified image
[P1-020/pres_pen=1.5] output_len=63  identified image
[P1-021] logprobs=True top_logprobs=3  content_correct=True
```

## Limitations

1. **Determinism (P1-013)**: SGLang's seed-based determinism is best-effort.
   Hardware-level non-determinism (NPU kernel scheduling, attention ordering)
   may cause minor token-level differences.  The test verifies exact match
   but includes a relaxed fallback that accepts semantic equivalence.

2. **logprobs (P1-021)**: The NPU backend may return logprobs in a different
   format than the CUDA backend.  The test uses the OpenAI-compatible
   response format and checks for presence of logprob fields, not specific
   values.

3. **Single model**: Only Qwen3.5-9B (GDN/Mamba architecture) is tested.
   Other VLMs (Qwen3-VL, Qwen2.5-VL, GLM-4.6V) may have different behavior
   with these parameters, especially on the dense-attention path.

## File Structure

```
test_design/
  03_testcase/
    multimodal/
      doc/
        test_multimodal_p1_part3_npu.md     # This file
      test_multimodal_p1_3_npu.py           # Test implementation
      utils.py                              # Shared helpers (no changes needed)
```
