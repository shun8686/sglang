# Multimodal P0 Basic NPU Tests

## Implemented Test Cases

This file implements **P0-001 through P0-005** from the multimodal interaction test
analysis report (`test_design/01_plan/multimodal-interaction-test-analysis.md`).

| Test ID | Test Name | Verification Points | Related Features |
|---------|-----------|-------------------|-----------------|
| P0-001 | Single image + text -> describe image content (Smoke Test) | Non-empty output, content relates to image, TTFT < 30 s | attention_backend, scheduling |
| P0-002 | Same image twice -> cache hit | Both outputs correct, TTFT2 < TTFT1 | Radix Cache, kv_cache_pool |
| P0-003 | Concurrent text + image requests -> isolation | All 15 succeed, no cross-pollution | scheduling, attention_backend |
| P0-004 | Multi-image -> compare two images | Output references both images, no OOM | attention_backend, scheduling |
| P0-005 | Variable size images -> different resolutions | All 3 sizes succeed, large no OOM, small no degradation | graph_compilation, attention_backend |

## Model Selection

- **Primary model**: `Qwen/Qwen3-VL-4B-Instruct` (path:
  `/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct`)
- **Reason**: Qwen3-VL is the primary VLM model on NPU with the widest feature
  support (vlm, vlm-moe, vlm-lora, vlm-reasoning). The 4B variant is small enough
  for fast CI iteration while still being representative of the VLM inference path
  (DeepStack ViT + decoder).
- **Alternative**: `Qwen/Qwen2.5-VL-3B-Instruct` could be substituted as a
  simpler non-MoE baseline if the primary model path is unavailable.

## Server Configuration

All tests share a single server started in `setUpClass`:

- **Device**: NPU (1x Ascend NPU)
- **TP size**: 1
- **Attention backend**: ascend
- **CUDA graphs**: disabled (`--disable-cuda-graph`)
- **Multimodal**: enabled (`--enable-multimodal`)
- **Memory fraction**: 0.4

## Known Limitations and Constraints

1. **TTFT threshold in P0-002**: The design doc originally proposed
   `TTFT2 < TTFT1 * 0.6` for the cache-hit test. Review (section 6.5 of the
   analysis report) noted that ViT encoding time dominates TTFT on NPU; caching
   only saves the LLM prefill portion. The implementation uses the relaxed
   constraint `TTFT2 < TTFT1`. This may still be flaky on a contended NPU;
   the threshold may need adjustment after empirical observation.

2. **Generated test images**: The test creates PNG images with geometric shapes
   (ellipses + text) using Pillow. These synthetic images are simple enough that
   well-known VLMs should describe them, but verification relies on keyword
   presence rather than semantic equivalence. If the model unexpectedly fails
   to recognise the synthetic pattern, the keyword check may trigger a false
   negative.

3. **Concurrency level in P0-003**: Limited to 15 total requests (10 text + 5
   image). Higher concurrency (50+50) is suggested by the design doc as a P1
   supplement but is not included here.

4. **No graph compilation**: All tests pass `--disable-cuda-graph`. Graph
   compilation tests are handled by P0-007 (not implemented in this file).

5. **No TP/PP/DP**: All tests use `--tp-size 1`. Multi-card parallelism is out
   of scope for these basic P0 tests.

6. **Single model**: Only Qwen3-VL-4B-Instruct is tested. Per the design doc's
   model matrix, P0 coverage ideally requires 4 models (Qwen3-VL, Qwen2.5-VL,
   Qwen3.5-MoE, GLM-4.6V) to cover all code paths. This file is a starting
   point for the most common path.

7. **CI registration**: Registered as `suite="per-commit-1-npu-a2"`. The
   estimated runtime (300 s) includes server startup; individual test methods
   are fast (5-20 s each after warmup).

## How to Run

### Prerequisites

- Ascend NPU environment with `torch_npu` and ascend runtime installed
- Model weights present at the cache path:
  `/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct`

### Run all tests

```bash
cd /home/d00662834/test-debug/sglang
python3 -m unittest \
  test_design/03_testcase/multimodal/test_multimodal_p0_basic_npu.py -v
```

### Run a single test

```bash
python3 -m unittest \
  test_design/03_testcase/multimodal/test_multimodal_p0_basic_npu.TestMultimodalP0Basic.test_001_single_image_smoke \
  -v
```

### Run with coverage

```bash
python3 -m coverage run -m unittest \
  test_design/03_testcase/multimodal/test_multimodal_p0_basic_npu.py
python3 -m coverage report -m
```

### Expected output

Each test prints a summary line, e.g.:

```
[P0-001] TTFT=3.427s  output_len=87
[P0-002] TTFT1=3.512s  TTFT2=1.823s  ratio=0.52
[P0-003] 10 text + 5 image all succeeded — no cross-pollution detected
[P0-004] output_len=143  prompt_tokens=1048
[P0-005/128x128] TTFT=2.101s  output_len=42
[P0-005/640x480] TTFT=3.334s  output_len=51
[P0-005/1920x1080] TTFT=8.912s  output_len=38
```

## File Location

```
test_design/
  01_plan/
    multimodal-interaction-test-analysis.md   # Design document
  03_testcase/
    multimodal/
      test_multimodal_p0_basic_npu.py         # Test implementation
      test_multimodal_p0_basic_npu.md         # This file
```
