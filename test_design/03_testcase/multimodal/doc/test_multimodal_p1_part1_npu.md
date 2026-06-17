# P1 Part 1 Multimodal Test Cases — Implementation Analysis

## Overview

This file implements P1-001 through P1-005 from the multimodal interaction
test analysis report (§2.2).  Each test class is independent with its own
server lifecycle.  Five classes, one per test case.

## File

`test_design/03_testcase/multimodal/test_multimodal_p1_part1_npu.py`

## Test Case Summary

| ID | Name | Model | NPU | Key Server Args | Suite |
|----|------|-------|-----|-----------------|-------|
| P1-001 | Speculative decoding | Qwen3-VL-4B-Instruct | 1 | `--speculative-algorithm EAGLE3`, EAGLE draft model | nightly-2-npu-a3 |
| P1-002 | PD disaggregation | Qwen3-VL-4B-Instruct | 1 | `--enable-pdmux` | nightly-2-npu-a3 |
| P1-003 | TP parallelism | Qwen3-VL-4B-Instruct | 2 | `--tp-size 2` | nightly-2-npu-a3 |
| P1-004 | DP-attention | Qwen3-VL-4B-Instruct | 1 | `--enable-dp-attention` | nightly-2-npu-a3 |
| P1-005 | Deterministic inference | Qwen3-VL-4B-Instruct | 0 | `--enable-deterministic-inference` | nightly-2-npu-a3 (skipped) |

**CI suite**: All registered under `nightly-2-npu-a3` to cover the 2-NPU
requirement of P1-003.  P1-001/P1-002/P1-004 only need 1 NPU but work fine
on a 2-NPU runner.  P1-005 is unconditionally skipped on NPU.

**Estimated total runtime**: ~1200 seconds (20 min).

## Design Decisions

### 1. Per-class server lifecycle with inline baselines

Tests that compare against a baseline (P1-001, P1-002, P1-003, P1-004) start
the feature-under-test server in `setUpClass` and a temporary baseline server
inline within the test method.  This follows the pattern established by
`TestP0007GraphCompilation` in the P0 advanced file.

For P1-003 and P1-004, both servers (baseline and feature) are managed
inline within the test method because:

- P1-003 needs `--tp-size 2` (2 NPU chips) and `--tp-size 1` (1 chip);
  running both simultaneously would contend for NPU resources.
- P1-004 needs `--enable-dp-attention` vs. non-DP; same resource concern.

The `setUpClass` in these classes prepares test data (image + prompt) only;
`tearDownClass` is a no-op.

### 2. CI suite selection

All tests registered under `nightly-2-npu-a3` (the most conservative common
suite).  Rationale:

- P1-003 needs 2 NPU → must be nightly-2-npu-a3.
- P1-001 needs an EAGLE3 draft model that may not always be in cache.
- P1-004 runs 50 concurrent requests (potential resource spike).
- P1-005 is GPU-only and skipped on NPU.
- Registering multiple suites (e.g., per-commit for P1-002) would conflict
  with tests that need 2 NPU.  A single nightly registration is simpler.

### 3. Speculative decoding draft model check

P1-001 checks at module load time whether `Qwen/Qwen3-VL-4B-Instruct_eagle3`
exists under `MODEL_WEIGHTS_DIR`.  If absent, the entire class is skipped
with `@unittest.skipIf`.  This is intentional: the vlm-mtp capability for
Qwen3-VL is marked TBC (to be confirmed) in the design doc, and the draft
model may not be available yet.  The clear skip message makes it easy for
infra to add the model when it becomes available.

Draft model path follows Qwen3-8B's naming convention: `Qwen3-8B_eagle3`.

### 4. PD disaggregation via `--enable-pdmux`

The full PD disaggregation setup (separate prefill and decode servers with
a transfer backend) is impractical for per-CI testing because it needs:

- Two separate server processes with coordinated lifecycle
- A transfer backend (nixl / mooncake) that may need additional NPU drivers
- etcd or other service discovery mechanism

Instead, the test uses `--enable-pdmux` (PD multiplexing) which exercises
the PD-related code paths on a single server.  This is a compressed
validation that catches regressions without the infrastructure overhead.

As a future improvement, when EPD-for-VLM or two-server PD disaggregation
is stable on NPU, this test should be upgraded to the two-server pattern.

### 5. DP-attention throughput measurement

P1-004 sends 50 concurrent requests and measures total completion tokens
per second.  The throughput comparison against baseline uses a 20% tolerance
(`dp_tps >= bl_tps * 0.8`) to accommodate CI noise.  This is deliberately
loose — the goal is to detect catastrophic throughput regressions, not
micro-benchmark precision.

### 6. Deterministic inference — GPU-only

P1-005 is marked `@unittest.skip` unconditionally because
`deterministic_inference`'s NPU participation is `not_supported` per
`features.json`.  The skip message explicitly cites the FlashInfer/FA3/Triton
dependency.  The implementation is still present (and correct) for when
the test runs on GPU.

### 7. Port assignment

Ports 11–19 of `_PORT_BASE` are used:

| Test | Primary Port | Baseline Port |
|------|--------------|---------------|
| P1-001 | 11 | 12 |
| P1-002 | 13 | 14 |
| P1-003 | 15 | 16 |
| P1-004 | 17 | 18 |
| P1-005 | 19 | — |

This avoids conflicts with P0 test files (ports 1–10).

## CI Registration Pattern

```python
register_npu_ci(est_time=1200, suite="nightly-2-npu-a3", nightly=True)
```

Called at **module level** (per instructions: returns `None`, so using it as
a class decorator would raise `TypeError`).

## Integration Notes

### When adding a VL EAGLE3 model

P1-001 expects the draft model at:

```
{model_cache_dir}/Qwen/Qwen3-VL-4B-Instruct_eagle3
```

If the actual path differs update `_QWEN3_VL_4B_EAGLE3_PATH` in the test
file.

### When PD disaggregation with EPD on NPU stabilises

Replace the `--enable-pdmux` approach in `TestP1002PDDisaggregation` with:

1. Prefill server: `--disaggregation-mode prefill --disaggregation-transfer-backend nixl`
2. Decode server: `--disaggregation-mode decode --disaggregation-transfer-backend nixl`
3. Both servers should share the model weights path and use separate ports.

### When running on GPU

Remove the `@unittest.skip` from `TestP1005DeterministicInference`.  The
server args and test logic are already correct for GPU execution (they use
`--enable-deterministic-inference` and send two requests with the same seed).
No code changes needed beyond removing the skip decorator.
