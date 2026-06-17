# Advanced NPU Multimodal P0 Test Cases (P0-006 through P0-010)

## Implemented Test Cases

### P0-006: Long text + image -> chunked prefill
- **Class**: `TestP0006ChunkedPrefill`
- **Model**: Qwen3-VL-4B-Instruct (`QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH`)
- **Server args**: `--device npu --attention-backend ascend --trust-remote-code --enable-multimodal --chunked-prefill-size 512`
- **Input**: ~3K token Chinese prefix + synthetic test image + "总结以上"
- **Verification**: Output references both the long prefix text and the image content
- **Port**: 30006 (base 30000 + 6)

### P0-007: Graph compilation + multimodal inference
- **Class**: `TestP0007GraphCompilation`
- **Model**: Qwen3-VL-4B-Instruct (`QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH`)
- **Two servers** (started sequentially within the test method):
  1. Baseline: with `--disable-cuda-graph` (port 30071)
  2. Graph-enabled: WITHOUT `--disable-cuda-graph` (port 30072)
- **Input**: Same synthetic test image + "描述图片" for both
- **Verification**: Both outputs reference image content; graph-enabled output is not empty/truncated; no crash
- **Ports**: 30071 and 30072

### P0-008: LoRA adapter + multimodal
- **Class**: `TestP0008LoRA`
- **Model**: Qwen3-VL-4B-Instruct with LoRA adapter
- **LoRA path**: `QWEN3_VL_LORA_WEIGHTS_PATH` (`.../Qwen/Qwen3-VL-4B-Instruct-LoRA`)
- **Skip condition**: LoRA weights not found at the expected path
- **Server args**: `--lora-paths test-lora=<path>`
- **Input**: Synthetic test image + "描述图片" (request uses `model="test-lora"`)
- **Verification**: Output references image content (LoRA has not broken ViT encoding)
- **Port**: 30008

### P0-009: GDN linear attention + visual encoder
- **Class**: `TestP0009GDNLinearAttention`
- **Model**: Qwen3.5-MoE (`QWEN3_5_MOE_VL_WEIGHTS_PATH`: `.../Qwen/Qwen3.5-MoE-VL`)
- **Skip condition**: Qwen3.5-MoE model not found in cache (official support marked as `[X]`)
- **Server args**: Standard NPU args (no special flags)
- **Input**: Synthetic test image + "描述这张图片"
- **Verification**: Output references image content; no NaN in output (GDN state corruption check)
- **Port**: 30009

### P0-010: GDN + MoE + visual encoder
- **Class**: `TestP0010GDNMoE`
- **Model**: Qwen3.5-MoE (`QWEN3_5_MOE_VL_WEIGHTS_PATH`)
- **Skip condition**: Same as P0-009
- **Server args**: Standard NPU args (no special flags)
- **Input**: Synthetic test image + "描述这张图片"
- **Verification**: Output references image content; no NaN; basic expert routing sanity check (vocabulary diversity ratio > 0.3)
- **Port**: 30010

## Models Required

| Test Case | Model | Path Constant | Availability |
|-----------|-------|---------------|--------------|
| P0-006, P0-007 | Qwen3-VL-4B-Instruct | `QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH` | Always (in Modelscope cache) |
| P0-008 | Qwen3-VL-4B-Instruct + LoRA | `QWEN3_VL_LORA_WEIGHTS_PATH` | May NOT be available |
| P0-009, P0-010 | Qwen3.5-MoE-VL | `QWEN3_5_MOE_VL_WEIGHTS_PATH` | May NOT be available |

## Skip Conditions

1. **P0-008** skips with `unittest.SkipTest` if LoRA weights are not found at `QWEN3_VL_LORA_WEIGHTS_PATH`. The LoRA adapter path is defined as `.../Qwen/Qwen3-VL-4B-Instruct-LoRA`. If the actual path differs in the target environment, edit the constant at the top of the test file.

2. **P0-009 and P0-010** skip with `unittest.SkipTest` if Qwen3.5-MoE model weights are not found. The model path is defined as `.../Qwen/Qwen3.5-MoE-VL`. Since Qwen3.5-MoE official support is marked as `[X]` (not supported), these tests are expected to skip in most NPU environments until the model is provisioned.

## How to Run

**Run all test cases:**
```bash
python3 -m unittest test_design/03_testcase/multimodal/test_multimodal_p0_advanced_npu.py
```

**Run a single test class:**
```bash
python3 -m unittest test_design/03_testcase/multimodal/test_multimodal_p0_advanced_npu.TestP0006ChunkedPrefill
python3 -m unittest test_design/03_testcase/multimodal/test_multimodal_p0_advanced_npu.TestP0007GraphCompilation
python3 -m unittest test_design/03_testcase/multimodal/test_multimodal_p0_advanced_npu.TestP0008LoRA
python3 -m unittest test_design/03_testcase/multimodal/test_multimodal_p0_advanced_npu.TestP0009GDNLinearAttention
python3 -m unittest test_design/03_testcase/multimodal/test_multimodal_p0_advanced_npu.TestP0010GDNMoE
```

**Run from the registered test directory:**
```bash
python3 -m unittest test_design.03_testcase.multimodal.test_multimodal_p0_advanced_npu.TestP0006ChunkedPrefill
```

## Port Allocation

All test ports offset from `_PORT_BASE = 30000 + device_index * 100`:

| Test Class | Port |
|------------|------|
| P0-006     | 30006 |
| P0-007 baseline | 30071 |
| P0-007 graph | 30072 |
| P0-008     | 30008 |
| P0-009     | 30009 |
| P0-010     | 30010 |

## Design Decisions

1. **Synthetic images**: All tests generate images programmatically using PIL, removing any dependency on external image URLs or files.

2. **Sequential server lifecycle in P0-007**: The two servers for graph comparison run sequentially (start-server-A / test / kill / start-server-B / test / compare) to avoid resource contention on a single NPU. This is safe because each run is independent.

3. **LoRA request semantics**: The LoRA adapter is selected per-request by passing `model="test-lora"` (matching the name in `--lora-paths`). The base model loaded by the server remains the Qwen3-VL-4B-Instruct.

4. **Vocabulary diversity heuristic for P0-010**: A simple unique-word ratio check serves as a basic expert routing sanity check. Abnormally low diversity (ratio < 0.3) can indicate expert routing collapse where all tokens route to the same expert, producing repetitive output.

5. **CI registration**: Registered as `suite="nightly-1-npu-a3" nightly=True` at module level (single registration for the entire file). Estimated runtime 2100 s due to sequential server launches across 5 independent test classes. Using nightly suite because multiple sequential server startups exceed per-commit time budgets.

## Known Limitations

- The LoRA adapter name/path is hardcoded and may need adjustment per environment.
- The vocabulary diversity threshold (0.3) is an approximation and may need tuning.
- GDN model availability is the primary blocker for P0-009 and P0-010.
