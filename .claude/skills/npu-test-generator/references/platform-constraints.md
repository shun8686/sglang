# Platform Constraints Reference

Compiled constraints per platform. Update this doc when platform support changes — do NOT ad-hoc grep on each invocation.

## NPU / Ascend (A2, A3)

### Server Flags (required)
| Flag | Value | Notes |
|------|-------|-------|
| `--attention-backend` | `ascend` | **Required**. Server won't use NPU without it. |
| `--sampling-backend` | `ascend` | Recommended for NPU. |
| `--dtype` | `auto` / `float16` / `bfloat16` | |

### Server Flags (fixed defaults on NPU)
| Flag | NPU Default | CUDA Default | Notes |
|------|------------|--------------|-------|
| `--page-size` | `128` | `1` | NPU fixed to 128; cannot be changed. |
| `--disable-radix-cache` | — | — | Often set `True` for NPU accuracy tests. |
| `--disable-cuda-graph` | — | — | Sometimes set `True` for accuracy tests. |

### Speculative Decoding
| Constraint | Details |
|------------|---------|
| Supported algorithms | `EAGLE3`, `NEXTN` only. `STANDALONE`, `NGRAM`, `EAGLE` (EAGLE-2) **NOT** supported on NPU. |
| `--speculative-draft-attention-backend` | Set to `ascend` for NPU. |
| `--speculative-draft-model-quantization` | Typically `unquant` on NPU. |
| `--speculative-moe-a2a-backend` | `ascend_fuseep` (NPU-specific). |
| `--speculative-attention-mode` | `decode` recommended for NPU EAGLE3 tests. |
| Verification path | **Greedy only** on NPU (`verify_tree_greedy_func` from `sgl_kernel_npu`). No sampling-based tree verification. |
| Tree kernel | `torch.ops.npu.build_tree_kernel_efficient()` (NPU-specific). Not `sgl_kernel.build_tree_kernel_efficient`. |
| Cache loc dtype | `int32` on NPU (`torch.ops.npu.cache_loc_update`). CUDA uses `int64` with Triton kernels. |
| CUDA graph runners | `EAGLEDraftNpuGraphRunner`, `EAGLEDraftExtendNpuGraphRunner` — NPU-specific subclasses. |
| DP attention + EAGLE3 | Uses `draft_tp_context` when `--enable-dp-attention` and algorithm is EAGLE3. |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | `1` recommended for NPU overlap scheduling. |
| `SGLANG_ENABLE_SPEC_V2` | Optional; requires `--speculative-eagle-topk 1`. |

### CI Registration
| Constraint | Value |
|------------|-------|
| Registration function | `register_npu_ci` (from `sglang.test.ci.ci_register`) |
| Per-commit suite | `stage-b-test-1-npu-a2` |
| Nightly suite | `nightly-1-npu-a3` or similar |
| Import path | `from sglang.test.ci.ci_register import register_npu_ci` |

### Model Paths & Test Utilities
| Constraint | Value |
|------------|-------|
| Model path source | `sglang.test.ascend.test_ascend_utils` (NOT `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`) |
| Test base class | `CustomTestCase` |
| GSM8K accuracy mixin | `GSM8KAscendMixin` (NPU-specific) |

### File Placement
| Constraint | Value |
|------------|-------|
| Test directory | `test/registered/ascend/<category>/` |
| File naming | `test_npu_<feature>.py` |
| Categories | `basic_function/`, `interface/`, `llm_models/`, `vlm_models/`, `embedding_models/`, `reward_models/`, `rerank_models/` |

### Unsupported on NPU
- Ktransformer (`--kt-*` flags)
- `--checkpoint-engine-*` flags
- `--grpc-mode`
- FlashInfer autotune
- NVLS / symm mem
- `--speculative-accept-threshold-single` and `--speculative-accept-threshold-acc` (greedy only)

### Attention Backends
| Backend | NPU | Notes |
|---------|-----|-------|
| `AscendAttnBackend` | ✓ | Main NPU attention backend. |
| `AscendAttnMultiStepDraftBackend` | ✓ | Multi-step draft attention for EAGLE3. |
| `AscendHybridLinearAttnBackend` | ✓ | Hybrid linear attention. |

---

## CUDA / NVIDIA

### Server Flags
| Flag | Value | Notes |
|------|-------|-------|
| `--attention-backend` | `flashinfer` / `fa3` / `triton` | Various backends supported. |
| `--dtype` | `auto` / `float16` / `bfloat16` | |

### Speculative Decoding
| Constraint | Details |
|------------|---------|
| Supported algorithms | `EAGLE`, `EAGLE3`, `STANDALONE`, `NGRAM`, `NEXTN` — all supported. |
| Verification | Sampling-based tree verification via `sgl_kernel.tree_speculative_sampling_target_only`. |
| Tree kernel | `sgl_kernel.build_tree_kernel_efficient`. |
| Cache loc dtype | `int64`. Triton kernels (`assign_extend_cache_locs`, etc.). |
| CUDA graph runners | `EAGLEDraftCudaGraphRunner`, `EAGLEDraftExtendCudaGraphRunner`. |

### CI Registration
| Constraint | Value |
|------------|-------|
| Registration function | `register_cuda_ci` |
| Per-commit suites | `stage-b-test-1-gpu-small`, `stage-b-test-1-gpu-large` |
| Nightly suites | Various. |

### File Placement
| Constraint | Value |
|------------|-------|
| Test directory | `test/registered/<category>/` |
| File naming | `test_<feature>.py` |

---

## AMD / ROCm (MI300X, etc.)

### Server Flags
| Flag | Value | Notes |
|------|-------|-------|
| `--attention-backend` | `rocm` / `aiter` | ROCm-specific. |
| `--dtype` | `float16` / `bfloat16` | |

### Speculative Decoding
| Constraint | Details |
|------------|---------|
| Supported algorithms | `EAGLE3` (verified). Others TBD. |
| Verification | Greedy-only path (`verify_tree_greedy_func`). |
| Draft extend graph | `AiterMultiStepDraftBackend` required for draft extend CUDA graph. |

### CI Registration
| Constraint | Value |
|------------|-------|
| Registration function | `register_cuda_ci` (shared with CUDA on some setups) |

---

## How to Use This Reference

**Phase 1a-E:** Read this doc instead of ad-hoc grep. The constraint list is pre-compiled and verified.

**Phase 1b Step 5:** For each TP, cross-reference the relevant platform section above. Add platform-specific values to `boundary_conditions` and `error_paths`.

**Phase 1c subagent:** Pass the relevant platform section as the "platform constraint list" input. The subagent uses it for SC-3 (missing_platform_constraint).

**Phase 3:** Use the CI Registration and File Placement tables for correct test script generation.

**Updating:** When SGLang adds/changes platform support, update this doc. Do NOT supplement with ad-hoc grep — all constraints should live here.
