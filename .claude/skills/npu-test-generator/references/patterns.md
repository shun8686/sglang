# SGLang Test Conventions — NPU (Ascend)

Extracted from real NPU test files under `test/registered/ascend/`. Reference when generating Phase 3 scripts.

---

## NPU (Ascend) — Test Conventions

### Base Classes

| Test Type | Base Class | When |
|---|---|---|
| Server E2E | `sglang.test.test_utils.CustomTestCase` | Test needs a running NPU server |
| Unit (no server) | `unittest.TestCase` | Logic test, no NPU needed |

### CI Registration (NPU)

```python
from sglang.test.ci.ci_register import register_npu_ci

# Per-commit (small NPU, runs on every PR)
register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)

# Nightly (larger NPU configs)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)
register_npu_ci(est_time=500, suite="nightly-8-npu-a3", nightly=True)
register_npu_ci(est_time=600, suite="nightly-16-npu-a3", nightly=True)

# Full/merged suite
register_npu_ci(est_time=1600, suite="nightly-npu-a3-merged", nightly=True)
register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="full-4-npu-a3", nightly=True)
```

### Available NPU CI Suites

| Suite | Runner | Frequency |
|---|---|---|
| `stage-b-test-1-npu-a2` | `linux-aarch64-a2-1` | Per-commit |
| `nightly-1-npu-a3` | `linux-aarch64-a3-2` | Nightly |
| `nightly-2-npu-a3` | `linux-aarch64-a3-2` | Nightly |
| `nightly-4-npu-a3` | `linux-aarch64-a3-4` | Nightly |
| `nightly-8-npu-a3` | `linux-aarch64-a3-16` | Nightly |
| `nightly-16-npu-a3` | `linux-aarch64-a3-16` | Nightly |
| `nightly-npu-a3-merged` | varies | Nightly |
| `full-1-npu-a3` | varies | Nightly |
| `full-4-npu-a3` | varies | Nightly |

### Server Fixtures (NPU E2E tests)

```python
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_30B_A3B_WEIGHTS_PATH,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNpuFeature(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend", "ascend",    # REQUIRED for NPU
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_basic(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Hello, world",
                "sampling_params": {"max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
```

### NPU Environment Variables

When the test needs NPU-specific env:
```python
import os

class TestNpuFeature(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Set NPU visible devices if not already set
        if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

        cls.npu_env = {**os.environ}
        # Feature-specific env vars
        cls.npu_env["ASCEND_USE_FIA"] = "1"
        # ... launch server with env=cls.npu_env
```

### NPU Test Directory Structure

```
test/registered/ascend/
  basic_function/
    backends/                  # test_npu_sampling_backend.py
    speculative_inference/     # test_npu_eagle3.py
    parallel_strategy/         # expert_parallelism, ...
    offloading/                # test_npu_offload_modes.py
    dllm/                      # test_npu_llada2_mini.py
    optimization_debug/        # test_npu_piecewise_graph_prefill.py
  interface/                   # test_npu_api.py, test_npu_matched_stop.py
  llm_models/                  # test_npu_qwen3_30b_attn_cp.py, ...
  vlm_models/                  # test_npu_qwen3_vl_8b_instruct.py, ...
  embedding_models/            # test_npu_bge_large_en_v1_5.py
  reward_models/               # test_npu_llama_3_1_8b_v0_2.py, ...
  rerank_models/               # test_npu_bge_reranker_v2_m3.py
  test_npu_memory_consumption.py
```

### NPU File Naming Convention

```
test_npu_<feature_name>.py     # All NPU test files use this prefix
```

### Per-Commit vs Nightly for NPU

Unlike CUDA which has many per-commit suites, NPU has ONE per-commit suite:
- `stage-b-test-1-npu-a2` — runs on every PR

All other NPU suites are nightly-only. New NPU tests should prefer `stage-b-test-1-npu-a2` when possible, adding `nightly-*-npu-a3` registrations for heavier configs.

### NPU Model Path Constants

Available from `sglang.test.ascend.test_ascend_utils`:
- `LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH` — small model for basic tests
- `LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH` — medium model
- `QWEN3_30B_A3B_WEIGHTS_PATH` — large model (MoE)
- Others: `BGE_RERANKER_V2_M3_WEIGHTS_PATH`, `INTERNLM2_7B_REWARD_WEIGHTS_PATH`, `DEEPSEEK_VL2_WEIGHTS_PATH`, `MINIMAX_M2_WEIGHTS_PATH`

---

## General Patterns (all backends)

### Assertion Patterns

```python
# Equality
self.assertEqual(result, expected_value)

# Exception
with self.assertRaises(ValueError):
    bad_function()

# Tensor
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

# HTTP
response = requests.post(url, json=payload)
self.assertEqual(response.status_code, 200)
self.assertIn("expected_key", response.json())
```

### Oracle Verification Patterns

The oracle verification pattern catalog. Each entry records a detection method discovered during test generation. When a new method is found during a session, add an entry following the template below.

**Entry template:**

```
### <method name>
- **来源**: <which session / feature discovered it>
- **检测对象**: <what it verifies: config, runtime behavior, performance, correctness>
- **确定性**: 高（API 结构字段）/ 中（日志关键字匹配）/ 低（输出比较）
- **适用场景**: <when to use>
- **不适用场景**: <when NOT to use>
- **Phase 2 模板** (step + expected_result):
- **Phase 3 模板** (Python code):
```

**Decision flow (top-to-bottom, first match wins):**

```
 1. Flag is a CLI argument (--flag, stored in ServerArgs)? → /server_info
 2. Flag is an env var with startup log message?       → log capture (keyword)
 3. Verifying model identity / architecture / type?    → /model_info
 4. Feature affects numerical output (logprobs,        → numerical vs HF
    embeddings, scores)?                                 (torch.allclose)
 5. Feature must NOT degrade output accuracy?          → GSM8K / MMLU accuracy
 6. Verifying exact token count / max_new_tokens?      → token count (meta_info)
 7. Verifying deterministic execution (temp=0)?        → temperature determinism
 8. Feature produces a metric (accept_rate, ttft)?     → /metrics or log capture
 9. Verifying semantic content in output?              → content assertion (assertIn)
10. Feature affects throughput/latency?                → throughput measurement
11. Verifying KV cache behavior (hit/miss, memory)?    → cache / memory verification
12. None of the above — no observable signal           → skip / mark as inherent limitation
```

---

#### /server_info — deterministic config verification

- **来源**: `test_npu_warmups.py` (lines 67-69); also discovered independently during EAGLE3 Phase 2 oracle_gap remediation
- **检测对象**: server CLI flags (configuration)
- **确定性**: 高 — `GET /server_info` returns `dataclasses.asdict(server_args)` as JSON
- **适用范围**: 仅限 CLI flag（存于 `ServerArgs` 的配置参数）。验证的是"服务器认为自己的配置是什么"，不是"运行时真的用了什么"。不覆盖：环境变量（不在 ServerArgs）、请求级参数（sampling params）、运行时行为（backend 是否真的走了某条路径）、模型身份（用 `/model_info`）、输出质量（用内容断言/GSM8K/HF 对比）、性能（用吞吐测量）。

Phase 3 pattern (confirmed by `test_npu_warmups.py`):
```python
r = requests.get(f"{url}/server_info")
info = r.json()
self.assertEqual(info["warmups"], "voice_chat")
self.assertEqual(info["speculative_eagle_topk"], 4)
self.assertEqual(info["disable_radix_cache"], True)
```

---

#### log capture — server stdout/stderr keyword matching

- **来源**: `test_npu_log_level.py` (lines 30-63), `test_npu_warmups.py` (lines 72-74)
- **检测对象**: server log messages (runtime behavior, initialization, HTTP request logging)
- **确定性**: 中 — keyword match depends on log format, log level, and message content
- **适用场景**: Verifying log level affects output (`--log-level-http`). Verifying warmup procedures ran. Environment variables whose activation produces unique log messages. Use **value-specific** patterns (e.g. `overlap_plan_stream` not just `overlap`). Use `assertNotIn` for negative checks (log suppressed).
- **不适用场景**: When `/server_info` is available (prefer it). When no unique log message exists for the flag.

Phase 3 pattern (confirmed by `test_npu_log_level.py`):
```python
# Capture stdout/stderr via popen_launch_server's return_stdout_stderr
out_log = open("./out_log.txt", "w+", encoding="utf-8")
err_log = open("./err_log.txt", "w+", encoding="utf-8")
process = popen_launch_server(
    model, DEFAULT_URL_FOR_TEST,
    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    other_args=other_args,
    return_stdout_stderr=(out_log, err_log),
)
# ... do inference ...
out_log.seek(0)
log_content = out_log.read()
self.assertIn("Running warmup voice_chat", log_content)
self.assertNotIn("POST /generate HTTP/1.1", log_content)
```

**Rule:** Always use the most specific pattern available. Prefer `assertNotIn` for verifying a log message is suppressed. Avoid `or` fallbacks to broad patterns that match unrelated log messages.

---

#### content assertion — semantic substring match in output

- **来源**: ~70% of NPU test files; canonical example `test_npu_api.py` (line 142)
- **检测对象**: output content correctness (semantic)
- **确定性**: 中 — depends on model determinism and prompt sensitivity
- **适用场景**: Verifying the model answers a factual question correctly. Checking that a specific piece of information appears in the output. The most common oracle in the NPU test suite.
- **不适用场景**: Verifying specific flag values (use `/server_info`). Numerical correctness (use HF comparison). Non-deterministic outputs (temp > 0).

Phase 3 pattern (confirmed by `test_npu_api.py`):
```python
response = requests.post(f"{url}/generate", json={
    "text": "What is the capital of France?",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
})
self.assertEqual(response.status_code, 200)
self.assertIn("Paris", response.json()["text"])
```

**Note:** Combine with `temperature=0` for deterministic outputs. For baseline comparison (feature vs non-feature), send the same prompt to both servers and compare — but note this has low determinism (different execution paths may produce different but equally valid completions).

---

#### throughput measurement — wall-clock tokens/sec

- **来源**: `test_npu_w8a8_quantization.py` (lines 78-103); also `bench_serving` pattern in `test_npu_no_chunked_prefill.py`
- **检测对象**: generation throughput (tokens/sec)
- **确定性**: 中 — noisy, depends on hardware thermal state and measurement timing
- **适用场景**: Verifying a feature provides measurable throughput improvement. Quantization throughput validation. Detecting performance regressions.
- **不适用场景**: Verifying specific flag values (use `/server_info`). Single-request latency tests (too noisy). Concurrent benchmark (use `run_bench_serving`).

Phase 3 pattern (confirmed by `test_npu_w8a8_quantization.py`):
```python
max_tokens = 256
response = requests.post(f"{url}/generate", json={
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
    "ignore_eos": True,
})
tic = time.perf_counter()
res = response.json()
tok = time.perf_counter()
throughput = max_tokens / (tok - tic)
self.assertGreaterEqual(throughput, 25)  # minimum tokens/sec threshold
```

**Note:** Use `ignore_eos=True` to ensure exact token count without early stopping. Warm up with a few requests before measurement. CI environments should use permissive lower bounds.

---

#### metrics endpoint — Prometheus /metrics verification

- **来源**: (placeholder — add when discovered)
- **检测对象**: runtime metrics (accept_rate, ttft, queue depth, ...)
- **确定性**: 高 — structured Prometheus output
- **适用场景**: Verifying speculative decoding metrics, queue metrics, cache hit rates.
- **不适用场景**: Configuration verification (use `/server_info`). Non-speculative features without dedicated metrics.

Phase 2 step template (tentative):
```
Step N: GET {base_url}/metrics → produces metrics_output
```

Phase 3 pattern (tentative):
```python
r = requests.get(f"{url}/metrics")
self.assertIn("sglang_accept_rate", r.text)
```

---

#### /model_info — model metadata verification

- **来源**: `test_npu_api.py` (lines 74-85)
- **检测对象**: model identity (path, architecture, type, capabilities)
- **确定性**: 高 — structured JSON with model_config fields
- **适用场景**: Verifying model was loaded correctly. Checking model type, architecture, weight version. Confirming model capabilities (e.g. `is_generation`, `has_image_understanding`).
- **不适用场景**: Runtime behavior verification. Performance measurement.

Phase 3 pattern:
```python
r = requests.get(f"{url}/model_info")
info = r.json()
self.assertEqual(info["model_path"], model)
self.assertEqual(info["model_type"], "llama")
self.assertEqual(info["architectures"][0], "LlamaForCausalLM")
self.assertTrue(info["is_generation"])
```

---

#### token count — meta_info.completion_tokens / usage

- **来源**: `test_npu_api.py` (line 143), `test_npu_fim_completion.py` (lines 85-88)
- **检测对象**: output token count matches expectation
- **确定性**: 高 — exact integer match
- **适用场景**: Verifying exact generation length. Checking max_new_tokens enforcement. Validating prompt/response token accounting.
- **不适用场景**: Content quality verification. Semantic correctness.

Phase 3 pattern:
```python
r = requests.post(f"{url}/generate", json={
    "text": "What is the capital of France?",
    "sampling_params": {"max_new_tokens": 20, "temperature": 0},
})
self.assertEqual(20, r.json()["meta_info"]["completion_tokens"])
self.assertGreater(r.json()["meta_info"]["prompt_tokens"], 0)
```

---

#### temperature determinism — identical output with temp=0

- **来源**: `test_npu_api.py` (lines 174-199), `test_npu_sampling_backend.py` (lines 56-96)
- **检测对象**: deterministic generation under greedy sampling
- **确定性**: 高 — exact string match for temp=0
- **适用场景**: Verifying deterministic execution. Detecting non-deterministic kernel bugs. Batch vs single-request consistency.
- **不适用场景**: Non-greedy sampling (temp > 0). Random or creative output testing.

Phase 3 pattern:
```python
# temp=0: same prompt same output (determinism)
r1 = requests.post(f"{url}/generate", json={
    "text": "The capital of Germany is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 10},
})
r2 = requests.post(f"{url}/generate", json={
    "text": "The capital of Germany is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 10},
})
self.assertEqual(r1.json()["text"], r2.json()["text"])
```

---

#### GSM8K / MMLU accuracy — benchmark-based correctness verification

- **来源**: `test_npu_tp1_bf16.py` (lines 43-74), `test_npu_w8a8_quantization.py` (lines 60-76), `test_npu_sampling_backend.py` (lines 43-54)
- **检测对象**: model output accuracy against a standardized benchmark
- **确定性**: 中 — statistical, depends on benchmark size and prompt sensitivity
- **适用场景**: Verifying speculative decoding does NOT degrade accuracy below a threshold. Quantization accuracy validation. TP/PP configuration validation.
- **不适用场景**: Unit tests (too slow — 200-1319 questions). Feature flag verification (use `/server_info`). Quick smoke tests.

Phase 3 pattern:
```python
from sglang.bench_offline_throughput_v2 import run_eval
args = SimpleNamespace(
    num_shots=5, num_questions=200, max_new_tokens=512,
    parallel=128, host=f"http://{host}", port=port,
)
metrics = run_eval(args)
self.assertGreaterEqual(metrics["accuracy"], 0.84)
```

**Note:** GSM8K tests typically take 5-15 minutes with 200-1319 questions. Only use for end-to-end feature validation, not per-commit regression testing.

---

#### numerical correctness vs Hugging Face — logprob / embedding / score comparison

- **来源**: `test_npu_original_logprobs.py`, `test_npu_bge_large_en_v1_5.py`, `test_npu_bge_reranker_v2_m3.py`, `test_npu_llama_3_1_8b_v0_2.py`
- **检测对象**: numerical equivalence between SGLang and Hugging Face reference
- **确定性**: 高 — `torch.allclose` with defined tolerance
- **适用场景**: Logprob accuracy verification. Embedding model correctness. Reranker/reward model score validation.
- **不适用场景**: Text generation tests (output token sequences diverge even with correct logprobs). Configuration verification.

Phase 3 pattern:
```python
with HFRunner(model_path, model_type="embedding") as hf_runner:
    hf_outputs = hf_runner.forward(prompts)
with SRTRunner(model_path, model_type="embedding",
               attention_backend="ascend") as srt_runner:
    srt_outputs = srt_runner.forward(prompts)

for i in range(len(prompts)):
    hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
    srt_logits = torch.Tensor(srt_outputs.embed_logits[i])
    similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
    self.assertTrue(torch.all(abs(similarity - 1) < 1e-5))
```

---

#### cache / memory verification — NPU memory bounds and cache hit behavior

- **来源**: `test_npu_radix_cache.py` (lines 61-128), `test_npu_memory_consumption.py` (lines 42-73)
- **检测对象**: KV cache reuse, memory consumption within bounds
- **确定性**: 高 — exact integer for token counts, floating-point for memory
- **适用场景**: Verifying radix/hierarchical cache hits. Checking memory consumption stays within budget. Validating cache flush works.
- **不适用场景**: Performance benchmarking (use `run_bench_serving`).

Phase 3 pattern:
```python
# Cache hit verification
r1 = requests.post(f"{url}/generate", json={
    "text": long_text, "sampling_params": {"max_new_tokens": 1},
})
self.assertEqual(r1.json()["meta_info"]["cached_tokens"], 0)
r2 = requests.post(f"{url}/generate", json={
    "text": long_text, "sampling_params": {"max_new_tokens": 1},
})
self.assertGreater(r2.json()["meta_info"]["cached_tokens"], 0)

# Memory bound verification
free, total = torch.npu.mem_get_info()
used_gb = (total - free - initial_used) / (1 << 30)
self.assertLessEqual(used_gb, 17.00)
```

### File Boilerplate

```python
import unittest

# ... test class ...

if __name__ == "__main__":
    unittest.main()
```
