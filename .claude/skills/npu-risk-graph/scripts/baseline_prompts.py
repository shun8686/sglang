#!/usr/bin/env python3
"""Agent prompt templates for NPU baseline test-to-feature mapping."""

# ============================================================
# Agent prompt: map test files to features
# ============================================================

AGENT_TEST_MAP_PROMPT = """You are an NPU (Huawei Ascend) test coverage analyzer. Your task is to read NPU test files and determine which platform features they exercise.

Given a batch of test files and the complete list of {feature_count} NPU features, produce a mapping from each test file to the features it actually tests.

## Key Rules

- Be PRECISE: only list features the test file demonstrably exercises. Do NOT include features just because the test file uses a shared utility or imports a common module.
- "attention_backend" should ONLY be mapped when the test specifically targets attention mechanics (MLA, GDN, mamba, hybrid linear, torch native, NSA, sinks). Do NOT include attention_backend just because --attention-backend ascend appears in other_args — ALL NPU tests use this flag.
- Distinguish carefully between related features:
  - quantization vs moe: check --quantization flag vs --moe-a2a-backend / --enable-expert-parallelism
  - moe vs expert_parallelism: moe = expert computation (fused_moe, topk routing), expert_parallelism = expert distribution across ranks (DeepEP, FuseEP, token dispatcher)
  - hicache vs radix_cache: --enable-hierarchical-cache = HiCache active; --disable-radix-cache does NOT mean HiCache active
  - graph_compilation vs speculative_decoding: both use NPU graphs but for different purposes
- Read the ENTIRE file content. Key signals are in:
  - Class docstring [Test Category] and [Test Target] tags
  - cls.model or TEST_MODEL_MATRIX model paths
  - other_args list (server command-line flags)
  - Mixin base classes (GSM8KAscendMixin = model accuracy test)
  - Environment variable assignments
  - Test method names

## Available Features ({feature_count} total)

{feature_table}

## Quality Score Guide

| Score | Description |
|-------|-------------|
| 1 | Basic smoke: launches server, runs 1-3 prompts, checks response status only |
| 2 | Threshold accuracy: has assertGreaterEqual on accuracy or throughput with a numeric threshold |
| 3 | Multi-config: tests multiple models/configs via TEST_MODEL_MATRIX, parametrize, or subTest |
| 4 | Reference oracle: has GSM8KAscendMixin, run_eval_few_shot_gsm8k, or compares against reference output |
| 5 | Full suite: oracle + multi-config + additional validation (fuzz, stress, boundary, logprobs check) |

## Test Files in This Batch ({batch_size} files)

For each file below, read the FULL CONTENT and determine:
1. Which features from the table above this file tests (1-5 features)
2. Quality score following the guide above
3. Whether it has a GSM8K accuracy oracle
4. The assertion type (threshold / reference_comparison / mixed)

{test_files_section}

## Output

Return a JSON object with:
- "batch_id": {batch_id}
- "mappings": array of objects, one per test file, each with:
  - test_file: repo-relative path (exactly as provided)
  - features_tested: list of feature names (from the table above)
  - quality_score: integer 1-5
  - has_gsm8k_oracle: boolean
  - assertion_type: one of "threshold", "reference_comparison", "mixed"
  - has_reference_oracle: boolean
  - rationale: one-sentence explanation (max 200 chars)

Return ONLY valid JSON matching this schema — no preamble, no markdown fences."""


def _extract_signals(feature: dict) -> list[str]:
    """Extract rich key signals from a feature definition for Agent matching.

    Signals are drawn from:
      - NPU-specific source files (basenames without _npu / ascend_ prefix noise)
      - Keywords from description (flag names, operator names, technology acronyms)
      - Participation level (only if not_supported — Agent should avoid mapping)
    """
    name = feature["name"]
    category = feature.get("category", "?")
    desc = feature.get("description", "") or ""
    source_files = feature.get("source_files", [])
    npu = feature.get("npu_participation", "")

    signals = []

    # Category always first
    signals.append(category)

    # Extract NPU-specific file basenames (strip common prefixes, keep distinctive parts)
    npu_files = [f for f in source_files if "hardware_backend/npu" in f]
    for f in npu_files:
        basename = f.rsplit("/", 1)[-1].replace(".py", "")
        # Strip common noise prefixes
        for prefix in ["ascend_", "npu_"]:
            if basename.startswith(prefix):
                basename = basename[len(prefix) :]
        if basename and basename not in ("__init__", "utils"):
            signals.append(basename)

    # Extract keywords from description: flag names (--xxx), env vars (SGLANG_*), tech terms
    import re

    # Server flags like --enable-hierarchical-cache, --moe-a2a-backend
    flags = re.findall(r"--([\w-]+)", desc)
    for flag in flags[:4]:  # cap to avoid bloat
        # Shorten common long flags
        short = flag.replace("enable-", "").replace("speculative-", "spec-")
        signals.append(short)

    # Env vars
    envs = re.findall(r"SGLANG_[\w_]+", desc)
    for env in envs[:2]:
        signals.append(env)

    # Technology keywords from description
    tech_keywords = {
        "fia": ["FIA"],
        "gdn": ["GDN"],
        "mamba": ["Mamba", "SSM"],
        "mla": ["MLA"],
        "nsa": ["NSA"],
        "dsa": ["DSA"],
        "eagle": ["EAGLE"],
        "deepep": ["DeepEP"],
        "fuseep": ["FuseEP"],
        "lora": ["LoRA"],
        "awq": ["AWQ"],
        "gptq": ["GPTQ"],
        "fp8": ["FP8"],
        "fp4": ["FP4"],
        "w8a8": ["W8A8"],
        "w4a4": ["W4A4"],
        "int4": ["INT4"],
        "cmo": ["CMO"],
        "moe": ["MoE"],
        "ep": ["EP"],
        "tp": ["TP"],
        "dp": ["DP"],
        "cp": ["CP"],
        "pp": ["PP"],
        "mooncake": ["Mooncake"],
        "rdma": ["RDMA"],
        "epd": ["EPD"],
        "vlm": ["VLM"],
        "vit": ["ViT"],
        "kv": ["KV cache"],
        "zbal": ["ZBAL"],
        "sgmv": ["SGMV"],
        "eplb": ["EPLB"],
        "xgrammar": ["XGrammar"],
        "radix": ["Radix"],
    }
    desc_lower = desc.lower()
    for key, labels in tech_keywords.items():
        if key in desc_lower:
            signals.extend(labels[:2])

    # Not_supported marker
    if npu == "not_supported":
        signals.append("NOT_SUPPORTED_ON_NPU")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in signals:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    return unique[:12]  # cap total signals


def render_feature_table(features: list[dict]) -> str:
    """Render the feature reference table for the Agent prompt.

    Args:
        features: list of feature dicts from features.json.

    Returns:
        Markdown-formatted table string.
    """
    header = "| Feature | Category | Key Signals | Description |"
    sep = "|---------|----------|-------------|-------------|"
    rows = []
    for f in features:
        name = f["name"]
        category = f.get("category", "?")
        desc = (f.get("description", "") or "")[:120]
        signals = _extract_signals(f)
        signals_str = ", ".join(signals)
        rows.append(f"| {name} | {category} | {signals_str} | {desc} |")

    return "\n".join([header, sep] + rows)


def build_batch_prompt(batch: dict, features: list[dict], batch_id: int) -> str:
    """Build the full Agent prompt for a single batch of test files.

    Args:
        batch: dict with 'test_files' list, each having path, class_names,
               base_classes, model, other_args, env_vars, registrations,
               test_methods, docstring_tags, content.
        features: list of 56 feature dicts.
        batch_id: integer batch index.

    Returns:
        Complete prompt string for an agent() call.
    """
    feature_table = render_feature_table(features)

    # Build test files section
    test_sections = []
    for i, tf in enumerate(batch["test_files"]):
        path = tf["path"]
        content = tf.get("content", "")
        # Summary of extracted context
        context_lines = []
        if tf.get("class_names"):
            context_lines.append(f"  Class: {', '.join(tf['class_names'])}")
        if tf.get("base_classes"):
            context_lines.append(f"  Mixins: {', '.join(tf['base_classes'])}")
        if tf.get("model"):
            context_lines.append(f"  Model: {tf['model']}")
        if tf.get("other_args"):
            context_lines.append(f"  Server args: {' '.join(tf['other_args'])}")
        if tf.get("env_vars"):
            context_lines.append(f"  Env vars: {tf['env_vars']}")
        if tf.get("docstring_tags"):
            context_lines.append(f"  Tags: {tf['docstring_tags']}")
        if tf.get("test_methods"):
            context_lines.append(f"  Methods: {', '.join(tf['test_methods'])}")

        context_str = (
            "\n".join(context_lines) if context_lines else "(no context extracted)"
        )

        test_sections.append(f"""### File {i+1}: {path}

**Extracted Context:**
{context_str}

**Full Content:**
```python
{content}
```""")

    test_files_section = "\n\n".join(test_sections)

    return AGENT_TEST_MAP_PROMPT.format(
        feature_count=len(features),
        feature_table=feature_table,
        batch_size=len(batch["test_files"]),
        batch_id=batch_id,
        test_files_section=test_files_section,
    )
