---
name: test-generator
description: >
  Three-phase closed-loop test generation pipeline with self-correction based on
  the CyberCorrect framework. Phase 1: extract test points from design docs + source code.
  Phase 2: generate detailed text test cases from test points.
  Phase 3: generate executable test scripts (pytest/unittest) from text cases.
  Each phase auto-detects errors, applies type-directed corrections, and checks convergence.
  Supports SGLang test conventions for CUDA (NVIDIA), NPU (Ascend), and AMD (ROCm) backends.
trigger: /test-gen
---

# /test-gen — CyberCorrect-Style Test Generation Pipeline

Three-phase closed-loop: source code + design docs → test points → text test cases → executable scripts.
Each phase: **Generate → Detect → Fix → Check Convergence** (repeat until converged or exhausted).

## Usage

```
/test-gen <path>                        # Full pipeline
/test-gen <path> --platform npu|cuda    # Target platform (default: cuda)
/test-gen <path> --phase 1|2|3          # Single phase only
/test-gen <path> --max-iter N           # Override max iterations
```

## Step 0 — Parse Input & Load References

### Parse

- `TARGET_PATH`: verify exists, stop if not
- `PLATFORM`: `npu` if `--platform npu`, `cuda` if `--platform cuda`, otherwise infer:
  - Path contains `ascend` → `npu`
  - Path under `python/sglang/` → `cuda`
  - Ambiguous → ask
- `PHASE`: `1`, `2`, `3`, or `all` (default)
- `MAX_ITER`: from `--max-iter`, defaults: P1=4, P2=5, P3=5

### Load References (read immediately — small, needed for every phase)

**SKILL_BASE**: The absolute path to this skill's directory. Compute it as the directory containing this SKILL.md file. All reference files and scripts live under SKILL_BASE. Subagent prompts use `<skill-base>` as a placeholder that MUST be replaced with SKILL_BASE before sending.

| File | Content |
|------|---------|
| `references/error-types.md` | Error taxonomy, severity weights, and τ→action lookup table |
| `references/convergence.md` | C1-C5 algorithm, diagnostic procedures, rollback protocol |
| `references/platform-constraints.md` | Pre-compiled platform constraints (NPU/CUDA/AMD) |

Additional reads per phase:
- Phase 1/2 HTML reports: `references/html-template.md`
- Phase 1: `references/phase1-prompts.md`
- Phase 2: `references/phase2-prompts.md`
- Phase 3: `references/phase3-prompts.md`, `references/patterns.md`

### Auto-Allow

Subagent runs `vc_check_*.py`, `json.load()`, `py_compile`. Configure auto-allow in project `.claude/settings.json`:

```json
{"permissions": {"autoAllow": [
  {"tool": "Bash", "pattern": "python.*vc_check.*\\.py"},
  {"tool": "Bash", "pattern": "python -c \"import json; json\\.load"},
  {"tool": "Bash", "pattern": "python -m py_compile"},
  {"tool": "Bash", "pattern": "mkdir -p.*test-out"}
]}}
```

**Confirm**: `"Target: <path> | Platform: <platform> | Phase: <phase> | Max iterations: <n>"`

### Knowledge Recording Rule

After each full pipeline execution, review whether any **new oracle verification method** was discovered or refined during the session. If yes:

1. Add/update the entry in `references/patterns.md` §"Oracle Verification Patterns" using the standard template (source, detection target, determinism, applicable/inapplicable scenarios, Phase 2 template, Phase 3 pattern).
2. If the discovery changes the decision flow (new higher-priority method), update the decision flowchart.
3. If a method was previously missing and caused an entire class of errors (e.g., oracle_gap), ensure the corresponding Plant G rules in `phase2-prompts.md` and correction templates in `correction-templates.md` reference the new method.

This rule ensures the skill accumulates detection knowledge across features and sessions, rather than re-discovering the same methods each time.

---

## Step 1 — Phase 1: Test Point Extraction

Skip if `--phase 2` or `--phase 3`. Read `references/phase1-prompts.md` for full prompts.

### 1a. Explore Source

**Graphify check (MANDATORY — do this FIRST):**

Check if `<project-root>/graphify-out/` directory exists.

**If graphify IS available:**
1. `graphify query "what are the main modules and their dependencies in <target>"`
2. `graphify explain "<key class or concept>"`
3. `graphify path "<source>" "<target>"` for platform-specific code paths
4. `graphify query "what design docs or config parameters relate to <target>"`

Graphify discovers cross-file relationships (e.g., EAGLEWorker → draft_tp_context → attention backend dispatch) that grep cannot. These steps are REQUIRED — do not skip.

**If graphify is NOT available (no `graphify-out/` directory):**

STOP and tell the user:
> "This project has no graphify knowledge graph. Cross-file architecture analysis will rely on grep/glob, which may miss key dependencies, call chains, and platform-specific code paths. Test point quality may be lower. Continue without graphify? (y/n)"

Do NOT proceed until the user explicitly replies "y" or "yes". If they decline, ask if they want to generate a graph first with `/graphify`.

**Source code analysis (always run):**

`grep -n "def \|class "` + Read source for signatures, docstrings, raise statements. APIs are NOT extracted as test targets — they are used only to discover: normal paths through each pipeline stage, error branches and exception types, and platform dispatch points (NPU vs CUDA).

**Design docs:** Glob `docs/**/*<module>*.mdx` / `*.md`. Note absence — NOT an error.

**Existing tests:** Glob `test/registered/**/test_*.py`. Read 1-2 for patterns.

**Platform constraints:** Read `references/platform-constraints.md`. DO NOT skip.

**Report:** `"Module: <N> source files analyzed | Dependencies: <from graphify> | Constraints: <list>"`. If graphify is available but the Dependencies field is empty, you skipped the graphify step — go back and run it.

### 1b. Generate Test Points (Plant G)

Output: `test-out/phase1/test-points.json`
Wrapper: `{"test_points": [...], "_correction_log": []}`

Each test point: `id`, `iteration`, `test_dimension` (always `"workflow"`), `feature`, `source_location` (file:line or doc section), `key_factors[]`, `precondition[]`, `expected_behavior`, `priority` (P0|P1|P2), `boundary_conditions[]`, `error_paths[]`

**`key_factors[]` — the irreducible variables that drive this TP's behavior.** Distill from precondition, boundary_conditions, and expected_behavior: which parameters, platform constraints, or configuration choices, if changed, would produce a meaningfully different outcome? Readers should understand the "what's being varied" without hunting across fields. Every non-deleted TP must have at least one key factor.

**Core principle — user-perspective, not implementation-detail:**

Every test point must come from the **user's/business perspective**: what does the user do, what does the user observe, what goes wrong from the user's view. Source code and internal implementation details are ONLY used to understand impact factors and data flow — they are NEVER test targets themselves. We are not pursuing code coverage; we are covering **usage scenarios**.

A test point is invalid (do NOT generate; delete during correction) if:
- Its expected_behavior can only be verified by inspecting internal state (tensor dtype, KV cache content, compilation status, specific API call chains)
- It describes "how the code works" rather than "what the user experiences"
- Its observable signal is indistinguishable from a broader scenario already covered by other TPs

**Checklist:**
```
[ ] Source code analyzed for normal paths + error branches (NOT extracted as test targets)
[ ] Design docs mined for user scenarios + configuration variations
[ ] Platform constraints reflected in boundary_conditions + error_paths
[ ] Core principle: every TP observable from user perspective (output/metric/exception/status/log). No internal-state-only claims. If a TP's only unique contribution is describing an internal implementation detail that other TPs cover end-to-end, delete it.
[ ] key_factors[] populated: the irreducible variables driving this TP (params, platform constraints, config choices). Readers do not need to hunt across fields to understand what is being varied.
[ ] Scenario categories: normal, boundary/configuration, error/exception, platform-specific
[ ] JSON syntax valid: python -c "import json; json.load(open('test-out/phase1/test-points.json', encoding='utf-8'))"
```

### 1c. Correction Cycle

Main never detects or fixes — only Plant G + C1-C5 judging. Two subagents alternate: Sub-A (odd iterations), Sub-B (even iterations). Full mechanics: `references/correction-cycle.md`.

**Main MUST NOT run VC checks** (`vc_check_phase1.py`) or SC checks (missing_boundary, factual_error, etc.) on the output. These are the subagent's job — Main running them duplicates work and corrupts the independence of the correction cycle. The ONLY validation Main runs after Plant G is: `python -c "import json; json.load(open('test-out/phase1/test-points.json', encoding='utf-8'))"` — confirm the file is parseable JSON before handing it to subagents. Always pass `encoding='utf-8'` on Windows; the default system encoding (GBK) will crash on Unicode characters.

Note: `vc_check_coverage.py` is NOT used in Phase 1 — there is no API coverage metric. Coverage is about scenario completeness (are all workflow stages covered?), which is a semantic check.

**BEFORE launching subagents**: Main MUST resolve all `<placeholders>` in the first-launch prompt to absolute paths. The subagents run in background contexts that do NOT share Main's working directory. Specifically:
- `<skill-base>` → resolve to the absolute path of this skill's base directory (the directory containing SKILL.md, `references/`, `scripts/`)
- `<source_dirs>` → absolute paths to scanned source directories
- `<list from Step 1>` → space-separated absolute paths to discovered source files
- `<path or "none">` → absolute path to design doc or the string `"none"`

| Parameter | Value |
|-----------|-------|
| Sub first-launch prompt | `references/phase1-prompts.md` §Step 3 (same prompt for both Sub-A and Sub-B) |
| MAX_ITER | 4 |
| Subagent spec | `subagent_type: "general-purpose"`, `run_in_background: true`. Launch BOTH at start. |
| Detection scripts | `vc_check_phase1.py` (schema validation only, no coverage check) |
| SC checks | missing_boundary, missing_precondition, missing_platform_constraint, factual_error, unverifiable_claim, logical_inconsistency, duplicate_tp, missing_scenario |

### 1d. Review & Decide

After each detection, evaluate top-to-bottom (first match wins). Full details: `references/convergence.md`. Also review the confidence summary — TPs below 0.70 should be prioritized in the next correction round.

```
s_t == 0                              → C1 CONVERGED → stop
t >= MAX_ITER                         → C5 EXHAUSTED → stop, argmin s(y)
t >= 3 AND τ_t == τ_{t-2} != τ_{t-1}  → C3 OSCILLATION → stop, argmin s(y)
s_t > s_{t-1} + 3                     → C4 OVERSHOOT → rollback, different strategy, continue
s_t <= s_{t-1} AND s_t >= s_{t-1} - 1 → C2 PLATEAUED → stop
Otherwise                              → continue
```

### 1e. Completion — HARD CHECKLIST

```
[ ] JSON valid (python -c "import json; json.load(...)")
[ ] All required fields present on every TP
[ ] HTML report generated: test-out/phase1/test-points.html (see html-template.md)
[ ] User shown: TP count, s_t trajectory, HTML path
[ ] User replied "y" or "yes"
```

If `--phase 1`: stop. Otherwise present results and ask: **"Continue to Phase 2? (y/n)"**

---

## Step 2 — Phase 2: Text Test Case Generation

Skip if `--phase 1` or `--phase 3`. Read `references/phase2-prompts.md` for full prompts.

### 2a. Read Phase 1 Output

Read `test-out/phase1/test-points.json`. Check for `_unresolved_errors` and `_correction_log` items with `action: "skipped"`. Account for known gaps when generating test cases. Add `_inherited_issues` to Phase 2 output.

### 2b. Generate Text Test Cases (Plant G)

Output: `test-out/phase2/test-cases.json`
Wrapper: `{"test_cases": [...], "_correction_log": []}`

Each test case: `id` (TC-NNN), `iteration`, `test_point_id` (TP-NNN), `title`, `description`, `preconditions[]`, `test_data{}`, `steps[]`, `expected_results{}`, `teardown[]`

Rules: every step produces a named result; later steps reference earlier results by name; assertions must reference SPECIFIC values.

### 2c. Correction Cycle

Same dual-Sub alternating pattern as Phase 1. See `references/correction-cycle.md`. Resolve placeholders as described in §1c.

| Parameter | Value |
|-----------|-------|
| Sub first-launch prompt | `references/phase2-prompts.md` §Step 2 (same prompt for both) |
| MAX_ITER | 5 |
| Detection scripts | `vc_check_phase2.py` |
| SC checks | vague_assertion, missing_precondition, inconsistent_trace, invalid_test_logic, structural_flaw, logical_inconsistency |

### 2d. Review & Decide

Same C1-C5 as 1d. Phase-specific: same TC-ID flagged for `structural_flaw` twice → mark "skipped".

### 2e. Completion — HARD CHECKLIST

```
[ ] JSON valid
[ ] All required fields present on every TC
[ ] HTML report generated: test-out/phase2/test-cases.html
[ ] User shown: TC count, s_t trajectory, HTML path
[ ] User replied "y" or "yes"
```

If `--phase 2`: stop. Otherwise: **"Continue to Phase 3? (y/n)"**

---

## Step 3 — Phase 3: Test Script Generation

Skip if `--phase 1` or `--phase 2`. Read `references/phase3-prompts.md` and `references/patterns.md`.

### 3a. Read Inputs & Patterns

1. Read `test-out/phase2/test-cases.json`; check for `_unresolved_errors` / `_inherited_issues`
2. Read `references/patterns.md` for platform test conventions
3. Glob existing test files; read 1-2 for import lines, base classes, CI registration

### 3b. Generate Test Script (Plant G)

Output: `test-out/phase3/test[_npu]_<module>.py`. One test method per text test case.

Platform rules:
- **Imports**: copy verbatim from existing test files
- **Base class**: `CustomTestCase` (server) or `unittest.TestCase` (unit)
- **NPU**: `--attention-backend ascend` in server args, model paths from `test_ascend_utils`
- **CI registration**: `register_npu_ci` / `register_cuda_ci` at module level
- **End with**: `if __name__ == "__main__": unittest.main()`

### 3c. Correction Cycle

Same dual-Sub alternating pattern. See `references/correction-cycle.md`. Resolve placeholders as described in §1c. Phase 3 uses real tool outputs (py_compile, grep).

| Parameter | Value |
|-----------|-------|
| Sub first-launch prompt | `references/phase3-prompts.md` §Step 3 (same prompt for both) |
| MAX_ITER | 3 |
| Detection tools | py_compile, grep for backend flag / CI registration / API names / assertions |

### 3d. Review & Decide

Same C1-C5 as 1d. Phase-specific: C4 compares by file+line (not IDs).

### 3e. Completion — HARD CHECKLIST

```
[ ] python -m py_compile <output> passes (exit 0)
[ ] Test file at correct platform path
[ ] CI registration matches platform
[ ] HTML report generated: test-out/phase3/test-script.html (see html-template.md §Phase 3)
[ ] User replied "y" or "yes"
```

**"Phase 3 complete. Accept? (y/n)"** — do not end until user confirms.

---

## Platform Quick Reference

| Aspect | NPU (Ascend) | CUDA (NVIDIA) |
|---|---|---|
| CI registration | `register_npu_ci` | `register_cuda_ci` |
| Per-commit suite | `stage-b-test-1-npu-a2` | `stage-b-test-1-gpu-small` |
| Model paths | `sglang.test.ascend.test_ascend_utils` | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` |
| Backend flag | `--attention-backend ascend` | (none) |
| Test dir | `test/registered/ascend/<category>/` | `test/registered/<category>/` |
| File prefix | `test_npu_` | `test_` |

## Output Structure

```
<project-root>/
  test-out/
    phase1/test-points.json
    phase1/test-points.html
    phase2/test-cases.json
    phase2/test-cases.html
    phase3/test[_npu]_<module>.py
```

## Reference Files

| File | When to Read | Content |
|------|-------------|---------|
| `references/error-types.md` | Step 0 (always) | Error taxonomy, severity weights, τ→action table |
| `references/convergence.md` | Step 0 (always) | C1-C5 algorithm, C4 diagnostic, rollback protocol |
| `references/correction-cycle.md` | Step 1c/2c/3c | Alternating Main↔Sub mechanics, SendMessage patterns |
| `references/platform-constraints.md` | Step 0 (always) | Platform-specific constraints for NPU/CUDA/AMD |
| `references/phase1-prompts.md` | Phase 1 | Exploration guide, Plant G checklist, Sub prompts |
| `references/phase2-prompts.md` | Phase 2 | TC schema, Plant G rules, Sub prompts |
| `references/phase3-prompts.md` | Phase 3 | Script generation rules, tool-grounded checks, Sub prompts |
| `references/html-template.md` | End of Phase 1/2 | HTML/CSS patterns for auto-generated reports |
| `references/patterns.md` | Phase 3 | SGLang test conventions per platform |
| `references/correction-templates.md` | Sub agents | Fix strategies per error type per phase |
