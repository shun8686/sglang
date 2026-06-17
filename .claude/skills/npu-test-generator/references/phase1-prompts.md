# Phase 1: Test Point Extraction Prompts

## Step 1: Source Exploration

```
A. GRAPHIFY CHECK (MANDATORY — do this FIRST):
   If graphify-out/ exists in the project root:
   1. graphify query "what are the main modules and their dependencies in <target>"
   2. graphify explain "<key class or concept>"
   3. graphify path "<source>" "<target>" for platform-specific code paths
   4. graphify query "what design docs or config parameters relate to <target>"
   This discovers cross-file relationships that grep cannot (e.g., call chains,
   backend dispatch, platform-specific code paths). These steps are REQUIRED.

   If graphify-out/ does NOT exist:
   STOP and tell the user: "No graphify knowledge graph found. Cross-file analysis
   will rely on grep/glob, which may miss key dependencies. Continue? (y/n)"
   Do NOT proceed without explicit "y" or "yes".

B. SOURCE CODE ANALYSIS (always run):
   1. grep -n "def \|class " <target> → line numbers and signatures
   2. Read each source file → docstrings, param types, raise statements
   3. APIs are NOT extracted as test targets — they are used to discover:
      - Normal paths: what does each pipeline stage do?
      - Error branches: what exceptions are raised, what guards exist?
      - Platform dispatch: where does NPU vs CUDA path diverge?

C. DESIGN DOCS (PRIMARY scenario source):
   Glob docs/**/*<module>*.mdx / *.md. Extract user actions, launch commands,
   configuration variations, request flows. These are the PRIMARY source for
   workflow test scenarios.

D. EXISTING TESTS:
   Glob test/registered/**/test_*.py. Read 1-2 for patterns, imports, CI style.

E. PLATFORM CONSTRAINTS (DO NOT SKIP):
   Read references/platform-constraints.md for the target platform.
   Each constraint should spawn at least one workflow scenario.

F. REPORT:
   "Module: <N> source files analyzed | Dependencies: <from graphify> | Constraints: <list>"
```

## Step 2: Generate Test Points (Plant G)

Output: `test-out/phase1/test-points.json`
Wrapper: `{"test_points": [...], "_correction_log": []}`

Test point schema:
```
{
  "id": "TP-NNN",
  "iteration": 1,
  "test_dimension": "workflow",
  "feature": "...",
  "source_location": "file:line (symbol) or doc section",
  "key_factors": ["param/constraint that drives this TP's behavior", "..."],
  "precondition": [...],
  "expected_behavior": "...",
  "priority": "P0|P1|P2",
  "boundary_conditions": [...],
  "error_paths": [...]
}
```

**key_factors extraction rule:** After filling all other fields, ask: "Which 2-5 variables, if changed, would produce a meaningfully different outcome?" These are the irreducible drivers. Extract them from preconditions (what must be in place), boundary_conditions (what values are being varied), and platform constraints (what is fixed by hardware). Do NOT copy-paste every boundary value — distill to the underlying variable. Example: `--speculative-eagle-topk=1` and `--speculative-eagle-topk=4` both distill to the single factor `"--speculative-eagle-topk (branching factor)"`.

**Methodology — workflow-driven scenario discovery:**

All test points are workflow scenarios. There is NO API dimension. Source code APIs are analysis aids only — they reveal normal paths, error branches, and platform dispatch points, but are never test targets themselves.

**Scenario sources (use ALL):**
1. Design doc (primary): user actions, launch commands, request flows, configuration variations
2. Source code (analysis): trace call chains to discover normal paths, error branches, platform dispatch
3. Existing tests (reference): what scenarios exist, what patterns are used
4. Platform constraints (mandatory): each constraint spawns at least one scenario

**Scenario categories to cover:**
- **Normal paths**: server launch, single request, batch requests, each pipeline stage
- **Boundary/configuration**: parameter extremes, feature toggles, platform defaults
- **Error/exception**: bad config, missing files, OOM, unsupported features
- **Platform-specific**: NPU vs CUDA dispatch paths, kernel differences, backend constraints

**Checklist:**
```
[ ] 1. Source code analyzed for normal paths + error branches (NOT extracted as test targets)
[ ] 2. Design docs mined for user scenarios + configuration variations
[ ] 3. FILL: expected_behavior = observable signal ONLY (output/metric/exception/status)
[ ] 4. PRIORITIZE: P0=crash/corrupt/wrong, P1=feature/config, P2=cosmetic/edges
[ ] 5. PLATFORM: reflect constraints from 1a-E in boundary_conditions + error_paths
[ ] 6. SELF-CHECK: all scenario categories covered. JSON valid.
[ ] 7. No __init__ or constructor-only scenarios (covered by workflow TPs)
```

## Step 3: Subagent Prompts

Two subagents alternate: Sub-A handles odd iterations (1, 3, 5), Sub-B handles even (2, 4). Both use the same prompt below. Main never detects or fixes — only judges C1-C5.

### First-Launch Prompt (Sub-A at Iter 1, Sub-B at Iter 2)

```
You are the CyberCorrect Correction Subagent for Phase 1.
Your role: detect errors first, then fix when Main tells you to.
Main always judges convergence (C1-C5). Do NOT judge convergence yourself.

IMPORTANT — s_t semantics: s_t = 0.8×N_crit + 0.5×N_mod + 0.2×N_min (pre-fix).
Convergence (C1) uses s_t == 0.

SETUP — read these NOW and keep in context for all future iterations:
- Source files: <list from Step 1>
- Design doc: <path or "none">
- <skill-base>/references/platform-constraints.md
- <skill-base>/references/correction-templates.md

CROSS-FILE CONTEXT (REQUIRED for factual_error checks — DO NOT SKIP):
  If graphify-out/ exists in the project root:
  1. graphify explain "<symbol>" --graph <project>/graphify-out/graph.json
  2. graphify path "<source>" "<target>" --graph <project>/graphify-out/graph.json
  This reveals call chains, imports, and platform code paths that single-file grep misses.
  Skip ONLY if graphify-out/ does not exist. Do NOT fall back to Grep/Read if graphify IS available.

THIS ITERATION (Iter 1): Read test-out/phase1/test-points.json. DETECT ONLY — do NOT fix:

NOTE: Some TPs may have a `"merged_into"` field (set by a previous correction cycle to deduplicate). These TPs are already resolved — skip them in ALL checks (VC, SC, confidence). Do NOT re-flag them as errors. They remain in the JSON as an audit trail only. Treat them identically to how Phase 2 will treat them: they exist but generate no downstream work.

1. VC: python <skill-base>/scripts/vc_check_phase1.py test-out/phase1/test-points.json
        (schema validation only — no coverage script. Coverage is about scenario completeness, checked via SC.)

2. SC checks: missing_boundary, missing_precondition, missing_platform_constraint,
   factual_error, unverifiable_claim, logical_inconsistency, duplicate_tp, missing_scenario
   (see error-types.md for detection criteria)

   For unverifiable_claim: test points must describe user-observable behavior.
   Internal implementation details (tensor dtype, compilation status, API call chains) are
   NOT valid test targets. If a TP's only unique contribution is internal detail already
   protected by other E2E TPs, flag it for DELETION (not downgrade / VERIFIABILITY NOTE).

   missing_scenario check: for each workflow stage (draft, verify, extend), each configuration axis
   (topk, num_steps, num_draft_tokens, attention_mode), each platform constraint from
   platform-constraints.md, and each error branch found in source — is there a corresponding TP?
3. Report: s_t (pre-fix), error list, and confidence summary. Do NOT apply corrections.

AFTER reporting, you MUST write confidence scores to disk as a SEPARATE step. This is REQUIRED — do not skip it:
4. WRITE test-out/phase1/confidence.json with per-TP scores on 5 dimensions (0-1 each):
   source_grounding, observability, specification_completeness, platform_fidelity, logical_coherence.
   Format: {"TP-001": {"source_grounding": 0.9, "observability": 0.8, ...}, ...}
   Then VERIFY the file was written: python -c "import json; json.load(open('test-out/phase1/confidence.json', encoding='utf-8'))"
```

### Detect SendMessage (resume, clean slate)

```
Iter <t>: Re-read test-points.json. Run VC+SC checks from scratch on the entire file. DETECT ONLY. Report s_t, error list, and confidence summary. Then WRITE confidence scores to test-out/phase1/confidence.json and VERIFY with python -c "import json; json.load(open('test-out/phase1/confidence.json', encoding='utf-8'))". DO NOT fix yet. Do NOT read _correction_log.
```

### Fix SendMessage (only if Main says CONTINUE)

```
Iter <t>: You are in CORRECTOR-ONLY mode. Apply corrections for the errors you detected in YOUR PREVIOUS TURN. Write _correction_log with "iteration": <t>. Increment iteration on affected TPs.

FORBIDDEN IN THIS TURN — do NOT do any of the following:
- Do NOT re-read the output file (work from your memory of the errors you detected)
- Do NOT run VC scripts or SC checks
- Do NOT compute s_t or judge convergence
- Do NOT report error counts or confidence scores
- Do NOT generate new test points UNLESS Main's message explicitly authorizes it
  (e.g., "Add N new test points with IDs TP-XXX, TP-YYY" for missing_scenario fixes)

Convergence judgment, detection, and s_t computation are the NEXT iteration's job (done by a DIFFERENT subagent). Your ONLY job this turn is to edit the file to fix the errors you previously detected.
```
