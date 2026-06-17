# Phase 2: Text Test Case Generation Prompts

## Step 1: Generate Text Test Cases (Plant G)

Read `test-out/phase1/test-points.json`. Check for `_unresolved_errors` and skipped items in `_correction_log` — account for known gaps. **Skip TPs with `"merged_into"` set** — these were deduplicated in Phase 1 and should not generate test cases.

Output: `test-out/phase2/test-cases.json`
Wrapper: `{"test_cases": [...], "_correction_log": []}`

Test case schema:
```
{
  "id": "TC-NNN",
  "iteration": 1,
  "test_point_id": "TP-NNN",
  "title": "snake_case_descriptive_name",
  "description": "One sentence what this test verifies",
  "preconditions": ["Server/model state", "Data that must exist"],
  "test_data": {"var_name": "concrete value with type"},
  "steps": [
    "Step 1: Action → produces result_1",
    "Step 2: Action using result_1 → produces result_2"
  ],
  "expected_results": {
    "result_n": "assertEqual(result_n, <specific value>)",
    "side_effects": "mock.foo.assert_called_once_with(<specific args>)"
  },
  "teardown": ["Cleanup step"]
}
```

**Rules:**
- Every step produces exactly ONE named result. Prefer splitting multi-variable outputs into separate steps for readability (e.g., `→ produces text_1` then `→ produces text_2` rather than `→ produces text_1, text_2`).
- Every expected_results key MUST exactly match its corresponding step output variable name. The VC trace checker does regex matching — semantic names like `"health_ok"` will break the trace even if the assertion value text references the correct variable. Use `"health_status"` not `"health_ok"`, `"response"` not `"response_ok"`.
- Later steps reference earlier results by name
- Assertions must reference SPECIFIC values
- For each test point, generate at least one test case
- **Oracle verification**: When a TC's core claim depends on a specific CLI flag (stored in `ServerArgs`), add a `GET /server_info` step and assert on the returned JSON fields. This covers `--flag value` parameters only — not environment variables, request-level params, or runtime behavior. See `references/patterns.md` §"Oracle Verification Patterns" for the decision flow and all available methods.

**JSON purity:** No Python expressions in JSON values. After writing, verify:
`python -c "import json; json.load(open('test-out/phase2/test-cases.json', encoding='utf-8'))"`

**Tokenization hazard:** Character count ≠ token count. LLM tokenizers produce far fewer tokens for repeated characters. Describe prompts semantically or note tokenizer verification is needed.

## Step 2: Subagent Prompts

Two subagents alternate: Sub-A handles odd iterations (1, 3, 5), Sub-B handles even (2, 4). Both use the same prompt below. Main never detects or fixes — only judges C1-C5.

### First-Launch Prompt (Sub-A at Iter 1, Sub-B at Iter 2)

```
You are the CyberCorrect Correction Subagent for Phase 2.
Your role: detect errors first, then fix when Main tells you to.
Main always judges convergence (C1-C5). Do NOT judge convergence yourself.

IMPORTANT — s_t: s_t = 0.8×N_crit + 0.5×N_mod + 0.2×N_min (pre-fix).
Convergence (C1) uses s_t == 0.

SETUP — read these NOW and keep in context:
- test-out/phase1/test-points.json (for structural_flaw comparison)
- <skill-base>/references/correction-templates.md (Phase 2 fix strategies)

CROSS-FILE CONTEXT (REQUIRED for structural_flaw checks — DO NOT SKIP):
  If graphify-out/ exists in the project root:
  1. graphify explain "<TC's referenced API>" --graph <project>/graphify-out/graph.json
  2. graphify path "<draft model>" "<target model>" --graph <project>/graphify-out/graph.json
  This reveals whether the test case's claimed behavior matches actual code dependencies.
  Skip ONLY if graphify-out/ does not exist. Do NOT fall back if graphify IS available.

THIS ITERATION (Iter 1): Read test-out/phase2/test-cases.json. DETECT ONLY — do NOT fix:
1. VC: python <skill-base>/scripts/vc_check_phase2.py test-out/phase2/test-cases.json
2. SC checks:
   a. vague_assertion (s=0.5): "should","correctly","properly" or no concrete value?
   b. missing_precondition (s=0.5): mock objects, test data, server state listed?
   c. inconsistent_trace (s=0.5): step N output referenced in step N+1 or expected_results?
   d. invalid_test_logic (s=0.8): recompute capacity/demand. Flag char-counted prompts.
   e. structural_flaw (s=0.8): test incapable of distinguishing pass vs fail?
   f. oracle_gap (s=0.8): For each expected_results assertion: mentally remove the feature's key flag from test_data. Would the assertion still pass? If yes, the oracle cannot distinguish the feature working from being absent. This catches the gap at TC design time — check ALL TCs, not just those with structural_flaw. See error-types.md for the full detection criteria.
   g. logical_inconsistency (s=0.8): precondition-step-expected_results chain coherent? see error-types.md
   h. duplicate_tc (s=0.5): two TCs with overlapping test_point_id, preconditions, steps, and expected_results? Higher bar than TP dedup — TCs testing different boundary values of the same TP are NOT duplicates.
3. Report: s_t (pre-fix), error list, and confidence summary. Do NOT apply corrections.

AFTER reporting, you MUST write confidence scores to disk as a SEPARATE step. This is REQUIRED — do not skip it:
4. WRITE test-out/phase2/confidence.json with per-TC scores on 5 dimensions (0-1 each):
   source_grounding, observability, specification_completeness, trace_coherence, logical_coherence.
   Format: {"TC-001": {"source_grounding": 0.9, "observability": 0.8, ...}, ...}
   Then VERIFY the file was written: python -c "import json; json.load(open('test-out/phase2/confidence.json', encoding='utf-8'))"
```

### Detect SendMessage (resume, clean slate)

```
Iter <t>: Re-read test-cases.json. Run VC+SC checks from scratch on the entire file. DETECT ONLY. Report s_t, error list, and confidence summary. Then WRITE confidence scores to test-out/phase2/confidence.json and VERIFY with python -c "import json; json.load(open('test-out/phase2/confidence.json', encoding='utf-8'))". DO NOT fix yet. Do NOT read _correction_log.
```

### Fix SendMessage (only if Main says CONTINUE)

```
Iter <t>: You are in CORRECTOR-ONLY mode. Apply corrections for the errors you detected in YOUR PREVIOUS TURN. Write _correction_log with "iteration": <t>. Increment iteration on affected TCs.

FORBIDDEN IN THIS TURN — do NOT do any of the following:
- Do NOT re-read the output file (work from your memory of the errors you detected)
- Do NOT run VC scripts or SC checks
- Do NOT compute s_t or judge convergence
- Do NOT report error counts or confidence scores
- Do NOT generate new test cases UNLESS Main's message explicitly authorizes it
  (e.g., "Add N new test cases with IDs TC-XXX, TC-YYY" for missing_scenario or structural_flaw fixes)

Convergence judgment, detection, and s_t computation are the NEXT iteration's job (done by a DIFFERENT subagent). Your ONLY job this turn is to edit the file to fix the errors you previously detected.
```

**structural_flaw note:** If same TC-ID flagged again with structural_flaw → mark "skipped" in _correction_log.

**NOT structural_flaw (do NOT flag):**
- Tests measuring proxy metrics (TTFT, hit rate, output correctness) instead of internal implementation — this is normal E2E testing
- Tests needing hardware-specific parameter tuning when documented
- Tests that distinguish pass/fail in current form even if measurement could be more precise
