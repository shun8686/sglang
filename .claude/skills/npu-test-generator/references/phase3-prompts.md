# Phase 3: Test Script Generation Prompts

## Step 1: Read Inputs & Patterns

1. Read `test-out/phase2/test-cases.json`; check for `_unresolved_errors` / `_inherited_issues`
2. Read `references/patterns.md` for platform-specific test conventions
3. Glob existing test files for the target platform:
   - NPU: `test/registered/ascend/<category>/test_npu_*.py`
   - CUDA: `test/registered/<category>/test_*.py`
4. Read 1-2 representative test files — extract import lines, base classes, CI registration

## Step 2: Generate Test Script (Plant G)

Translate each text test case into a Python test method.

Output path:
- NPU: `test-out/phase3/test_npu_<module>.py`
- CUDA: `test-out/phase3/test_<module>.py`

**Platform rules (follow EXACTLY):**
- **Imports**: Copy from existing test files verbatim
- **Base class**: `CustomTestCase` for server tests, `unittest.TestCase` for unit tests
- **CI registration**: Module-level, platform-specific function + suite name from patterns.md
- **NPU server args**: MUST include `"--attention-backend", "ascend"`
- **NPU model paths**: From `sglang.test.ascend.test_ascend_utils`, NOT `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`
- **End with**: `if __name__ == "__main__": unittest.main()`

**Data fidelity rule (DO NOT SKIP):**
A TC may have been modified through multiple correction iterations. Its `expected_results` in Phase 2 JSON is the **authoritative source** — do NOT rely on memory, template defaults, or what "looks right." When translating each TC:
1. Read its `expected_results` from `test-cases.json` **as it currently exists**.
2. Every `assert*` in the test method must correspond to an expected_result entry. If a Phase 2 correction changed `assertIn("4")` to `assertRegex(r'\b4\b')`, the test method MUST use the corrected version.
3. If a TC has `"skipped": true`, do NOT generate a test method for it.

## Step 3: Subagent Prompts

Two subagents alternate: Sub-A handles odd iterations (1, 3, 5), Sub-B handles even (2, 4). Both use the same prompt below. Phase 3 uses real tool outputs (py_compile, grep) as ground truth. Main never detects or fixes — only judges C1-C5.

### First-Launch Prompt (Sub-A at Iter 1, Sub-B at Iter 2)

```
You are the CyberCorrect Correction Subagent for Phase 3 (tool-grounded).
Your role: detect errors first, then fix when Main tells you to.
Main always judges convergence (C1-C5). Do NOT judge convergence yourself.

IMPORTANT — s_t: s_t = 0.8×N_crit + 0.5×N_mod + 0.2×N_min (pre-fix).

SETUP — read these NOW and keep in context:
- <skill-base>/references/correction-templates.md (Phase 3 fix strategies)
- <skill-base>/references/platform-constraints.md (platform-specific flags/paths)
- <skill-base>/references/patterns.md (SGLang test conventions)

CROSS-FILE CONTEXT (REQUIRED for API name verification — DO NOT SKIP):
  If graphify-out/ exists in the project root:
  1. graphify explain "<imported symbol>" --graph <project>/graphify-out/graph.json
  2. graphify path "<script>" "<source module>" --graph <project>/graphify-out/graph.json
  This reveals whether imported names actually exist and their correct call signatures.
  Skip ONLY if graphify-out/ does not exist. Do NOT fall back to Grep if graphify IS available.

CONTEXT:
- Script to audit: <path to generated .py file>
- Source directory: <path for API verification>
- Phase 2 test case count: <N>
- Phase 2 output: <path to test-cases.json> (for config_mismatch + step_fidelity checks)

CHECKS (VC: run Bash/Grep. SC: read + compare. DETECT ONLY — do NOT fix):

VC (tool-grounded):
1. Syntax (s=0.8): python -m py_compile <script>
2. Backend flag (s=0.8, NPU): Grep "attention-backend.*ascend"
3. CI registration (s=0.8): Grep register_npu_ci / register_cuda_ci
4. File placement (s=0.5, NPU): Verify path matches ascend/<cat>/test_npu_*.py
5. API names (s=0.5): Grep source for each import used in script
6. Assertions (s=0.5): Grep -c "self.assert"; count methods
7. Coverage (s=0.2): Method count vs Phase 2 test case count

SC (semantic — read script + cross-reference Phase 2 TCs):
8. config_mismatch (s=0.8): For each test method, cross-reference its actual server args (from setUpClass or inline popen_launch_server) against the source TC's test_data.other_args. Flag when a method's runtime config differs from what the TC requires. Classic case: setUpClass uses topk=1 but TC-004 requires topk=4.
9. oracle_gap (s=0.8): For each test method, mentally simulate: remove the feature's key flag from server args. Would all assertions still pass? If yes, the oracle cannot distinguish the feature working from being absent. Flag unverifiable claims.
10. step_fidelity (s=0.5): Does the test method cover all steps from the source TC? Flag missing setup logic, skipped assertions, or consolidated steps that lose coverage.
11. teardown_leak (s=0.5): For every popen_launch_server call, verify a corresponding kill_process_tree exists in a finally block (or tearDownClass for shared servers). Flag cleanup gaps on error paths.

Report: s_t (pre-fix) = 0.8*N_crit + 0.5*N_mod + 0.2*N_min (combining VC + SC). Report error list. Do NOT apply corrections.

NOTE: Phase 3 does NOT write confidence scores (output is a .py file, not structured test design data). Skip the confidence step — it only applies to Phases 1 and 2.
```

### Detect SendMessage (resume, clean slate)

```
Iter <t>: Re-check script. Run all VC+SC checks from scratch. DETECT ONLY. Report s_t and error list. DO NOT fix yet. Do NOT read _correction_log.
```

### Fix SendMessage (only if Main says CONTINUE)

```
Iter <t>: You are in CORRECTOR-ONLY mode. Apply corrections for the errors you detected in YOUR PREVIOUS TURN.

FORBIDDEN IN THIS TURN — do NOT do any of the following:
- Do NOT re-read the output file (work from your memory of the errors you detected)
- Do NOT run py_compile, grep, or any other detection tool
- Do NOT compute s_t or judge convergence
- Do NOT report error counts

Convergence judgment and detection are the NEXT iteration's job (done by a DIFFERENT subagent). Your ONLY job this turn is to edit the .py file to fix the errors you previously detected.
```
