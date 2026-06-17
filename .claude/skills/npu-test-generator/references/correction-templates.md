# Type-Directed Correction Templates

Based on CyberCorrect Section III-C. Each error type τ has a targeted correction instruction, avoiding generic "please fix" prompts.

## Severity-Adaptive Correction Intensity

Per the paper (Section III-C): the correction intensity adapts to error severity s.

```
s > 0.7  (critical)  →  FULL REGENERATION from the error location ℓ onward
                        Re-read source, regenerate the affected section entirely
                        Example: factual_error at critical → re-read the relevant source section,
                        correct the specific claim inline

s ≤ 0.7  (moderate/minor) → MINIMAL TARGETED EDIT
                        Preserve correct content, only inject the fix at location ℓ
                        Example: vague_assertion → replace only the assertion text,
                        keep all other test case fields unchanged
```

**Severity reference** (from error-types.md):
| Severity | s Score | Correction Mode |
|---|---|---|
| critical | 0.8 | Full regeneration from ℓ |
| moderate | 0.5 | Targeted edit |
| minor | 0.2 | Targeted edit (minimal) |

**When multiple errors exist**: apply corrections in descending severity order.
Critical errors first (may require regeneration that subsumes minor fixes).

## Phase 1 Corrections

### missing_scenario

```
The following workflow scenario categories have no test point coverage:
{list_of_missing_scenarios}

This is a scenario gap, NOT an API gap. Do NOT add per-class or per-method TPs.

1. Identify which workflow stage / config axis / platform constraint / error branch is uncovered
2. Design a user-observable scenario that exercises this gap
3. Add ONE workflow test point for the scenario — frame it as an externally verifiable behavior

Add the new test point(s) to the JSON output.
```

Note: `missing_scenario` severity is **moderate (s=0.5)** per error-types.md. This is a semantic check — the subagent compares the set of TPs against the known workflow stages, config axes, platform constraints, and error branches from source.

### missing_boundary

```
For test point {id} ({feature}), the following boundary values are missing:
- Parameter: {param_name}, type: {type}
  Missing: {list_of_missing_boundaries}

Add boundary test point variants for each missing case:
- Type is numeric: test 0, negative, max value, min value
- Type is string: test empty string, very long string, special chars
- Type is list/collection: test empty list, single element, duplicates
```

### missing_precondition

```
For test point {id} ({feature}), the following preconditions are missing:
- {list_of_missing_preconditions}

Review the source constructor and method signatures. For each dependency:
1. What must exist before this feature can execute? (DB rows, files, config)
2. What state must the system be in? (auth, initialized, connected)
3. What mock objects must be set up? (stubs, fixtures)

Add these as explicit preconditions in the test point.
```

### missing_platform_constraint

```
For test point {id} ({feature}), the platform constraint for parameter {param} is not captured.

Platform: {platform_name} ({npu|cuda|amd})
Constraint source: {doc_file}:{line} — {constraint_description}
Current test point says: {current_values}
Platform requires: {fixed_value_or_restricted_range}

Update the test point:
1. boundary_conditions: Replace generic values with platform-valid values ONLY
   - Remove boundary values that are invalid on this platform
   - Add the platform's fixed value as the primary boundary
2. expected_behavior: Add note about platform-specific enforcement
   - "On {platform}, {param} is fixed to {value} by the {backend} backend"
3. error_paths: Add attempts to use non-platform values
   - "--{param} {invalid_value} → startup warning, auto-corrected to {platform_value}"
4. precondition: Add the platform constraint as a known limitation
   - "Platform: {platform_name} requires {param}={value}"

Example — Ascend NPU page_size constraint:
  boundary_conditions: ["--page-size 128 (Ascend native, mandatory)"]
  error_paths: ["--page-size != 128 on Ascend → auto-corrected or rejected"]
  expected_behavior: "...On Ascend NPU, page_size is fixed to 128 by the backend."
```

### factual_error

```
Test point {id} has an incorrect technical claim in expected_behavior or precondition.
  Current claim: "{incorrect_statement}"
  Actual behavior (from design doc / source): "{correct_statement}"

Fix by:
1. Re-read the relevant section of the design doc and source code
2. Identify which flag/API/mechanism actually controls what
   Example: --enable-hierarchical-cache controls L2 only, not L1 (RadixAttention)
3. Rewrite expected_behavior with the correct allocation of responsibilities
4. Update precondition to note the separation of concerns
5. Add boundary_conditions and error_paths that reflect the correct model
```

### unverifiable_claim (Phase 1)

```
Test point {id} asserts a behavior with no externally observable signal from the user's perspective:
  Claim: "{unverifiable_claim}"
  Problem: This requires inspecting internal state that cannot be surfaced in E2E testing.
           Test points must serve user-perspective usage scenarios, not code-coverage goals.

Two options:
A. DELETE the entire TP if its ONLY unique contribution is an internal implementation detail.
   Other TPs already cover the end-to-end behavior this internal detail protects.
   Remove the TP from the JSON array entirely. Do NOT renumber remaining IDs.
   Add a _correction_log entry: {"action": "deleted", "reason": "internal-only TP, covered by <other TP IDs>"}

B. DELETE just the unverifiable claims from expected_behavior if the TP also has valid
   user-observable scenarios. Keep only the externally verifiable portions.
   Update confidence scores accordingly (observability <= 0.3 for the remaining claims).

Adding a [VERIFIABILITY NOTE] is NOT sufficient — internal-only TPs waste downstream phase effort.
If you cannot identify a user-observable contribution unique to this TP, delete it.

Decision rule:
- Is this TP's expected_behavior describing what a USER sees/measures? -> Keep (Option B for minor edits)
- Is it describing HOW the code works internally, already verified by TP-002/TP-025? -> Delete (Option A)
```

### vc_id_uniqueness (Phase 1 VC)

```
Test point(s) {id_list} have duplicate IDs or malformed priorities.

Fix by:
1. Duplicate IDs: renumber to restore unique TP-NNN numbering
2. Bad priority: replace with correct P0/P1/P2 value (P0=core, P1=feature, P2=edge)
```

### vc_json_schema (Phase 1 VC)

```
Test point(s) {id_list} missing required fields.

For each affected test point:
1. Read the source/doc to understand what the test point should cover
2. Add the missing field(s) with appropriate values from source context
   Required: id, feature, source_location, precondition, expected_behavior, priority
```

## Phase 2 Corrections

### missing_precondition

```
Test case {id} is missing setup declarations:
  Missing: {list_of_missing_setups}

For each missing item, add an explicit entry to preconditions or test_data:
- Mock object → add to preconditions: "Mock <name> configured to return <value>"
- Server state → add to preconditions: "Server started with <args>"
- Test data value → add to test_data: "<var_name>": "<concrete value> (<type>)"
- Model/path → add to preconditions: "Model <name> accessible at <path>"
```

### vague_assertion

```
Test case {id} has a vague expected result: "{current_expected}"

Replace with a concrete assertion:
- If checking a value: "self.assertEqual({actual}, {specific_value})"
- If checking a range: "self.assertTrue({min} <= {actual} <= {max})"
- If checking a shape: "self.assertEqual({tensor}.shape, ({dims}))"
- If checking an exception: "with self.assertRaises({ExceptionType}): ..."
- If checking approximate equality: "torch.testing.assert_close({actual}, {expected}, rtol=1e-4, atol=1e-4)"
```

### inconsistent_trace

```
Test case {id} has inconsistent step chaining:
Step {n} produces: {output_var}
Step {n+1} expects: {missing_input}

Fix by:
1. Rename variables to establish a clear chain: result_1 → result_2 → final_result
2. If a step is out of order, reorder steps
3. If a dependency is missing, insert a step to produce it
```

### invalid_test_logic

```
Test case {id} has insufficient eviction/capacity pressure:
  target_pages: {target_pages}
  filler_pages_per_request: {filler_pages_per_req}
  fillers: {filler_count}
  total_demand: {total} pages, capacity: {capacity} pages
  overflow: {overflow} pages < target_pages {target_pages} pages → DESIGN FLAW

Fix strategy (in priority order):
1. INCREASE filler size (e.g. 1024→8000 tokens) to create more pages per request
2. INCREASE filler count to push more total pages into L1
3. REDUCE target size (e.g. 5000→3000 tokens) to reduce pages that need evicting
4. REDUCE L1 fraction (e.g. 0.05→0.02) to shrink capacity
5. After fix: verify overflow >= target_pages AND margin >= 2 filler requests

Goal: total_demand - capacity >= target_pages (all target pages evicted)
      plus safety margin of at least 2 extra filler requests
```

### oracle_gap (Phase 2 — NEW)

```
Test case {id} has assertions that pass even when its target feature flag is absent.

  Feature flag: {flag}
  Current assertions check: {what_assertions_check}
  Missing: {what_would_differ_if_flag_absent}

Fix (three options, try in order):
A. **Use /server_info (preferred):** Add a step `GET {base_url}/server_info -> server_info`
   and assert on the specific flag field. This is deterministic, structured, and
   independent of log format. See `references/patterns.md` §"Oracle Verification Patterns"
   for flag name mapping (hyphens -> underscores, bool -> Python True/False).
B. **Log keyword (fallback):** For flags not exposed via /server_info (e.g., env vars),
   add a log capture step and check for a keyword unique to that flag's activation.
   Use value-specific patterns (e.g., `topk=4`, not just `topk`).
C. **Skip:** If neither A nor B works, mark the TC as "skipped" in _correction_log
   with reason: "inherent E2E limitation — {flag} effect not observable at output layer"

This check runs at TC design time (Phase 2), catching oracle gaps BEFORE
script generation.
```

### structural_flaw (Phase 2 — NEW)
```
Test case {id} has a structural design flaw that field-level corrections cannot fix:
  {description_of_flaw}

This test case MUST be regenerated from its source test point (re-read the TP, re-apply Plant G rules). Do NOT restart the entire phase.
Do NOT attempt to fix it by editing fields — that will cause C4 overshoot.

Regeneration instructions:
1. Re-read the test point to understand what the test must prove.
2. Identify the measurement gap: what observable signal proves the claimed behavior?
3. Redesign the step sequence to produce that signal.
4. If a direct measurement is impossible (e.g. cannot isolate L1 vs L2 hit),
   either: (a) change the test scope to what IS measurable, or
   (b) split into multiple test cases each proving one part.

Examples of structural flaws:
- TC claims to verify L2 hit but only measures total latency improvement.
- Precondition says "cache pre-populated" but Step 1 populates it (contradiction).
- Expected result depends on a metric that doesn't exist or doesn't isolate the target.
```

### vc_json_schema (Phase 2 VC)

```
Test case(s) {id_list} have missing required fields or duplicate keys in expected_results.

Fix by:
1. Missing fields: add the required field with appropriate content
   Required: id, test_point_id, title, description, preconditions, test_data, steps, expected_results, teardown
2. Duplicate keys: merge the two assertions into one (if same key) or rename one (if different assertions)
```

### vc_trace_crosscheck (Phase 2 VC)

```
Test case {id} has variable trace breaks:
  Orphaned keys in expected_results (no step produces them): {list}
  Unused variables produced by steps (no assertion references them): {list}

Fix by:
1. Orphaned keys: either (a) add a step that produces the missing variable, or
   (b) rename the expected_results key to match an existing step output
2. Unused variables: either (a) add an assertion that references the variable, or
   (b) remove the unused step output if it's genuinely not needed
```

### syntax_error

```
The compiler reported at line {line}:
{error_message}

Read the line and its surrounding context. Common fixes:
- Missing import: add `from {module} import {name}` at top
- Indentation: align with surrounding block
- Name error: check the variable/function name against source
- Type error: check the expected type of the argument
```

### weak_oracle

```
Test method {test_name} has {n} assertion(s) but should verify:
{list_of_properties_to_verify}

Add concrete assertions:
- For function return value: `self.assertEqual(result, expected_value)`
- For tensor output: `torch.testing.assert_close(output, expected_tensor, ...)`
- For exceptions: `with self.assertRaises(SpecificError): ...`
- For side effects: `mock_obj.method.assert_called_once_with(args)`
```

### inconsistent_trace (script)

```
Test {test_name} references names not found in source:
{grep_results_showing_missing_names}

Fix by:
1. Search the actual source module for the correct name: grep for similar names
2. Update the import or direct reference to use the real name
3. If the function was renamed/removed, find the replacement API
```

### missing_backend_flag (NPU/AMD)

```
The test is for {platform} but the server launch is missing the backend flag.

For NPU (Ascend):
  Add to other_args: "--attention-backend", "ascend"

For AMD (ROCm):
  Add to other_args: "--attention-backend", "rocm"

This flag is REQUIRED — the server will not use the correct hardware without it.
```

### wrong_ci_registration

```
The test is for {platform} but uses the wrong CI registration.

For NPU tests:
  Replace register_cuda_ci / register_cpu_ci with register_npu_ci
  Use NPU suites: "stage-b-test-1-npu-a2" (per-commit) or "nightly-*-npu-a3" (nightly)
  Import: from sglang.test.ci.ci_register import register_npu_ci

For CUDA tests:
  Replace register_npu_ci with register_cuda_ci
  Use CUDA suites: "stage-b-test-1-gpu-small" / "stage-b-test-1-gpu-large"
  Import: from sglang.test.ci.ci_register import register_cuda_ci
```

### missing_coverage

```
Test script {filename} has {N_methods} test methods but Phase 2 has {M_cases} text test cases.
Missing methods for: {list_of_uncovered_case_ids}

For each uncovered text test case:
1. Read the text test case from test-out/phase2/test-cases.json
2. Generate a test_<title> method following the script's existing patterns
3. Copy imports, fixtures, and assertion style from existing methods in the file
4. Ensure the new method's name matches the text case title (snake_case)
```

### wrong_file_placement

```
The test file is in the wrong directory for {platform}.

For NPU:
  Move to: test/registered/ascend/<category>/test_npu_<feature>.py
  Categories: basic_function, interface, llm_models, vlm_models, embedding_models, reward_models, rerank_models

For CUDA:
  Move to: test/registered/<category>/test_<feature>.py

Rename file to match platform naming convention if needed.

## Phase 3 SC Corrections (semantic — added for logic-gap detection)

### config_mismatch

```
Test method {test_name} uses server args that don't match its source TC {tc_id}.

  TC {tc_id} test_data.other_args requires: {tc_args}
  Actual server config used: {actual_args}
  Mismatch: {list_of_differing_flags}

Fix by:
1. If the method shares a setUpClass: move it to a separate test class with its own
   setUpClass that launches the server with the TC-specified args.
2. If the method has its own server launch: correct the args to match the TC.
3. If setUpClass uses a sensible default shared by multiple TCs: document the
   configuration mapping in a class-level comment and verify no TC's core claim
   depends on a different flag value.

Example: TC-004 requires topk=4 but setUpClass uses topk=1 — the method can't
verify multi-branch draft behavior. Give it a dedicated server launch.
```

### oracle_gap (Phase 3)

```
Test method {test_name} passes all assertions even when its target feature flag
is removed or set to a different value.

  Feature flag: {flag}
  Current assertions check: {what_assertions_check}
  Missing: {what_would_change_if_flag_absent}

Fix (three options, try in order):
A. **Use /server_info (preferred):** Add `requests.get(f"{url}/server_info")` and assert
   on the specific flag field. See `references/patterns.md` §"Oracle Verification Patterns".
B. **Log keyword (fallback):** For flags not exposed via /server_info (e.g., env vars),
   add a log capture assertion. Use value-specific patterns (e.g., `topk=4`, not `topk`).
C. **Skip:** If neither A nor B works, and this TC has been flagged for oracle_gap in a
   prior iteration (2-strike), mark as "skipped" in _correction_log with reason:
   "inherent E2E limitation — flag effect not observable at output layer".

This extends weak_oracle beyond assertion count to assertion relevance.
```

### step_fidelity

```
Test method {test_name} does not cover all steps from source TC {tc_id}.

  TC steps: {list_of_tc_steps}
  Covered in code: {list_of_covered}
  Missing: {list_of_missing}

Fix by:
1. For each missing step: add the corresponding code block (data setup,
   intermediate check, or assertion) to the test method.
2. If a step is intentionally skipped (e.g., log parsing not feasible
   in this test structure): add a comment explaining why and note the
   coverage gap.
```

### teardown_leak

```
Test method {test_name} launches a server but does not guarantee cleanup
on all exit paths.

  Server launch: {line_number}
  Cleanup: {present_or_missing}
  Risk: if an assertion fails between launch and cleanup, kill_process_tree
        is never called, leaking the server process and port.

Fix by:
Wrap the server launch and all subsequent operations in try/finally:

  proc = popen_launch_server(...)
  try:
      ... test logic ...
  finally:
      kill_process_tree(proc.pid)
```
```
