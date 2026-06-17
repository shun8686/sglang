# Error Type Taxonomy

Maps CyberCorrect error types τ to the test generation domain. Each error type has a detection method, severity level, and correction strategy.

## Phase 1 Error Types (Test Points)

| τ | Name | Detect By | Severity | Correction |
|---|---|---|---|---|
| `missing_scenario` | (SC) Workflow scenario gap — a pipeline stage, configuration variant, or error path has no test point | Subagent checks: for each workflow stage (draft, verify, extend), each configuration axis (topk, num_steps, num_draft_tokens, attention_mode), each platform constraint, and each error branch in source — is there a corresponding TP? | **moderate (s=0.5)** | Add a workflow test point covering the missing scenario. Do NOT add per-API TPs — frame it as a user-observable scenario. |
| `missing_boundary` | Boundary values not enumerated | For each parameter, list {empty, zero, max, negative, edge} and check coverage | **minor (s=0.2)** | Add boundary variant test points for each parameter |
| `missing_precondition` | Setup conditions missing | Check each feature against required state/dependencies from source | **moderate (s=0.5)** | Review constructor/call signatures, add missing preconditions |
| `missing_platform_constraint` | Platform-specific limitations not captured | Per-TP, per-constraint check against `references/platform-constraints.md`. For each relevant constraint: is it reflected in boundary_conditions, error_paths, and expected_behavior? | **critical (s=0.8)** | Re-read `platform-constraints.md`, add platform-specific values to TP boundaries and error paths |
| `factual_error` | expected_behavior or precondition contains incorrect technical claim | Cross-check each test point's claims against design doc + source code. Does the flag/API actually do what the test point says? Example: 'L1/L2 allocated' when flag only controls L2. | **critical (s=0.8)** | Re-read the relevant doc/source section, correct the factual claim. For flag interactions: check which tier each flag actually controls. |
| `unverifiable_claim` | Test point asserts a behavior with no observable external signal from user perspective | For each test point's expected_behavior: can it be verified from outside the system (output, metrics, exception, status code)? If the claim requires inspecting internal state (tensor values, KV cache contents, memory layout, compilation status, specific internal API calls) that cannot be surfaced → the claim is unverifiable in E2E testing. **IS unverifiable**: 'draft KV cache has correct content', 'internal tensor shape = X', 'function runs WITHOUT torch.compile', 'cache_loc dtype is int32'. **NOT unverifiable**: claims about latency/throughput/accept_rate/accuracy/output correctness/log output — these are measurable user-observable signals. | **critical (s=0.8)** | **Delete the TP** if its ONLY unique contribution is an internal implementation detail (other TPs cover the end-to-end behavior it protects). If the TP also contains a valid user-observable scenario, delete just the unverifiable claims and keep the rest; otherwise delete the entire TP. Adding a [VERIFIABILITY NOTE] is no longer sufficient — internal-only TPs waste downstream phase effort. |
| `vc_id_uniqueness` | (VC) Duplicate or malformed test point IDs/fields | Python script: check IDs are unique, priorities in {P0,P1,P2}. NOT LLM judgment. (source_location format is NOT validated by VC — semantic correctness of source_location is checked by SC factual_error.) | **critical (s=0.8)** | Fix duplicate IDs or malformed priority values |
| `logical_inconsistency` | Preconditions, expected_behavior, and error_paths form an incoherent logic chain | Three checks per TP: (1) **Precondition-exclusion**: does any error_path describe a condition already excluded by a precondition? (e.g., precondition says "valid draft model path" but error_path says "missing draft model → config error" — the error_path scenario is unreachable). (2) **Behavior reachability**: can the expected_behavior actually be observed given the preconditions? If preconditions describe a running server but expected_behavior describes startup behavior, the chain is broken. (3) **Error-path relevance**: is each error_path genuinely reachable from within the TP's scope? A TP about `--speculative-num-steps` having an error_path about "OOM" is weakly connected — the TP doesn't test memory allocation. | **critical (s=0.8)** | Fix the contradiction: remove the excluded scenario from error_paths, add a missing precondition, or split into two TPs if the contradictory cases are both worth testing. |
| `duplicate_tp` | Two or more TPs describe substantively overlapping scenarios | Pairwise comparison of all TPs on: feature + preconditions + expected_behavior. **Merge** when TP-A's scope is a subset of TP-B's (e.g., TP-005 tests "verify stage" and TP-023 tests "greedy verification constraint" — same pipeline step, same NPU constraint). **Keep both** when they test genuinely different boundary conditions or different phases of a pipeline (e.g., draft vs verify). All TPs are workflow dimension — no API vs workflow distinction. | **moderate (s=0.5)** | **Merge procedure (do NOT delete — preserve IDs):** 1. Move any unique boundary_conditions and error_paths from the narrower TP into the broader (surviving) TP. 2. Add `"merged_into": "TP-XXX"` to the narrower TP's fields. 3. Add a _correction_log entry. Downstream phases (Phase 2) MUST skip TPs with `merged_into` set. Never renumber IDs. |
| `vc_json_schema` | (VC) Test point missing required fields | `python -c` one-liner to check field presence. NOT LLM judgment. | **moderate (s=0.5)** | Add the missing field(s) to the affected test point |

## Phase 2 Error Types (Text Test Cases)

| τ | Name | Detect By | Severity | Correction |
|---|---|---|---|---|
| `vague_assertion` | Expected result too vague | Check: does "expected" contain specific values/operators/shapes? | **moderate (s=0.5)** | Replace with concrete assertEqual/assertClose/assertIn with explicit values |
| `missing_precondition` | Test data or mock setup missing | Check: are all external dependencies explicitly set up? | **moderate (s=0.5)** | Insert data initialization and mock setup steps |
| `inconsistent_trace` | Step outputs not used by later steps | Verify step N's output variable appears as input in step N+1 or assertion | **moderate (s=0.5)** | Fix variable references across steps, reorder if needed |
| `invalid_test_logic` | Test parameters cannot achieve claimed effect | For tests involving threshold/capacity/eviction/overflow: compute the numerical outcome. Does total_demand > capacity with sufficient margin? Do filler requests create enough pressure to evict ALL target pages (not just a fraction)? Example: test claims 'L1 overflow forces L2 hit' but calculation shows overflow=1 page when 24 target pages need eviction — margin too small. **Tokenization hazard**: Tests that use character-counted prompts (e.g. `"a" * 128`) to target specific token counts are invalid — LLM tokenizers produce far fewer tokens than the character count for repeated single characters. Always use the model's tokenizer to verify token counts, or describe the prompt semantically rather than by character count. | **critical (s=0.8)** | Recalculate with larger filler size, more filler requests, or smaller L1 fraction to achieve sufficient eviction pressure. Ensure overflow >= target_pages for full eviction guarantee. For tokenization issues: add precondition noting tokenizer verification is required. |
| `structural_flaw` | Test design cannot be fixed by field edits; requires regeneration | The test case has an inherent design problem: steps don't actually test the claimed feature, contradictory preconditions reveal a flawed sequence, or the measurement method cannot isolate the target behavior. Field-level fixes (vague_assertion, missing_precondition) only patch symptoms. **NOT structural_flaw**: E2E tests measuring proxy metrics (TTFT, hit rate, output correctness) instead of internal implementation details (exact prefetch start time, per-layer overlap). This is normal E2E testing — external behavior is the valid target. Similarly, tests that require parameter tuning for specific hardware capacities are not structural flaws if the tuning notes are documented. | **critical (s=0.8)** | Regenerate this single test case from its source test point (re-read the TP, re-apply Plant G rules). Do NOT attempt field-level correction — it will fail or produce C4 overshoot. Regenerate with corrected understanding of what the test must prove and how to prove it. If same TC-ID flagged again → mark "skipped". |
| `oracle_gap` | Assertions pass even when the target feature's key configuration is absent | For each expected_results assertion: mentally remove the feature's key flag(s) from test_data. Would the assertion still pass? If yes, the test case cannot distinguish the feature working from being absent. Catches the gap at design time — before script generation. Example: TC expects to verify `--speculative-eagle-topk=4` but assertions only check HTTP 200 + text non-empty, which pass identically with topk=1. | **critical (s=0.8)** | Add an assertion that depends on the feature's observable signal (specific metric value range, log keyword unique to that flag, or output characteristic that demonstrably changes with the flag value). If no observable signal exists for that flag, mark the TC as `skipped` in _correction_log with reason: `"inherent E2E limitation — flag effect not observable at output layer"`. |
| `vc_json_schema` | (VC) TC missing required fields or duplicate keys | `python -c` one-liner for field presence + duplicate key detection. NOT LLM judgment. | **moderate (s=0.5)** | Add missing fields or deduplicate keys |
| `duplicate_tc` | Two or more TCs describe substantively overlapping test procedures | Pairwise comparison of test cases on: test_point_id + preconditions + steps + expected_results. Flag when TC-A's steps are a subset of TC-B's, or when both TCs map to the same TP and exercise identical scenarios. Higher bar than TP dedup — TCs that test different boundary values of the same TP are NOT duplicates. | **moderate (s=0.5)** | Merge by keeping the more thorough TC, remove the narrower one. If both test meaningfully different boundary conditions, keep both. |
| `logical_inconsistency` | Steps, expected_results, and teardown form an incoherent logic chain | Three checks per TC: (1) **Step chain**: does each step produce a result that a later step or expected_result actually uses? If step 3 produces `result_3` but no step or assertion references it, the chain is broken. (2) **Precondition-step coherence**: do the preconditions enable the first step? If preconditions say "server not started" but step 1 calls a server API, the TC is contradictory. (3) **Teardown completeness**: does teardown clean up everything the preconditions and steps created? A TC that starts a server but has empty teardown is logically incomplete. | **critical (s=0.8)** | Fix the chain: remove unused steps, insert missing connections, or split the TC. |
| `vc_trace_crosscheck` | (VC) Step output variables don't match expected_results keys | Python script: extract "→ produces" vars, cross-ref against expected_results keys. Finds orphaned keys/unused vars deterministically. | **critical (s=0.8)** | Fix variable name mismatches or add missing step outputs |

## Phase 3 Error Types (Test Scripts)

### VC Errors (tool-grounded, deterministic)

| τ | Name | Detect By | Severity | Correction |
|---|---|---|---|---|
| `syntax_error` | Script fails to compile | `python -m py_compile <file>` exit code != 0 | **critical (s=0.8)** | Read compiler stderr, fix the exact line number |
| `inconsistent_trace` | Imported names don't match source | `grep` for function/class names used in test against actual source exports | **moderate (s=0.5)** | Replace with correct import paths from source |
| `weak_oracle` | No assertions or only `assertTrue(exists)` | `grep -c "assert"` per test method — must be >= 1 with specific values | **moderate (s=0.5)** | Add `self.assertEqual(expected, actual)` with concrete expected values |
| `missing_coverage` | Text test case has no corresponding test method | Compare test method count against Phase 2 test case count | **minor (s=0.2)** | Add missing `def test_*` method for uncovered text case |
| `missing_backend_flag` | NPU/AMD backend flag not in server args | For NPU: check `--attention-backend ascend` in other_args. For AMD: check `--attention-backend rocm` | **critical (s=0.8)** | Add the required backend flag to other_args |
| `wrong_ci_registration` | CI registration doesn't match platform | Check: NPU tests use `register_npu_ci`, CUDA use `register_cuda_ci` | **critical (s=0.8)** | Replace with correct registration function and suite |

### SC Errors (LLM-driven, semantic — added to catch logic gaps VC misses)

| τ | Name | Detect By | Severity | Correction |
|---|---|---|---|---|
| `config_mismatch` | Test method's actual server args don't match its source TC's test_data configuration | Cross-reference each test method's server launch args against the corresponding TC's `test_data.other_args` (or test_data fields). Flag when a method shares a `setUpClass` that uses a different configuration than the TC requires (e.g., TC-004 requires `topk=4` but runs on a `topk=1` shared server). | **critical (s=0.8)** | Give the test method its own server launch with the correct args, or create a separate test class with a matching `setUpClass`. |
| `step_fidelity` | Test method's steps don't cover the source TC's procedural flow | For each TC step: is there a corresponding code block in the test method? Flag missing setup steps, skipped assertions, or consolidated logic that loses coverage. | **moderate (s=0.5)** | Add the missing step logic or assertion to the test method. |
| `oracle_gap` | Assertions pass even when the claimed feature is absent | For each test method: remove the feature's key flag(s) from server args in mental simulation. Would the assertions still pass? If yes, the oracle cannot distinguish the feature working from it being absent. This extends `weak_oracle` beyond assertion count to assertion *relevance*. Example: `test_tc004` with topk=1 config still passes all assertions — the oracle can't tell topk=4 from topk=1. | **critical (s=0.8)** | Add an assertion that depends on the feature's observable signal (metric value, log keyword, output characteristic that changes with the flag). If no observable signal exists for that flag, mark the TC as `skipped` in _correction_log. **2-strike rule**: If the same TC-ID is flagged for `oracle_gap` in two consecutive iterations (after a fix attempt), stop fixing and mark as `skipped` with reason: `"inherent E2E limitation — flag effect not observable from output layer"`. This prevents infinite depth-spiral on structural E2E limitations that no amount of assertion refinement can resolve. |
| `teardown_leak` | Server processes or resources not cleaned up in all exit paths | Check: does every `popen_launch_server` have a corresponding `kill_process_tree` in a `finally` block (or `tearDownClass` for shared servers)? Flag missing cleanup on error paths. | **moderate (s=0.5)** | Wrap server launch in try/finally with kill_process_tree in the finally block. |
| `wrong_file_placement` | Test file in wrong directory | For NPU: must be `test/registered/ascend/<category>/test_npu_*.py` | **moderate (s=0.5)** | Move to correct directory, rename with platform prefix |

## Severity Mapping to s score

| Severity | s Value | Correction Intensity |
|---|---|---|
| critical | 0.8 | Full regeneration from error location |
| moderate | 0.5 | Targeted edit of specific section |
| minor | 0.2 | Minimal insert/addition |

The aggregate severity s = 0.8×N_critical + 0.5×N_moderate + 0.2×N_minor (weighted sum, following the paper's Eq.5 aggregation principle).

## Test Point Priority Rubric

Priorities are assigned by **consequence radius** — what breaks and for whom if this test point fails:

| Priority | Judgment Rule | Typical Scenarios |
|----------|-------------|-------------------|
| **P0** | The feature is unusable or produces wrong output. All users are affected. | Startup crash, incorrect/corrupted output, measurable regression that defeats the purpose of the feature, silent fallback to a degraded mode without user awareness. |
| **P1** | A common configuration path or important error handling is broken. Affects users who use a specific flag, parameter value, or sub-feature combination. | Parameter variant produces degraded output, missing/bad config causes obscure error, concurrent-request handling bug, long-running-operation degradation, platform-specific flag missing or wrong. |
| **P2** | Impact is narrow — advanced/optional features, observability gaps, or rarely-used configuration combinations. | Optional tuning knob malfunction, metrics/logs missing or wrong, edge parameter values, environment variables with narrow scope. |

**Decision process (top-to-bottom, first match wins):**

```
1. Failure breaks the feature entirely or produces wrong results for all users?  → P0
2. Failure degrades behavior for users on a specific config/param path?          → P1
3. Failure affects only optional/advanced sub-features or observability?         → P2
```

When in doubt between two levels, choose the higher (more critical) one.

## τ→Action Quick Reference

Used by the correction agent when applying fixes. Each τ maps to a specific location ℓ and a concrete action.

| Phase | τ | Location ℓ | Action |
|---|---|---|---|
| 1 | missing_scenario | workflow stage / config axis / platform constraint | Add workflow test point for the missing scenario; never add per-API TPs |
| 1 | missing_boundary | test point ID + field | Append to boundary_conditions |
| 1 | missing_precondition | test point ID | Append to precondition array |
| 1 | missing_platform_constraint | test point ID + param | Update boundary_conditions + error_paths |
| 1 | factual_error | test point ID + field | Replace the specific field value |
| 1 | unverifiable_claim | test point ID | Delete TP if internal-only; otherwise delete unverifiable claims. Adding a [VERIFIABILITY NOTE] is not sufficient — see correction-templates.md. |
| 1 | vc_id_uniqueness | test point ID | Fix duplicate ID or malformed field |
| 1 | logical_inconsistency | test point ID + field | Fix contradiction: remove excluded scenario, add precondition, or split TP |
| 1 | duplicate_tp | test point ID pair | Merge by keeping broader TP; remove narrower unless boundary conditions differ |
| 1 | vc_json_schema | test point ID | Add missing field(s) |
| 2 | vague_assertion | test case ID + key | Replace with concrete assertion |
| 2 | missing_precondition | test case ID | Append to preconditions/test_data |
| 2 | inconsistent_trace | test case ID + step | Fix variable name or reorder steps |
| 2 | invalid_test_logic | test case ID | Recalculate test_data numerical params |
| 2 | logical_inconsistency | test case ID | Fix chain: remove unused steps, connect missing refs, or split TC |
| 2 | duplicate_tc | test case ID pair | Merge by keeping more thorough TC; keep both if boundary conditions differ |
| 2 | oracle_gap | test case ID | Add feature-dependent assertion; if no observable signal exists, mark TC as skipped |
| 2 | structural_flaw | test case ID | Regenerate this single test case from its source test point (re-read the TP, re-apply Plant G rules). Do NOT restart the entire phase. If same TC-ID flagged again → mark "skipped". |
| 2 | vc_json_schema | test case ID | Add missing fields or deduplicate keys |
| 2 | vc_trace_crosscheck | test case ID + step | Fix variable name mismatch |
| 3 | syntax_error | file + line | Edit tool: fix specific line |
| 3 | missing_backend_flag | file | Edit other_args to add flag |
| 3 | wrong_ci_registration | file | Edit to replace registration call |
| 3 | wrong_file_placement | file | Move to correct directory |
| 3 | inconsistent_trace | file + import | Edit to fix import name |
| 3 | weak_oracle | test method | Edit to insert assertions |
| 3 | config_mismatch | test method + setUpClass | Give the method its own server launch with correct args, or create a separate test class |
| 3 | oracle_gap | test method | Add assertion dependent on feature's observable signal; if none exists, mark TC as skipped |
| 3 | step_fidelity | test method | Add missing step logic or assertion |
| 3 | teardown_leak | test method | Wrap server launch in try/finally with kill_process_tree |
| 3 | missing_coverage | file | Add test method for missing case |

## Confidence Assessment

Subagents compute confidence for every TP/TC during detection. Each dimension scores 0-1. Confidence is reported alongside s_t — it is NOT written into the output JSON.

### Phase 1 Dimensions (Test Points)

| Dimension | Measures | 1.0 | 0.5 | 0.0 |
|-----------|----------|-----|-----|-----|
| `source_grounding` | Is the TP anchored to real code/docs? | `source_location` is exact `file:line (symbol)` | File path only, no symbol | No source_location or entirely wrong |
| `observability` | Is expected_behavior user-observable? | All claims are user-observable (output, metric, exception, status, log). No internal implementation details. | Mixed: contains one internal detail alongside valid user-observable claims, but the internal detail is not the primary contribution | Claims almost entirely about internal state with no user-observable proxy. TP should be deleted. |
| `specification_completeness` | Are all fields filled with real content? | precondition ≥3, boundary ≥3, error_paths ≥2, key_factors ≥1, no placeholder text | One field is sparse or has generic entries | Multiple empty arrays, "TBD", or placeholder values |
| `platform_fidelity` | Are platform constraints reflected? | boundary + error_paths include platform-specific values | Platform constraints mentioned vaguely | No platform constraints on a platform-specific TP |
| `logical_coherence` | Is the precondition→behavior→error chain coherent? | No contradictions, error_paths are reachable from preconditions | Minor disconnect (weak error_path relevance) | precondition explicitly excludes an error_path scenario |

### Phase 2 Dimensions (Text Test Cases)

| Dimension | Measures | 1.0 | 0.5 | 0.0 |
|-----------|----------|-----|-----|-----|
| `source_grounding` | Does the TC trace back to a valid TP? | `test_point_id` references an existing, well-formed TP | References a TP with issues | No test_point_id or references nonexistent TP |
| `observability` | Are expected_results concrete and verifiable? | All assertions have specific values | One assertion is vague ("approximately correct") | Multiple `assertTrue(exists)` without concrete values |
| `specification_completeness` | Are all fields filled? | preconditions ≥3, steps ≥3, test_data populated, teardown ≥1 | One sparse field | Multiple empty or placeholder fields |
| `trace_coherence` | Do step outputs flow to later steps/assertions? | Every step result is referenced downstream | One orphaned step variable | Multiple broken traces (`inconsistent_trace` flagged) |
| `logical_coherence` | Do preconditions→steps→expected→teardown form a closed loop? | Preconditions enable step 1, teardown cleans up everything | Minor gap (unnecessary precondition) | Contradiction (preconditions say "no server", steps call API) |

### Aggregate & Rating

```
aggregate = mean of all 5 dimensions (rounded to 2 decimals)
```

| Aggregate | Rating | Action |
|-----------|--------|--------|
| ≥ 0.85 | **Strong** | Ready for next phase |
| 0.70–0.84 | **Adequate** | Usable; consider improvements next iteration |
| 0.50–0.69 | **Weak** | Must be improved before next phase |
| < 0.50 | **Insufficient** | Regenerate or significantly rework |

### Report Format

Sub reports confidence in its detection output alongside s_t:

```
Confidence (mean aggregate across all TPs): 0.78
TPs below 0.70: TP-004 (0.55, weak observability), TP-012 (0.60, weak completeness)
```
