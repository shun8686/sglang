# Convergence Criteria & Rollback

Based on CyberCorrect Section III-D: Convergence Judge with Rollback, Algorithm 1.
Adapted for the dual-subagent (Sub-A ↔ Sub-B) alternating correction cycle with adjacent-iteration comparison. Main only judges — never detects or fixes.

## State Tracking

At each iteration t within a phase:
- `s_t` = weighted severity sum: s_t = 0.8×N_critical + 0.5×N_moderate + 0.2×N_minor
  This follows the paper's Eq.5: `s = w1·s_SC + w2·max(s_VC) + w3·(1-min(v_j))`
  which aggregates multiple error signals into a single continuous score.
- `d_t` = number of errors detected (for reporting only; convergence uses s_t)
- `error_types_t` = set of error type labels detected at iteration t
- `Version Buffer` = list of all outputs {(y_1, s_1), (y_2, s_2), ..., (y_t, s_t)}

**IMPORTANT**: All convergence decisions use s_t (weighted sum), NOT d_t (raw count).
Example: Iter 1 has 14 critical + 8 moderate + 4 minor → s = 0.8×14+0.5×8+0.2×4 = 16.0.
Iter 2 has 4 critical + 3 moderate + 8 moderate → s = 0.8×4+0.5×11+0.2×0 = 8.7.
The continuous s_t value reflects partial improvement that the old max-based s_t could not.

## Algorithm

```
1. y_0 ← Generate(x)                              # Initial generation (Plant G)
2. Buffer ← {y_0}                                  # Initialize version buffer
3. for t = 1 to T_max do:
4.   (τ_t, s_t) ← Detect(y_{t-1})                  # Detect on PRIOR output (pre-fix state)
5.   if s_t == 0 then return y_{t-1}               # [C1] Converged — no errors
6.   if t >= T_max then                            # [C5] Exhausted
7.     return argmin_{y in Buffer} s(y)
8.   if t >= 3 AND τ_t == τ_{t-2} AND τ_t != τ_{t-1}  # [C3] Oscillation
9.     return argmin_{y in Buffer} s(y)
10.  if s_t > s_{t-1} + 3 then                     # [C4] Overshoot (δ=3)
11.    run C4 Diagnostic Procedure (below)          # Classify before acting
12.    if TRUE_C4 (correction failure): rollback; continue  # Try different strategy
13.    if DEPTH_SPIRAL: continue normally           # Do NOT rollback
14.  if s_t <= s_{t-1} AND s_t >= s_{t-1} - 1 then # [C2] Plateaued
15.    return y_{t-1}                               # Marginal improvement, stop
16.  u_t ← C(τ_t, s_t, ℓ_t)                        # Type-directed correction
17.  y_t ← apply_correction(x, y_{t-1}, u_t)        # Corrected output
18.  Buffer ← Buffer ∪ {y_t}                        # Save to version buffer
19. end for
20. return argmin_{y in Buffer} s(y)                # [C5] Lowest severity version
```

**Note**: t=1 has no s_0 for comparison — C2/C4 are skipped, execution falls to correction (line 16). C3 only activates at t≥3 (needs two prior error type sets).

## Five Convergence Criteria

Evaluated top-to-bottom; first match wins.

### C1: Perfect Convergence
```
s_t == 0  →  CONVERGED
```
No errors remain (weighted severity is zero). Proceed to next phase.

### C5: Maximum Iterations
```
t >= T_max  →  EXHAUSTED
```
T_max defaults: Phase 1 = 3, Phase 2 = 5, Phase 3 = 3.
Return argmin_{y in Buffer} s(y) — version with lowest weighted severity.

### C3: Oscillation Detection
```
t >= 3 AND error_types_t == error_types_{t-2} AND error_types_t != error_types_{t-1}
→  OSCILLATION
```
Same error type set reappears every other iteration — the system is oscillating between
two states without making progress. Return argmin_{y in Buffer} s(y).

Only valid at t ≥ 3 (requires at least two prior error type sets for comparison).
At t=1 and t=2, C3 is skipped.

### C4: Overshoot Rollback (does NOT terminate phase)
```
s_t > s_{t-1} + 3  →  POTENTIAL OVERSHOOT  (δ = 3)
```
The weighted severity increased by more than 3 points compared to the previous iteration.
This is a large jump — roughly 4 extra critical errors or 6 extra moderate errors.
When triggered, run the **C4 Diagnostic Procedure** below before deciding.

The δ=3 threshold is set high to avoid false C4 triggers from detector-strictness differences
between Main and Sub in the alternating cycle. A small s_t increase (≤3) from a stricter
detector is normal and falls to the "Otherwise → continue" path.

#### C4 Diagnostic Procedure

When s_t > s_{t-1} + 3, the **main agent** must diagnose whether this is a depth spiral
or true correction failure BEFORE deciding C4 rollback.

Note: "ID" below means TC-ID for Phase 2, TP-ID for Phase 1. The logic is identical —
compare whether the same artifacts are flagged across iterations.

**Step A — Compare error locations (IDs) across iterations:**

```
For each error e in τ_t:
  - Was this exact ID flagged in τ_{t-1}?
  - If YES → mark as "persistent" (same test case still has issues)
  - If NO → mark as "new location" (subagent found issues in different test cases)
```

**Step B — Trace new errors back to the correction diff:**

```
For each error e marked "new location" in τ_t:
  - Did we modify this exact field/line in the previous correction step?
  - Grep the changed TC to check if the error references a field we touched
  - If YES → mark as "correction-induced" (our fix created a new bug)
  - If NO → mark as "pre-existing but previously undetected"
```

**Step C — Classify the s_t increase:**

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| ≥ 50% of errors are "persistent" (same IDs as t-1) | **True correction failure** — fixes didn't work | C4 ROLLBACK: restore from backup, try different strategy |
| ≥ 1 error is "correction-induced" (traces to modified field) | **Correction introduced new bug** — fix was wrong | C4 ROLLBACK for that specific TC only, not the whole file |
| ≥ 80% of errors are "new location" AND "pre-existing" | **Depth spiral** — subagent found issues we never touched | Continue normally, apply corrections |
| Error nature shifted from surface (missing fields) to deeper (measurement validity, boundary conditions) | **Depth spiral** — subagent reading more carefully | Continue normally |

**Step D — Decision rule summary:**

```
IF (persistent_count / total_errors >= 0.5) OR (correction_induced_count >= 1):
    → TRUE C4: rollback to .backup, try different correction strategy, continue
ELSE:
    → DEPTH SPIRAL: continue normally, do NOT rollback
```

#### C4 Alternative Strategies (when TRUE C4 triggers rollback)

After rollback, pick a different correction approach for the next iteration:

**Strategy A — Escalate correction intensity:**
The standard template says "targeted edit." Escalate to "full regeneration."
Example: `vague_assertion` was fixed by replacing the assertion text, but it's still vague.
Instead, regenerate the entire expected_results block from scratch.

**Strategy B — Switch from field-level to structural fix:**
If the same artifact is flagged in consecutive iterations with the same error type,
the field-level fix is treating a symptom. Go back to the Plant G step and
regenerate the entire artifact. Treat it as `structural_flaw` even if not flagged as one.

**Strategy C — Refine the subagent's detection focus** (when the subagent keeps finding
new errors of the same type at different locations):
Send a refined prompt via SendMessage to the existing subagent:
- Add: "Focus ONLY on errors that would cause test failure.
  Do NOT flag issues that are merely imprecise but still distinguish pass from fail."
If the subagent is truly unavailable, launch a replacement with the refined focus instructions.

**Strategy D — Accept and document:**
If the error stems from an inherent limitation (e.g., E2E test can't prove internal timing),
mark it as "skipped" in _correction_log with a clear reason. The downstream phase will
see it in `_unresolved_errors` and account for it.

**Decision rule for picking a strategy:**
```
Same ID + same τ type twice → Strategy B (treat as structural, regenerate)
Same τ type + different IDs → Strategy C (refine subagent focus)
Surface errors persist → Strategy A (escalate intensity)
Inherent limitation → Strategy D (accept and document)
```

### C2: Plateau
```
s_t <= s_{t-1}  AND  s_t >= s_{t-1} - 1  →  PLATEAUED
```
The correction improved quality, but only marginally (≤ 1.0 s_t reduction).
Further iterations are unlikely to produce meaningful gains — stop and move on.

C2 is checked AFTER C4, so overshoot cases are excluded first.
C2 returns the current (pre-fix) output y_{t-1} — no further corrections are applied.

## Parameter Values

| Parameter | Value | Rationale |
|---|---|---|
| T_max | P1=4, P2=5, P3=5 | Phase 3 needs room for oracle_gap 2-strike skip (2 rounds to trigger + final round to converge) |
| δ (overshoot threshold) | 3 | ~4 critical or ~6 moderate errors. High to avoid false C4 from detector-strictness differences in alternating Main↔Sub cycle |
| ε (plateau window) | 1 | s_t improvement ≤ 1 is considered marginal; stop |

## Version Buffer

Maintain a buffer of ALL outputs, each with their s score (weighted sum):
```
Buffer = {(y_0, s_0=None), (y_1, s_1), (y_2, s_2), ..., (y_t, s_t)}
```
Note: y_0 (initial Plant G output) has no s_0 since detection only starts at Iter 1.
On oscillation or exhaustion, select the entry with the lowest s score.
This follows the paper's `argmin_{y in Buffer} s(y)`.

**Practical implementation**: The `.backup` file stores the immediately-previous version
(for C4 rollback). To support C3/C5 argmin selection, keep each iteration's output:
```
test-out/phaseN/<file>.iter1
test-out/phaseN/<file>.iter2
...
```
Or track scores in conversation context: `Buffer = [(y_1, s_1=10.8), (y_2, s_2=2.9), ...]`
and restore the winning version by knowing which iteration produced it.

## Rollback Protocol

1. Before correction: copy the output file to a backup.
   `cp test-out/phaseN/<file> test-out/phaseN/<file>.backup`
2. Apply correction → produce y_t (overwrite the output file).
3. Detect → compute s_t = 0.8×N_crit + 0.5×N_mod + 0.2×N_min.
4. If TRUE C4 (correction failure, not depth spiral):
   - Restore from backup: `cp test-out/phaseN/<file>.backup test-out/phaseN/<file>`
   - Record which correction strategy caused the failure. Do NOT retry it.
   - Continue to next iteration (do NOT stop the phase) with a different strategy.
5. If DEPTH SPIRAL or s_t improved: add (y_t, s_t) to Buffer, remove the backup file.

## Fail-Open Behavior

If a phase exhausts without converging:
1. Select y_best = argmin_{y in Buffer} s(y)
2. Add `_unresolved_errors` field listing remaining errors
3. Continue to next phase (never block pipeline)
4. Downstream phase MUST read `_unresolved_errors` and account for them
