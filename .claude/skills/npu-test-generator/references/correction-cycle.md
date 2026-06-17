# Correction Cycle — Main schedules, Sub-A + Sub-B alternate

Main NEVER detects or fixes. Main only: Plant G (initial generation), scheduling, C1-C5 judging.
Two independent subagents alternate: Sub-A handles odd iterations (1, 3, 5...), Sub-B handles even (2, 4...).
This keeps Main's context clean and ensures strict Sensor E independence from Generator G.

## Pattern

```
Iter 1: Main(Plant G) → Sub-A(detect) → Main(judge C1-C5) → Sub-A(fix)
Iter 2: Sub-B(detect) → Main(judge C1-C5) → Sub-B(fix)
Iter 3: Sub-A(detect) → Main(judge C1-C5) → Sub-A(fix)
Iter 4: Sub-B(detect) → Main(judge C1-C5) → Sub-B(fix)
...
```

- **Main**: Plant G (initial output), then ONLY C1-C5 judging + scheduling. Never detects or fixes. Never runs VC/SC scripts. The only validation Main performs after Plant G is JSON syntax: `python -c "import json; json.load(...)"`.
- **Sub-A**: Odd iterations (1, 3, 5...). Detection + fix (when Main says continue). Responsible for ALL VC and SC checks.
- **Sub-B**: Even iterations (2, 4...). Detection + fix (when Main says continue). Responsible for ALL VC and SC checks.
- **Clean slate principle**: Each detection is fully independent. A subagent sees only the current file — it does NOT read `_correction_log` and does NOT know what previous iterations found or fixed. Main must not mention prior results in the detect message. This prevents confirmation bias and ensures new errors or missed fixes are caught by fresh eyes. Applies equally to both first-contact and returning subagents — the file may have been modified by the other subagent since last visit.

## Subagent Lifecycle

Launch BOTH at the start of the correction cycle (before Iter 1).

**Main MUST wait for the designated subagent before judging C1-C5**, even if the other agent finishes first:

| Iteration | Designated subagent | What to do |
|-----------|-------------------|-------------|
| 1 (odd) | **Sub-A** | Wait for Sub-A's detection. If Sub-B finishes first, note its result but do NOT use it for Iter 1 convergence judgment. |
| 2 (even) | **Sub-B** | Wait for Sub-B's detection. If Sub-A finishes first, note its result but do NOT use it for Iter 2 convergence judgment. |
| 3+ | alternating | Same pattern — always wait for the designated agent per the alternating schedule. |

Rationale: the designated subagent is responsible for BOTH detection and fix in a given iteration. Using the wrong agent breaks the alternating cycle and can cause depth-spiral false positives or missed oscillation patterns (C3 relies on comparing the SAME subagent's error sets at t and t-2).

**BEFORE launching**: Main MUST resolve ALL `<placeholders>` in the first-launch prompt to absolute paths. The subagent runs in a background context that may not share the main agent's working directory. Key placeholders:

| Placeholder | Resolve to | Example |
|---|---|---|
| `<skill-base>` | `SKILL_BASE` absolute path (see SKILL.md Step 0) | `C:/Users/.../.claude/skills/test-generator` |
| `<source_dirs>` | Space-separated absolute paths to scanned source dirs | `/abs/path/python/sglang/srt/speculative/ /abs/path/python/sglang/srt/hardware_backend/npu/` |
| `<list from Step 1>` | Absolute paths to discovered source files | `/abs/path/python/sglang/srt/speculative/eagle_worker.py ...` |
| `<path or "none">` | Absolute path to design doc file, or the literal string `"none"` | `/abs/path/docs/speculative_decoding.md` or `none` |
| `<project>` | Absolute path to project root | `/abs/path/sglang` |
| Phase 3 only: | | |
| `<path to generated .py file>` | Absolute path to the Phase 3 output script | `/abs/path/test-out/phase3/test_npu_eagle3.py` |
| `<path for API verification>` | Source directory to grep for API name verification | `/abs/path/python/sglang/srt/speculative/` |
| `<path to test-cases.json>` | Absolute path to Phase 2 output (for SC config_mismatch + step_fidelity cross-reference) | `/abs/path/test-out/phase2/test-cases.json` |
| `<N>` | Integer count of Phase 2 test cases (for coverage comparison) | `25` |

If a subagent receives unresolved `<skill-base>`, it will look for scripts and references in the wrong directory and fall back to manual VC-equivalent checks, degrading detection quality.

```
Agent(
  description="Phase <N> Sub-A corrector",
  subagent_type="general-purpose",
  run_in_background=true,
  name="phase<N>-corrector-A",
  prompt="<first-launch prompt with ALL placeholders resolved to absolute paths>"
)

Agent(
  description="Phase <N> Sub-B corrector",
  subagent_type="general-purpose",
  run_in_background=true,
  name="phase<N>-corrector-B",
  prompt="<first-launch prompt with ALL placeholders resolved to absolute paths>"
)
```

Both use the same first-launch prompt. Each loads its own independent SETUP context (source files, docs, references). Sub-B is NOT launched lazily at Iter 2 — it needs SETUP context ready before its first use.

After each response, a subagent **stops** (status: completed). Subsequent interactions use `SendMessage` to the correct agent, which **resumes it from its saved transcript**.

**Important**: SendMessage is asynchronous. The agent's response arrives as a `<task-notification>`. Always WAIT for the notification before proceeding.

## Iteration Flow (same for both Sub-A and Sub-B)

Each iteration does a **clean independent detection**. The subagent receives the current file and runs VC+SC from scratch. It does NOT see `_correction_log`, does NOT know what previous iterations found or fixed. This prevents confirmation bias — each detection is a fresh pair of eyes.

```
1. BACKUP: cp <output-file> <output-file>.backup

2. SendMessage to the correct subagent — DETECT ONLY (clean slate).
   Tool signature: `SendMessage(to, summary, message)` — EXACTLY three string parameters. Do NOT add `type`, `recipient`, `content`. Extra params cause InputValidationError.
   - `to`: the subagent's agentId (e.g., `"aa6125eb5b51e36a9"`)
   - `summary`: short description like `"Detect command for Iter 2"`
   - `message`: the prompt text below (plain string, not a JSON object)
   - First contact: Full first-launch prompt from phase<N>-prompts.md.
     BEFORE SENDING: Main MUST substitute all <placeholders> with actual values.
   - Resume (return visit): Short prompt — "Iter <t>: Re-read <output-file>. Run VC+SC+Confidence checks from scratch on the entire file. DETECT ONLY. Report s_t, error list, and confidence summary. DO NOT fix yet."
     Do NOT mention what previous iterations found or fixed.

2a. WAIT for <task-notification> from the subagent with s_t and error list.

3. Main judges with C1-C5 (see references/convergence.md).
   RECORD (iteration=t, s_t) in the Version Buffer.

4. If continuing (C4/Otherwise):
   SendMessage to the SAME subagent. Same tool call pattern: `SendMessage(to=<agentId>, summary="Fix command for Iter <t>", message="...")`.
4a. WAIT for notification.
   After the subagent fixes: write _correction_log, increment iteration,
   save snapshot <file>.iter<t>, record (t, s_t) in Buffer.
   If C4 rollback: restore from .backup, no fixes, next iter with different strategy.
   If C1/C2/C3/C5: go to completion.
```

### Which subagent at which iteration

| Iteration | Subagent | SendMessage to |
|-----------|----------|---------------|
| 1 | Sub-A | agentId-A |
| 2 | Sub-B | agentId-B |
| 3 | Sub-A | agentId-A (resume) |
| 4 | Sub-B | agentId-B (resume) |
| 5 | Sub-A | agentId-A (resume) |

## SendMessage Templates

### First contact

Use the full first-launch prompt from `references/phase<N>-prompts.md`. The prompt already includes self-contained SETUP and DETECT instructions — do NOT add extra tasks or context about previous iterations.

### Resume — detect phase (clean slate)

```
Iter <t>: Re-read <output-file>. Run VC+SC+Confidence checks from scratch on the entire file. DETECT ONLY. Report s_t, error list, and confidence summary. DO NOT fix yet. Do NOT read _correction_log.
```

### Resume — fix phase (only if Main says CONTINUE)

```
Iter <t>: You are in CORRECTOR-ONLY mode. Apply corrections for the errors you detected in YOUR PREVIOUS TURN. Write _correction_log with "iteration": <t>. Increment iteration on affected items.

FORBIDDEN IN THIS TURN — do NOT do any of the following:
- Do NOT re-read the output file (work from your memory of the errors you detected)
- Do NOT run VC scripts or SC checks
- Do NOT compute s_t or judge convergence
- Do NOT report error counts or confidence scores
- Do NOT generate new items UNLESS Main's message explicitly authorizes it
  (e.g., "Add N new test points/cases with IDs X, Y, Z" for missing_scenario fixes)

Convergence judgment, detection, and s_t computation are the NEXT iteration's job (done by a DIFFERENT subagent). Your ONLY job this turn is to edit the file to fix the errors you previously detected.
```

**Separation of concerns**: A subagent receiving this message MUST NOT re-detect, re-evaluate, or judge convergence. Detection happens in a SEPARATE turn (the detect phase), by a SEPARATE subagent in the next iteration. If a subagent oversteps and re-detects during the fix phase, it corrupts the convergence trajectory — Main's C1-C5 judgment becomes invalid because it was applied to errors the correction itself discovered and fixed. The cycle depends on one subagent fixing, then a DIFFERENT subagent detecting with fresh eyes.

**Important**: The detect message must NOT mention what previous iterations found, fixed, or missed. Each detection is independent.

## Recovery

If a subagent is no longer available (timeout, crash, context eviction): launch a replacement with `subagent_type: "general-purpose"`, `run_in_background: true`, and the full first-launch prompt. Use the new agentId for all subsequent SendMessage calls to that role.

Note: a "completed" (stopped) agent IS still available — SendMessage resumes it from transcript.

## Context Window

Each subagent only sees its own iterations' transcript. Sub-A sees Iter 1, 3, 5; Sub-B sees Iter 2, 4. This is roughly half the context of the old single-subagent design. Main's context stays minimal — only scheduling metadata, s_t trajectory, and the Version Buffer.

For MAX_ITER=4 (Phase 1): Sub-A runs 2 times (Iter 1 + Iter 3), Sub-B runs 1 time (Iter 2). For MAX_ITER=5 (Phase 2/3): Sub-A runs 3 times (Iter 1, 3, 5), Sub-B runs 2 times (Iter 2, 4). Sub-B's SETUP cost is amortized across 2 detection rounds.

## Convergence

After each detection, Main judges with C1-C5 (see `references/convergence.md`):

```
s_t == 0                              → C1 CONVERGED → stop
t >= MAX_ITER                         → C5 EXHAUSTED → stop, argmin s(y)
t >= 3 AND τ_t == τ_{t-2} != τ_{t-1}  → C3 OSCILLATION → stop, argmin s(y)
s_t > s_{t-1} + 3                     → C4 OVERSHOOT → rollback, continue
s_t <= s_{t-1} AND s_t >= s_{t-1} - 1 → C2 PLATEAUED → stop
Otherwise                              → continue
```

s_t = 0.8×N_crit + 0.5×N_mod + 0.2×N_min (pre-fix, before corrections applied).

**SC error counting rule**: Count by **affected objects**, not by error type. One `oracle_gap` affecting 5 TCs → N_crit += 5, not += 1. This ensures s_t is comparable across subagents that may group errors differently. VC errors are already counted per-object by the scripts.

**C3 note**: τ_t and τ_{t-2} are now produced by the SAME subagent (both Sub-A or both Sub-B), making oscillation detection more reliable since the same "sensor" is being compared.

## Version Buffer

C3 (oscillation) and C5 (exhausted) need `argmin_{y in Buffer} s(y)`. Maintain a mapping across all iterations:

```
Buffer = [(y_1, s_1=10.8), (y_2, s_2=2.9), (y_3, s_3=3.1), ...]
```

**How to maintain:**
- After EACH detection, record `(iteration=t, s_t)` in conversation memory.
- After EACH correction (when continuing), save `<file>.iter<t>` snapshot.
- When C3 or C5 fires: review Buffer, find lowest s_t, restore `<file>.iter<best_t>`.

**Note**: y_0 (initial Plant G output) has no s_0 — first detection at Iter 1. Buffer starts at t=1.

**Fallback**: If Buffer wasn't maintained, use `.backup` file as best available version.
