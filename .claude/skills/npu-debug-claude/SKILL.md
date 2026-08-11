---
name: "npu-debug"
description: "Debug and execute sglang test cases on Ascend NPU inside Docker containers. Invoke when user needs to run tests on Ascend NPU, check card status, or debug container environments."
---

# NPU Debugging Expert (Inside Container)

This skill runs sglang tests **directly inside the NPU Docker container** on Ascend 910C NPU (1-card-dual-chip: NPU N = chips 2N, 2N+1). All commands below are executed within the container — no `docker exec` wrapping needed.

> **Path convention**: `[PROJECT_ROOT]` is the project root inside the container (same mount as the host). Always substitute it literally when constructing commands (not as a shell variable).

## Core Rules

1. **PYTHONPATH**: Always APPEND with `$PYTHONPATH` (never overwrite). Inside the container, just use `$PYTHONPATH` directly (no escaping needed).
2. **Idle cards only**: Only use NPUs showing "No running processes found in NPU X" in `npu-smi info` — both chips must have zero processes.
3. **Never change test intent** — only fix infrastructure: paths, field names, mocks.
4. **Log everything** to `test_design/log/test_<name>_$(date +%Y%m%d_%H%M%S).log`
5. **Use `python -m unittest`** with dot-separated module path.

## Workflow

### 1. Pre-check
```bash
bash [PROJECT_ROOT]/.claude/skills/npu-debug/scripts/pre_check.sh [model_path]
```
Extracts idle chip IDs directly from the container's NPU. Exit code != 0 → abort and report.

### 2. Analyze Test
Read the test file. Determine:
- Integration test (starts server) → needs NPU + model path
- Unit test (mocks only) → no NPU needed
- Model path: grep for `_WEIGHTS_PATH` or `MODEL_WEIGHTS_DIR` in test file and imports

### 3. Re-check & Execute
Re-run pre_check. Then run directly:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<chip_ids> && \
export PYTHONPATH=[PROJECT_ROOT]/python:<test_parent_dir>:$PYTHONPATH && \
cd [PROJECT_ROOT] && \
python -m unittest <module.path>.<TestClass>.<test_method> 2>&1 | tee test_design/log/test_<name>_$(date +%Y%m%d_%H%M%S).log
```

Module path uses dots, not slashes: `test_design.03_testcase.rl.test_npu_rl_sleep_tool_call.TestNpuToolCallWithSleep.test_tool_call_pause_during_generation`

## Log Analysis

```bash
# Results summary
grep -E '(test_|OK|FAIL|ERROR|^Ran)' [PROJECT_ROOT]/test_design/log/<logfile> | tail -20
# Errors
grep -E '(AttributeError|KeyError|TypeError|RuntimeError|assert)' [PROJECT_ROOT]/test_design/log/<logfile>
```

## ASCEND_RT_VISIBLE_DEVICES

`ASCEND_RT_VISIBLE_DEVICES` uses **chip IDs** (not NPU IDs). `--base-gpu-id` is an index into this list, not absolute chip ID.

| NPU | Chips  | Example                  |
|-----|--------|--------------------------|
| 0   | 0,1    | `ASCEND_RT_VISIBLE_DEVICES=0,1` |
| 1   | 2,3    | `ASCEND_RT_VISIBLE_DEVICES=2,3` |
| ... | ...    | ...                      |
| 7   | 14,15  | `ASCEND_RT_VISIBLE_DEVICES=14,15` |

**Multi-server** (baseline + EAGLE3): make chips for both visible, use `--base-gpu-id 0` for first server, `--base-gpu-id 1` for second.

## Stop Server

```bash
pkill -9 -f sglang 2>/dev/null; sleep 2; ps aux | grep sglang | grep -v grep | wc -l
```


## Modification Guidelines

**Allowed** (infrastructure only): fix API response paths, add missing mock attributes, set ServerArgs fields, fix model paths, fix field name mismatches.

**NOT allowed** (changes test intent): relax assertions, change expected values, skip tests.
