---
name: "npu-debug"
description: "Debug and execute sglang test cases in remote NPU Docker containers. Invoke when user needs to run tests on Ascend NPU, check card status, or debug container environments."
---

# NPU Remote Debugging Expert

Run sglang tests in Docker containers on Ascend 910C NPU (1-card-dual-chip: NPU N = chips 2N, 2N+1).

> **Path convention**: `[PROJECT_ROOT]` is your workspace `<cwd>`. The container shares the same filesystem, so this path works inside the container too. Always substitute it literally when constructing commands (not as a shell variable).

## Core Rules

1. **PYTHONPATH**: Always APPEND with `$PYTHONPATH` (never overwrite). Inside `docker exec ... bash -c "..."`, escape as `\$PYTHONPATH`.
2. **Idle cards only**: Only use NPUs showing "No running processes found in NPU X" in `npu-smi info` — both chips must have zero processes.
3. **Never change test intent** — only fix infrastructure: paths, field names, mocks.
4. **Log everything** to `test_design/log/test_<name>_$(date +%Y%m%d_%H%M%S).log`
5. **Use `python -m unittest`** with dot-separated module path.

## Workflow

### 0. Container Selection
If user didn't specify a container: list with `docker ps --format "{{.Names}} {{.Status}}" | grep -iE 'sgl|npu|ascend'`, then AskUserQuestion.

### 1. Pre-check
```bash
bash [PROJECT_ROOT]/.trae/skills/npu-debug/scripts/pre_check.sh <container> [model_path]
```
Extracts idle chip IDs. Exit code != 0 → abort and report.

### 2. Analyze Test
Read the test file. Determine:
- Integration test (starts server) → needs NPU + model path
- Unit test (mocks only) → no NPU needed
- Model path: grep for `_WEIGHTS_PATH` or `MODEL_WEIGHTS_DIR` in test file and imports

### 3. Re-check & Execute
Re-run pre_check. Then:

```bash
docker exec <container> bash -c "
  export ASCEND_RT_VISIBLE_DEVICES=<chip_ids> && \
  export PYTHONPATH=[PROJECT_ROOT]/python:<test_parent_dir>:\$PYTHONPATH && \
  cd [PROJECT_ROOT] && \
  python -m unittest <module.path>.<TestClass>.<test_method> 2>&1 | tee test_design/log/test_<name>_\$(date +%Y%m%d_%H%M%S).log
"
```

Module path uses dots, not slashes: `test_design.03_testcase.rl.test_npu_rl_sleep_tool_call.TestNpuToolCallWithSleep.test_tool_call_pause_during_generation`

## Log Analysis

```bash
# Results summary
docker exec <container> bash -c "grep -E '(test_|OK|FAIL|ERROR|^Ran)' [PROJECT_ROOT]/test_design/log/<logfile> | tail -20"
# Errors
docker exec <container> bash -c "grep -E '(AttributeError|KeyError|TypeError|RuntimeError|assert)' [PROJECT_ROOT]/test_design/log/<logfile>"
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

## Modification Guidelines

**Allowed** (infrastructure only): fix API response paths, add missing mock attributes, set ServerArgs fields, fix model paths, fix field name mismatches.

**NOT allowed** (changes test intent): relax assertions, change expected values, skip tests.
