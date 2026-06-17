---
name: "npu-debug"
description: "Debug and execute sglang test cases in remote NPU Docker containers. Invoke when user needs to run tests on Ascend NPU, check card status, or debug container environments."
---

# NPU Remote Debugging Expert

This skill provides comprehensive debugging and test execution capabilities for sglang test cases running in Docker containers on Ascend 910C NPU environments.

## Core Principles

**CRITICAL: Never change test case intent**
- Only fix infrastructure issues (missing paths, wrong field names, incomplete mocks)
- If a test fails due to assertion logic, keep it failing for manual review
- Do NOT relax assertions or change thresholds to make tests pass
- Document all modifications and their rationale

**CRITICAL: PYTHONPATH MUST append, NEVER overwrite**
- The Ascend container has a pre-configured `PYTHONPATH` that includes essential paths
  (e.g., `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages` for `tbe` module).
  Overwriting it causes GE initialization failure → `SetPrecisionMode` error code 500001.
- **ALWAYS** use `$PYTHONPATH` to append additional paths: `export PYTHONPATH=/path/to/add:$PYTHONPATH`
- When inside `docker exec ... bash -c "..."`, use `\$PYTHONPATH` so the variable is expanded inside the container, not on the host.
- **NEVER** do: `export PYTHONPATH=/path1:/path2` (this overwrites the existing PYTHONPATH)

## Core Workflow

### 0. Container Selection (FIRST TIME ONLY)

**If the user has NOT explicitly specified a container name, PAUSE and ask the user to confirm.**

Before doing anything else, check if the user provided a container name in their request. If not:

1. List all running NPU containers:
   ```bash
   docker ps --format "{{.Names}} {{.Status}} {{.Image}}" | grep -iE 'sgl|vllm|npu|ascend'
   ```

2. **Pause and ask the user which container to use** via `AskUserQuestion`. Provide the list of running containers as options, with the one used most recently in the conversation as default (Recommended).

3. **Do NOT proceed** with any NPU operations until the user confirms a container.

**Examples of "user provided a container name":**
- "在 sgl-dingshun-v0.5.10 上执行..." → container is specified, proceed directly
- "用 sglang-zsy 这个容器跑一下..." → container is specified, proceed directly
- "执行这个测试用例" → container NOT specified, MUST ask user first
- "跑一下 test_xxx" → container NOT specified, MUST ask user first

### 1. Pre-check Script (Recommended)

**Use the pre-check script before every test execution.** The script automates the following checks:

```bash
# Usage: bash .trae/skills/npu-debug/scripts/pre_check.sh <container_name> [model_path]

# Basic check (container status + NPU availability)
bash /home/d00662834/test-debug/sglang/.trae/skills/npu-debug/scripts/pre_check.sh <container_name>

# Full check including model path verification
bash /home/d00662834/test-debug/sglang/.trae/skills/npu-debug/scripts/pre_check.sh <container_name> /path/to/model
```

**What the script checks (in order):**
1. Container is running
2. `npu-smi info` executes successfully inside container
3. Identifies **completely idle** NPU cards (both chips show "No running processes found")
4. Maps idle NPU IDs to chip IDs for `ASCEND_RT_VISIBLE_DEVICES`
5. Verifies model path exists (optional, only if model path provided)
6. Prints recommended chip assignments for different test scenarios

**Exit codes:**
- `0` - All checks passed, usable chips printed
- `1` - Fatal error (container not found, npu-smi failed)
- `2` - No completely idle NPU cards available
- `3` - Model path not found

### 1a. Manual NPU Card Status Check

If you need to inspect card details manually:

```bash
docker exec <container_name> bash -c "npu-smi info"
```

**Key information to extract:**
- NPU ID (0-7 for 910C)
- Chip ID (each NPU has 2 chips, e.g., NPU 5 = chips 10,11)
- HBM memory usage (single chip has 64GB)
- Running processes per chip

**Identify completely idle cards:**
- **Only** cards showing "No running processes found in NPU X" in `npu-smi info` output qualify as completely idle.
- Both chips of the NPU must have **zero** running processes.
- **Never** use a card that has any running processes, even if memory usage appears low (< 1000MB). A card with any running process may become busy at any time.
- HBM usage is **NOT** a reliable indicator of idleness — always check the process table.

### 2. ASCEND_RT_VISIBLE_DEVICES Mapping

**Critical: 910C is 1-card-dual-chip architecture**

| NPU ID | Chip IDs | Example |
|--------|----------|---------|
| 0 | 0, 1 | ASCEND_RT_VISIBLE_DEVICES=0,1 |
| 1 | 2, 3 | ASCEND_RT_VISIBLE_DEVICES=2,3 |
| 2 | 4, 5 | ASCEND_RT_VISIBLE_DEVICES=4,5 |
| 3 | 6, 7 | ASCEND_RT_VISIBLE_DEVICES=6,7 |
| 4 | 8, 9 | ASCEND_RT_VISIBLE_DEVICES=8,9 |
| 5 | 10, 11 | ASCEND_RT_VISIBLE_DEVICES=10,11 |
| 6 | 12, 13 | ASCEND_RT_VISIBLE_DEVICES=12,13 |
| 7 | 14, 15 | ASCEND_RT_VISIBLE_DEVICES=14,15 |

**IMPORTANT**: `ASCEND_RT_VISIBLE_DEVICES` specifies **chip IDs**, not NPU IDs!

### 2.1. Using --base-gpu-id for Multiple Servers

When a test needs to launch **multiple sglang servers simultaneously** (e.g., baseline + speculative decoding comparison), each server must run on a different chip. Use the `--base-gpu-id` server argument to specify which chip each server should use.

**Key points:**
- `--base-gpu-id` is an **index** into the `ASCEND_RT_VISIBLE_DEVICES` list, NOT an absolute chip ID
- `--base-gpu-id 0` → first chip in `ASCEND_RT_VISIBLE_DEVICES`
- `--base-gpu-id 1` → second chip in `ASCEND_RT_VISIBLE_DEVICES`
- For TP=N, the server uses N contiguous chips starting from that index
- Two servers **cannot** share the same chip

**Example: Launch baseline and EAGLE3 servers on chips 6 and 7**

```bash
# Docker command: make chips 6,7 visible
export ASCEND_RT_VISIBLE_DEVICES=6,7
```

```python
# Baseline server uses chip 6 (index 0 in visible list)
base_process = popen_launch_server(
    model_path,
    base_url,
    timeout=timeout,
    other_args=["--tp-size", "1", "--base-gpu-id", "0"],
    env={**os.environ, "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1"},
)

# EAGLE3 server uses chip 7 (index 1 in visible list)
spec_process = popen_launch_server(
    model_path,
    spec_url,
    timeout=timeout,
    other_args=["--tp-size", "1", "--speculative-algorithm", "EAGLE3", ..., "--base-gpu-id", "1"],
    env={**os.environ, "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1"},
)
```

**Chip allocation for common scenarios:**

| ASCEND_RT_VISIBLE_DEVICES | Server 1 | Server 2 | Actual chips used |
|---------------------------|----------|----------|-------------------|
| `0,1` | `--base-gpu-id 0` | `--base-gpu-id 1` | chip 0, chip 1 |
| `6,7` | `--base-gpu-id 0` | `--base-gpu-id 1` | chip 6, chip 7 |
| `6,7,8,9` | `--base-gpu-id 0` (TP=2) | `--base-gpu-id 2` (TP=2) | chips 6,7 and chips 8,9 |

### 3. Execute Test Cases

**Log saving rule: ALL test execution logs MUST be saved to `test_design/log/` with a timestamped filename.**

Use the naming convention: `test_design/log/test_<descriptive_name>_$(date +%Y%m%d_%H%M%S).log`

**PYTHONPATH rule: ALWAYS use `$PYTHONPATH` to append, NEVER overwrite.**
Within `docker exec <container> bash -c "..."`, the `$` in `$PYTHONPATH` must be escaped as `\$` so it expands inside the container, not on the host.

**Standard execution command:**

```bash
docker exec <container_name> bash -c "
  export ASCEND_RT_VISIBLE_DEVICES=<chip_ids> && \
  export PYTHONPATH=/path/to/sglang/python:\$PYTHONPATH && \
  cd /home/d00662834/test-debug/sglang && \
  python <test_file_path> 2>&1 | tee test_design/log/test_<descriptive_name>_\$(date +%Y%m%d_%H%M%S).log
"
```

**Example with idle NPU 5 and 6:**

```bash
docker exec sgl-dingshun-v0.5.10 bash -c "
  export ASCEND_RT_VISIBLE_DEVICES=10,11,12,13 && \
  export PYTHONPATH=/home/d00662834/test-debug/sglang/python:\$PYTHONPATH && \
  cd /home/d00662834/test-debug/sglang && \
  python test_design/03_testcase/chunked_prefill/test_chunked_prefill_functional_npu.py 2>&1 | tee test_design/log/test_chunked_prefill_\$(date +%Y%m%d_%H%M%S).log
"
```

### 4. Environment Verification Checklist

Before executing tests, verify:

- [ ] Container is running: `docker ps | grep <container_name>`
- [ ] NPU cards are accessible: `npu-smi info` succeeds
- [ ] **Completely idle cards** (with "No running processes found") are identified and reserved
- [ ] Both chips of the selected NPU have zero running processes
- [ ] PYTHONPATH appends (not overwrites) sglang paths via `$PYTHONPATH`
- [ ] Test file exists in container
- [ ] Model paths exist (for performance tests): `ls <model_path>`

### 5. Common Issues and Solutions

#### Issue 1: No completely idle NPU cards available

**Action**: 
1. Do NOT execute tests
2. Report current card usage and which cards have running processes
3. Wait for cards to be released or ask user to free resources
4. **Never** use a card that has any running processes, even under low memory usage

#### Issue 2: Subprocess still running warnings

**Solution**: Clean up processes after tests
```bash
docker exec <container_name> bash -c "
  ps aux | grep -E 'sglang|python.*test_' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
"
```

### 6. Test Execution Protocol

**Step-by-step process:**

0. **Select container** (skip if user already specified one)
   - If no container name given, list available containers and ask user to confirm
   - Do not proceed until container is confirmed

1. **Run pre-check script**
   ```bash
   bash /home/d00662834/test-debug/sglang/.trae/skills/npu-debug/scripts/pre_check.sh <container_name> [model_path]
   ```
   - The script checks container status, NPU card availability, and model path
   - If exit code is non-zero, fix the issue before proceeding
   - Extract `ASCEND_RT_VISIBLE_DEVICES` value from the script output

2. **Decision point**:
   - If **completely idle** cards found → proceed with test execution
   - If NO completely idle cards → skip execution, report status, do NOT use partially occupied cards

3. **Analyze test file** (before execution)
   - Check if it's a unit test (uses mocks) or integration test (starts server)
   - Identify required model paths
   - Check for known issues (missing attributes, wrong field names)

4. **Re-check NPU availability IMMEDIATELY before execution**
   **CRITICAL: Always re-check before running the test command.**
   ```bash
   bash /home/d00662834/test-debug/sglang/.trae/skills/npu-debug/scripts/pre_check.sh <container_name>
   ```
   - Cards that were idle minutes ago may now be occupied
   - Confirm the selected NPU still shows idle (exit code 0)
   - If the card is no longer idle, abort and restart from step 1

5. **Execute test** (only if completely idle cards available)

   **CRITICAL: Always use `python -m unittest`

   **PYTHONPATH requirement for unittest:**
   When the test file uses relative imports (e.g., `from utils import ...`), the
   test file's parent directory MUST be added to PYTHONPATH.

   ```bash
   docker exec <container_name> bash -c "
     export ASCEND_RT_VISIBLE_DEVICES=<idle_chip_ids> && \
     export PYTHONPATH=/path/to/sglang/python:<test_parent_dir>:\$PYTHONPATH && \
     cd /home/d00662834/test-debug/sglang && \
     python -m unittest <test_module_path>.<TestClass>.<test_method> 2>&1 | tee test_design/log/test_<descriptive_name>_\$(date +%Y%m%d_%H%M%S).log
   "
   ```

   **Example** (test with relative imports):
   ```bash
   docker exec sgl-dingshun-0616 bash -c "
     export ASCEND_RT_VISIBLE_DEVICES=12,13 && \
     export PYTHONPATH=/home/d00662834/test-debug/sglang-main/python:/home/d00662834/test-debug/sglang/test_design/03_testcase/multimodal:\$PYTHONPATH && \
     cd /home/d00662834/test-debug/sglang && \
     python -m unittest test_design.03_testcase.multimodal.test_multimodal_p1_3_npu.TestMultimodalP1ParameterInteractions.test_011_temperature_multimodal 2>&1 | tee test_design/log/test_011_temperature_multimodal_\$(date +%Y%m%d_%H%M%S).log
   "
   ```

   **NOTE:** `<test_module_path>` uses dot-separated Python module notation (e.g.,
   `test_design.03_testcase.multimodal.test_multimodal_p1_3_npu`), NOT slash-separated
   file paths.

6. **Monitor execution**
   - Check command status periodically
   - Analyze logs for errors
   - Track test progress

7. **Analyze failures**
   - Distinguish between infrastructure issues and assertion failures
   - Fix infrastructure issues only
   - Keep assertion failures for manual review

8. **Clean up**
   - Stop any remaining sglang server processes
   - Verify NPU cards are released

### 7. Log Analysis

Logs are saved to `test_design/log/` with timestamped filenames. Use `ls -t test_design/log/ | head -1` to find the latest log file.

**Check test results:**
```bash
docker exec <container_name> bash -c "grep -E '(test_|OK|FAIL|ERROR|Ran)' test_design/log/<log_filename> | tail -50"
```

**Check detailed errors:**
```bash
docker exec <container_name> bash -c "tail -n 200 test_design/log/<log_filename>"
```

**Search for specific error patterns:**
```bash
docker exec <container_name> bash -c "grep -E '(AttributeError|KeyError|TypeError|RuntimeError)' test_design/log/<log_filename>"
```

### 8. Test Classification

**Unit Tests** (fast, no server):
- Use Mock objects
- Test internal logic (e.g., PrefillAdder, config validation)
- Execution time: < 1 second
- No NPU usage required
- Example: `test_prefill_adder_chunk_boundary_npu.py`, `test_chunked_prefill_config_npu.py`

**Integration Tests** (slow, starts server):
- Launch real sglang server
- Test end-to-end functionality
- Execution time: 10-300 seconds
- Require NPU and model files
- Example: `test_chunked_prefill_functional_npu.py`, `test_chunked_prefill_perf_npu.py`

**Performance Tests** (very slow, benchmark):
- Compare metrics (TTFT, throughput)
- Require specific model files
- Execution time: 200-400 seconds
- May have hardware-specific assertions
- Example: `test_chunked_prefill_perf_npu.py`

### 9. NPU Memory Considerations

- Single chip HBM: 64GB
- Typical sglang server usage: ~2-4GB for model weights + KV cache
- If HBM usage > 60GB, card is likely occupied
- **Always prefer cards with lowest memory usage** — but only among cards that are **completely idle** (no running processes)
- **Never** select a card that has running processes, even if HBM usage appears low
- For 8B models: ~15GB model weights + ~17GB KV cache = ~32GB total

### 10. Container-Specific Notes

**For sgl-dingshun-v0.5.10:**
- Base image: `quay.io/ascend/sglang:v0.5.10-npu.rc1-a3`
- Python path: `/home/d00662834/test-debug/sglang/python`
- Model cache: `/root/.cache/modelscope/hub/models/`
- Available models:
  - `/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
  - `/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct`
  - `/root/.cache/modelscope/hub/models/Qwen/` (symlink to `/home/weights/Qwen`)
- Default shell: bash

### 11. Quick Reference Commands

```bash
# Run pre-check script (recommended before every test)
bash /home/d00662834/test-debug/sglang/.trae/skills/npu-debug/scripts/pre_check.sh <container> [model_path]

# Check all NPU status
docker exec <container> bash -c "npu-smi info"

# Find idle NPUs
docker exec <container> bash -c "npu-smi info | grep -A 1 'No running processes'"

# Kill all sglang processes
docker exec <container> bash -c "ps aux | grep -E 'sglang|python.*test_' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null"

# Check running Python processes
docker exec <container> bash -c "ps aux | grep python"

# View test logs (use latest log in test_design/log/)
docker exec <container> bash -c "ls -t test_design/log/ | head -1 | xargs -I{} cat test_design/log/{}"

# Check available models
docker exec <container> bash -c "ls -la /root/.cache/modelscope/hub/models/"

# Search for correct API usage patterns
grep -r "output_ids" test_design/ --include="*.py" | head -20

# Check benchmark result field names
grep -A 50 "result = {" python/sglang/bench_serving.py | grep "throughput"

# Quick test execution (PYTHONPATH always appended with $PYTHONPATH)
docker exec <container> bash -c "
  export ASCEND_RT_VISIBLE_DEVICES=<chip_ids> && \
  export PYTHONPATH=/path/to/sglang/python:<test_parent_dir>:\$PYTHONPATH && \
  cd /home/d00662834/test-debug/sglang && \
  python -m unittest test_design.03_testcase.xxx.TestClass.test_method 2>&1 | tee test_design/log/test_xxx_\$(date +%Y%m%d_%H%M%S).log
"
```

## Best Practices

1. **ALWAYS check card status before execution** - never assume cards are available
2. **Use completely idle cards only** - never interfere with running workloads. Only use NPUs showing "No running processes found".
3. **Clean up after tests** - ensure processes are terminated
4. **Log everything** - use `tee` to capture output to `test_design/log/` with timestamped filenames
5. **Verify environment** - check PYTHONPATH (always append with `$PYTHONPATH`, never overwrite), dependencies, file paths
6. **Handle errors gracefully** - analyze logs, fix issues, re-run if needed
7. **Respect resource constraints** - 910C has limited cards, use them wisely
8. **Never change test intent** - only fix infrastructure issues
9. **Document all modifications** - explain what was changed and why
10. **Verify fixes don't break other tests** - run related tests after changes

## Modification Guidelines

**Allowed modifications** (do NOT change test intent):
- Fix API response format access paths
- Add missing mock object attributes
- Set required ServerArgs fields
- Correct model paths
- Fix field name mismatches
- Add missing tool function parameters
- Fix method call side effects

**NOT allowed** (changes test intent):
- Relax assertion thresholds
- Change expected values
- Modify comparison operators
- Skip failing tests
- Change test logic
