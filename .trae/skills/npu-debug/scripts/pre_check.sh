#!/bin/bash
# ============================================================
# NPU Pre-check Script
# ============================================================
# This script performs environment checks before executing NPU
# test cases inside a Docker container.
#
# Usage:
#   ./pre_check.sh <container_name> [model_path]
#
# Checks performed:
#   1. Container running status
#   2. NPU card availability (npu-smi info)
#   3. Identify completely idle cards (no running processes)
#   4. Map idle NPU IDs to chip IDs (for ASCEND_RT_VISIBLE_DEVICES)
#   5. Verify model path exists (optional)
#
# Exit codes:
#   0 - All checks passed, usable chips printed to stdout
#   1 - Fatal error (container not found, npu-smi failed)
#   2 - No completely idle NPU cards available
#   3 - Model path not found
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <container_name> [model_path]" >&2
    exit 1
fi

CONTAINER="$1"
MODEL_PATH="${2:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

ALL_PASSED=true

# -----------------------------------------------------------
# Step 1: Check container status
# -----------------------------------------------------------
info "Step 1/5: Checking container '$CONTAINER' status..."

if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
    container_status=$(docker ps --filter "name=${CONTAINER}" --format "{{.Status}}")
    pass "Container '$CONTAINER' is running (${container_status})"
else
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
        container_status=$(docker ps -a --filter "name=${CONTAINER}" --format "{{.Status}}")
        fail "Container '$CONTAINER' exists but is NOT running (${container_status})"
    else
        fail "Container '$CONTAINER' does not exist"
    fi
    ALL_PASSED=false
    exit 1
fi

# -----------------------------------------------------------
# Step 2: Check NPU card status inside container
# -----------------------------------------------------------
info "Step 2/5: Running npu-smi info inside container..."

NPU_OUTPUT=$(docker exec "$CONTAINER" bash -c "npu-smi info" 2>&1) || {
    fail "Failed to run npu-smi info inside container"
    echo "$NPU_OUTPUT"
    exit 1
}
pass "npu-smi info executed successfully"

# -----------------------------------------------------------
# Step 3: Identify completely idle cards
# -----------------------------------------------------------
info "Step 3/5: Identifying completely idle NPU cards..."

# Parse lines showing "No running processes found in NPU X"
IDLE_NPUS=()
while IFS= read -r line; do
    if [[ $line =~ No\ running\ processes\ found\ in\ NPU\ ([0-9]+) ]]; then
        IDLE_NPUS+=("${BASH_REMATCH[1]}")
    fi
done < <(echo "$NPU_OUTPUT")

if [ ${#IDLE_NPUS[@]} -eq 0 ]; then
    fail "No completely idle NPU cards found!"
    echo ""
    echo "Current NPU status (processes):"
    echo "$NPU_OUTPUT" | grep -E "^\|" | grep -v "NPU.*Chip.*Process" | grep -v "^+" | grep -v "^=" | head -40
    info "TIP: Wait for running processes to finish or ask user to free resources"
    exit 2
fi

# NPU ID to chip IDs mapping (910C: 1 NPU = 2 chips)
# NPU 0 -> chips 0,1; NPU 1 -> chips 2,3; ... NPU 7 -> chips 14,15
declare -A NPU_TO_CHIPS
for npu_id in "${IDLE_NPUS[@]}"; do
    chip0=$((npu_id * 2))
    chip1=$((npu_id * 2 + 1))
    NPU_TO_CHIPS[$npu_id]="${chip0},${chip1}"
done

echo ""
echo "  Completely idle NPU cards:"
for npu_id in "${IDLE_NPUS[@]}"; do
    echo "    - NPU ${npu_id} -> chips ${NPU_TO_CHIPS[$npu_id]} (ASCEND_RT_VISIBLE_DEVICES=${NPU_TO_CHIPS[$npu_id]})"
done
pass "Found ${#IDLE_NPUS[@]} idle NPU card(s): $(IFS=,; echo "${IDLE_NPUS[*]}")"

# Print full NPU summary
echo ""
echo "  Full NPU memory summary:"
echo "$NPU_OUTPUT" | grep -E "^\|" | grep -E "NPU.*Ascend|Chip.*Phy-ID|NPU.*Chip.*Process" -A 1 | head -30
echo ""

# -----------------------------------------------------------
# Step 4 (optional): Verify model path
# -----------------------------------------------------------
if [ -n "$MODEL_PATH" ]; then
    info "Step 4/5: Verifying model path '$MODEL_PATH'..."
    if docker exec "$CONTAINER" bash -c "test -d '$MODEL_PATH' && ls '$MODEL_PATH'/config.json >/dev/null 2>&1"; then
        pass "Model path '$MODEL_PATH' exists and contains config.json"
    else
        # Try to list available models
        available_models=$(docker exec "$CONTAINER" bash -c "ls '$MODEL_PATH' 2>/dev/null; ls \$(dirname '$MODEL_PATH') 2>/dev/null" | head -20)
        fail "Model path '$MODEL_PATH' not found or missing config.json"
        if [ -n "$available_models" ]; then
            echo "  Available models in parent directory:"
            echo "$available_models" | head -10 | sed 's/^/    /'
        fi
        ALL_PASSED=false
        exit 3
    fi
else
    info "Step 4/5: Skipping model path check (no model path provided)"
fi

# -----------------------------------------------------------
# Step 5: Print recommended chip assignment
# -----------------------------------------------------------
info "Step 5/5: Generating chip assignment recommendations..."

echo ""
echo "  ================================================"
echo "   RECOMMENDED CHIP ASSIGNMENTS"
echo "  ================================================"

# Suggest using the first idle NPU for single-server tests
first_npu="${IDLE_NPUS[0]}"
first_chips="${NPU_TO_CHIPS[$first_npu]}"
echo ""
echo "  Single-server (no TP):"
echo "    ASCEND_RT_VISIBLE_DEVICES=\"${first_chips}\""
echo "    -> Uses NPU ${first_npu} (chips ${first_chips})"
echo ""

if [ ${#IDLE_NPUS[@]} -ge 2 ]; then
    second_npu="${IDLE_NPUS[1]}"
    second_chips="${NPU_TO_CHIPS[$second_npu]}"
    echo "  Two-server (e.g., baseline + speculative):"
    echo "    ASCEND_RT_VISIBLE_DEVICES=\"${first_chips},${second_chips}\""
    echo "    -> Server 1: --base-gpu-id 0 (chip ${first_chips%%,*})"
    echo "    -> Server 2: --base-gpu-id 1 (chip ${second_chips%%,*})"
    echo ""
fi

echo "  TP=2 (single server, 2 chips):"
echo "    ASCEND_RT_VISIBLE_DEVICES=\"${first_chips}\""
echo "    -> Uses NPU ${first_npu} both chips (chips ${first_chips})"
echo ""

# Print suggested environment export
echo "  Export command:"
echo "    export ASCEND_RT_VISIBLE_DEVICES=${first_chips}"
echo ""

# -----------------------------------------------------------
# Summary
# -----------------------------------------------------------
echo "  ================================================"
if [ "$ALL_PASSED" = true ]; then
    echo "   ALL CHECKS PASSED"
else
    echo "   SOME CHECKS FAILED (review above)"
fi
echo "  ================================================"

# Machine-readable output (last line) for programmatic consumption
echo ""
echo "# MACHINE_READABLE: IDLE_NPUS=${IDLE_NPUS[*]}; CHIPS=${first_chips}"