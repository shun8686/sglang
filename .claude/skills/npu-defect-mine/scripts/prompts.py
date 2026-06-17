"""
Agent classification prompt templates for NPU defect mining.

Two prompt families:
- AGENT_PR_PROMPT: Deep analysis from PR data (confidence 0.8-0.95)
- AGENT_COMMIT_PROMPT: Shallow analysis from commit data (confidence 0.5-0.7)
"""

# ============================================================
# Agent-PR: Deep Classification Template
# ============================================================

AGENT_PR_PROMPT = """You are an NPU (Huawei Ascend) defect classification expert.
Analyze this GitHub Pull Request data and classify the bug fix.

## Input Data
- PR Title: {pr_title}
- PR Description: {pr_body}
- PR Labels: {pr_labels}
- Review Comments: {review_comments}
- Linked Issues: {linked_issues}
- Files Changed: {files_changed}

## Classification Rules

### Category
- `precision_loss`: Numerical precision drift (FP32→BF16 roundtrip, quant scale shift)
- `crash`: Program crash (segfault, HCCL timeout, CUDA illegal memory)
- `perf_regression`: Performance degradation >10%
- `compatibility`: CANN/torch_npu version incompatibility, platform-specific issues
- `compile_error`: Build/compilation issue

### Severity
- `10` (Critical): Crash / OOM / Hang — program terminates, user immediately aware
- `3` (Major): Basic function error / Precision loss — wrong output or detectable degradation
- `1` (Minor): Performance regression / Compilation / Boundary issue
### Root Cause
Write a specific, one-sentence root cause. Be concrete:
Bad: "memory_layout", "cann_api", "input_processing"
Good: "int32 overflow in concat_mla kernel address computation for sequences > 30000 tokens"
If limited evidence, append "(limited evidence)" to the root cause text.

## Output Format
Return a JSON object matching this schema:
{
  "defects": [
    {
      "bug_id": "BUG-YYYY-NNN",
      "category": "...",
      "severity": "10 (Critical) | 3 (Major) | 1 (Minor)",
      "root_cause": "...",
      "confidence": 0.0-1.0,
      "rationale": "One sentence explaining the classification"
    }
  ]
}
"""

# ============================================================
# Agent-Commit: Shallow Classification Template
# ============================================================

AGENT_COMMIT_PROMPT = """You are an NPU defect classification expert working with LIMITED data.
Analyze this git commit (no PR available) and provide your best classification.

## Input Data
- Commit SHA: {sha}
- Commit Message: {message}
- Files Changed: {files_changed}
- Diff Stats: {diff_stats}

## Task
Classify this potential bug fix. Since you only have commit-level data,
your confidence will be lower than with PR data. Be conservative.

## Classification Rules
(Same categories, severity, root_cause as the PR prompt, but note that
you have less information to work with.)

## Important
- Set `severity` to 10 (Critical), 3 (Major), or 1 (Minor)
- Set `confidence` to 0.5-0.7 maximum (commit data is inherently limited)
- Set `needs_review: true` for ALL entries (human review required)
- If you cannot determine a field confidently, use "unknown"

## Output Format
{
  "defects": [
    {
      "bug_id": "BUG-YYYY-NNN",
      "category": "...",
      "severity": "10 (Critical) | 3 (Major) | 1 (Minor)",
      "root_cause": "unknown",
      "confidence": 0.5,
      "needs_review": true,
      "rationale": "Limited to commit message analysis"
    }
  ]
}
"""

# ============================================================
# Near-Miss Extraction Prompt
# ============================================================

NEAR_MISS_PROMPT = """You are an NPU risk analyst reviewing PR discussions.
Identify "near-miss" signals — potential bugs that were caught during code review.

## Input Data
- PR Number: {pr_number}
- PR Title: {pr_title}
- Review Comments: {review_comments}

## What to Look For
A "near-miss" is a review comment thread where:
1. A reviewer points out a potential problem (not just style/naming)
2. The author acknowledges it and makes a change
3. The problem, if NOT caught, would have been a real bug

## Signal Keywords
- "could cause" / "might lead to" / "race condition"
- "did you consider" / "be careful with" / "this is dangerous"
- "CANN version" / "NPU behavior" / "stream synchronization"
- "good catch" / "nice catch" / "I missed that"
- "under heavy load" / "edge case" / "corner case"

## Output Format
{
  "near_misses": [
    {
      "type": "near_miss",
      "source_pr": {pr_number},
      "source_comment_url": "URL to the specific comment thread",
      "risk_pattern": "Short description of the risk pattern",
      "module": "Affected source module",
      "symptom_if_not_caught": "What would have happened if not caught",
      "would_affect_features": ["feature_names"],
      "trigger_condition": "Conditions that would trigger this bug",
      "caught_by": "code_review",
      "reviewer_comment_sentiment": "concerned | warning | critical",
      "weight": 0.3,
      "recommendation": "Suggested test or safeguard"
    }
  ]
}
"""
