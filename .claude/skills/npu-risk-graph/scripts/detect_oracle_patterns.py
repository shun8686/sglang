#!/usr/bin/env python3
"""
Detect oracle patterns in existing test files via AST analysis.

Scans test/registered/ for Python test files and identifies which oracle
patterns each test uses by inspecting imports, class inheritance, method
names, assertion patterns, and key parameters.

Updates test_pattern_assignments in .sglang-risk/db/test_patterns.json.

Usage:
    python detect_oracle_patterns.py              # scan + update
    python detect_oracle_patterns.py --dry-run    # preview only, no write
    python detect_oracle_patterns.py --test <path> # single file
    python detect_oracle_patterns.py --list       # list all detected patterns

Detection signals per oracle type:
    gsm8k_accuracy:
        - Inherits from GSM8KMixin or GSM8KAscendMixin
        - Class has test_gsm8k method (or inherits from mixin)
        - Class attribute: accuracy = <float> or gsm8k_accuracy_thres = <float>

    exact_match_short:
        - assertEqual(text_output, ref_output) in method body
        - max_new_tokens <= 32 (short generation)
        - temperature=0 (greedy)

    logprob_rescore:
        - Method named test_logprob_spec_v2_match or similar
        - Compare decode logprobs vs prefill rescore logprobs
        - assertLess(max_diff, <tolerance>) or assertAlmostEqual(delta=...)
        - max_new_tokens=0 for rescore phase

    token_oracle_canary:
        - Inherits from CanaryE2EBase
        - Server arg: --sampling-backend token_oracle
        - Env: SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=1
"""

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

_script_path = Path(__file__).resolve()
REPO_ROOT = _script_path.parent
while not (REPO_ROOT / ".git").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

TEST_DIR = REPO_ROOT / "test" / "registered"
PATTERNS_PATH = REPO_ROOT / ".sglang-risk" / "db" / "test_patterns.json"


# ── Detection rule: (import_match, class_bases, methods, assertions, params, env_vars) ──


class OracleDetector:
    """Base detector: walks an AST module and scores oracle pattern matches."""

    def __init__(self, tree: ast.Module, file_path: str):
        self.tree = tree
        self.file_path = file_path
        self.source = self._read_source()

    def _read_source(self) -> str:
        try:
            return Path(self.file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    # -- helpers --

    def _has_import(self, *names: str) -> bool:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for n in names:
                        if n in alias.name:
                            return True
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for n in names:
                        if n in node.module:
                            return True
        return False

    def _class_bases(self) -> list[str]:
        bases = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
        return bases

    def _method_names(self) -> list[str]:
        names = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                names.append(node.name)
        return names

    def _class_attributes(self) -> dict[str, object]:
        """Extract simple class-level assignments (strings, numbers, lists)."""
        attrs = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                val = self._eval_const(stmt.value)
                                if val is not None:
                                    attrs[target.id] = val
        return attrs

    def _eval_const(self, node: ast.expr) -> Optional[object]:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            items = [self._eval_const(e) for e in node.elts]
            return items if all(i is not None for i in items) else None
        return None

    def _source_contains(self, pattern: str) -> bool:
        return pattern in self.source

    def _args_contain(self, *fragments: str) -> bool:
        """Check if any argument list in the source contains all fragments."""
        for frag in fragments:
            if frag not in self.source:
                return False
        return True

    # -- detectors --

    def detect_gsm8k(self) -> Optional[dict]:
        bases = self._class_bases()
        has_mixin = any(b in bases for b in ("GSM8KMixin", "GSM8KAscendMixin"))
        if not has_mixin:
            return None

        attrs = self._class_attributes()
        threshold = attrs.get("accuracy") or attrs.get("gsm8k_accuracy_thres")
        spec_args = []
        if self._source_contains("speculative_algorithm"):
            spec_args.append("speculative")

        return {
            "pattern_id": "gsm8k_accuracy",
            "weight": 1.0,
            "role": "primary",
            "confidence": 0.95,
            "signals": {
                "mixin": [b for b in bases if "GSM8K" in b],
                "threshold": threshold,
                "speculative": bool(spec_args),
            },
        }

    def detect_exact_match(self) -> Optional[dict]:
        source = self.source

        # Must have assertEqual comparing generated TEXT outputs — not
        # generic dict access or HTTP status codes.  Look for:
        #   assertEqual(output, ref_output)
        #   assertEqual(a["text"], b["text"])
        #   assertEqual(text_a, text_b)
        output_ref = bool(
            re.search(r"self\.assertEqual\(\s*\w+\s*,\s*self\.\w+_output", source)
            or re.search(r"self\.assertEqual\(\s*\w+\s*,\s*\w+_output", source)
        )
        text_compare = bool(re.search(r'assertEqual\([^)]*\["text"\][^)]*\)', source))
        gen_output_compare = bool(
            re.search(r"assertEqual\([^)]*output[^)]*ref[^)]*\)", source)
        )

        if not (output_ref or text_compare or gen_output_compare):
            return None

        # Check for max_new_tokens value (handles both "max_new_tokens": 8 and max_new_tokens=8)
        token_match = re.search(r'max_new_tokens["\']?\s*[=:]\s*["\']?\s*(\d+)', source)
        max_tokens = int(token_match.group(1)) if token_match else None

        # Check temperature=0
        has_temp_zero = bool(re.search(r'temperature["\']?\s*[=:]\s*0', source))

        # generate() or /generate must be present
        is_generation_test = bool(
            re.search(r"\.generate\(", source) or "/generate" in source
        )

        if not is_generation_test:
            return None

        return {
            "pattern_id": "exact_match_short",
            "weight": 0.9,
            "role": "primary",
            "confidence": (
                0.85 if (has_temp_zero and max_tokens and max_tokens <= 32) else 0.6
            ),
            "signals": {
                "assert_equal_text": True,
                "max_new_tokens": max_tokens,
                "temperature_zero": has_temp_zero,
                "generation": True,
            },
        }

    def detect_logprob_rescore(self) -> Optional[dict]:
        source = self.source
        methods = self._method_names()

        # Must have a logprob comparison method
        has_key_method = any(m.startswith("test_logprob") for m in methods)

        # Two-phase signal: max_new_tokens=0 for rescore phase is STRONG
        has_rescore_phase = bool(
            "max_new_tokens" in source
            and re.search(r'max_new_tokens["\']?\s*[=:]\s*0\b', source)
        )

        # return_logprob must be present (both phases need it)
        has_return_logprob = "return_logprob" in source

        # If missing both the key method name AND the rescore phase, skip.
        # (e.g. test_npu_original_logprobs compares SGLang vs HF logprobs,
        #  not decode vs rescore — that's a different kind of test.)
        if not has_key_method and not has_rescore_phase:
            return None

        # Tolerance signal
        tol_match = re.search(
            r"(?:assertLess|assertAlmostEqual|delta\s*=\s*)(0\.\d+)", source
        )
        tolerance = float(tol_match.group(1)) if tol_match else None

        # Confidence calculation
        if has_key_method and has_rescore_phase and tolerance:
            confidence = 0.95  # strongest: all three signals present
        elif has_key_method and has_rescore_phase:
            confidence = 0.85
        elif has_key_method and has_return_logprob:
            confidence = 0.7  # decent: method + logprob return
        elif has_rescore_phase:
            confidence = 0.75  # rescore phase present but method name generic
        else:
            confidence = 0.55  # borderline

        return {
            "pattern_id": "logprob_rescore",
            "weight": 0.9,
            "role": "primary",
            "confidence": confidence,
            "signals": {
                "key_method": has_key_method,
                "tolerance": tolerance,
                "rescore_phase": has_rescore_phase,
                "return_logprob": has_return_logprob,
            },
        }

    def detect_token_oracle(self) -> Optional[dict]:
        source = self.source
        bases = self._class_bases()

        has_canary_base = any("Canary" in b for b in bases)
        has_token_oracle_arg = "--sampling-backend token_oracle" in source
        has_token_oracle_env = "ENABLE_TOKEN_ORACLE" in source
        has_canary_mode = "CanaryMode" in source

        if not (has_canary_base or has_token_oracle_arg or has_token_oracle_env):
            return None

        # Confidence: strongest when --sampling-backend is directly in source.
        # When inheriting from CanaryE2EBase + ENABLE_TOKEN_ORACLE env, the
        # server arg lives in the base class — still a reliable signal (0.85).
        if has_token_oracle_arg:
            confidence = 0.95
        elif has_canary_base and has_token_oracle_env:
            confidence = 0.85
        elif has_canary_base:
            confidence = 0.80
        else:
            confidence = 0.70

        return {
            "pattern_id": "token_oracle_canary",
            "weight": 1.0,
            "role": "primary",
            "confidence": confidence,
            "signals": {
                "canary_base": has_canary_base,
                "sampling_backend_oracle": has_token_oracle_arg,
                "token_oracle_env": has_token_oracle_env,
                "canary_mode": has_canary_mode,
            },
        }

    def detect_all(self) -> list[dict]:
        results = []
        for detector in [
            self.detect_gsm8k,
            self.detect_exact_match,
            self.detect_logprob_rescore,
            self.detect_token_oracle,
        ]:
            result = detector()
            if result:
                results.append(result)
        return results


def _rel_path(abs_path: str) -> str:
    """Convert absolute path to repo-relative (forward-slash)."""
    try:
        return str(Path(abs_path).relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


def scan_tests(test_dir: Path = None, single_file: str = None) -> list[dict]:
    """Scan test files and return test_pattern_assignments entries."""
    if test_dir is None:
        test_dir = TEST_DIR

    assignments = []

    if single_file:
        py_files = [Path(single_file)]
    else:
        py_files = sorted(test_dir.rglob("test_*.py"))

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  SKIP {py_file.name}: {e}")
            continue

        detector = OracleDetector(tree, str(py_file))
        patterns = detector.detect_all()

        if not patterns:
            continue

        rel_path = _rel_path(str(py_file))

        # Extract test class name
        test_class = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    test_class = node.name
                    break

        entry = {
            "test_id": rel_path,
            "test_class": test_class,
            "patterns": [
                {
                    "pattern_id": p["pattern_id"],
                    "weight": p["weight"],
                    "role": p["role"],
                    "confidence": p["confidence"],
                    "detail": _format_detail(p),
                }
                for p in patterns
            ],
        }
        assignments.append(entry)

    return assignments


def _format_detail(pattern: dict) -> str:
    """Generate a human-readable detail string from detection signals."""
    sig = pattern.get("signals", {})
    pid = pattern["pattern_id"]

    if pid == "gsm8k_accuracy":
        thresh = sig.get("threshold")
        return f"accuracy threshold={thresh}" if thresh else "GSM8K accuracy test"
    elif pid == "exact_match_short":
        mt = sig.get("max_new_tokens")
        tz = sig.get("temperature_zero")
        parts = []
        if mt:
            parts.append(f"max_new_tokens={mt}")
        if tz:
            parts.append("temperature=0")
        return ", ".join(parts) if parts else "exact text match"
    elif pid == "logprob_rescore":
        tol = sig.get("tolerance")
        return f"tolerance={tol}" if tol else "logprob rescore comparison"
    elif pid == "token_oracle_canary":
        parts = []
        if sig.get("sampling_backend_oracle"):
            parts.append("--sampling-backend token_oracle")
        if sig.get("token_oracle_env"):
            parts.append("ENABLE_TOKEN_ORACLE")
        return ", ".join(parts) if parts else "canary oracle"
    return ""


def load_patterns(path: Path = None) -> dict:
    if path is None:
        path = PATTERNS_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_patterns(data: dict, path: Path = None):
    if path is None:
        path = PATTERNS_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated: {path}")


# Confidence threshold: patterns below this are excluded from auto-assignment
# but still reported in --dry-run for manual review.
AUTO_CONFIDENCE_THRESHOLD = 0.75


def merge_assignments(
    existing: list[dict],
    scanned: list[dict],
) -> list[dict]:
    """Merge scanned assignments into existing list, preserving manual entries.

    - Patterns with confidence >= AUTO_CONFIDENCE_THRESHOLD are auto-assigned.
    - Patterns with confidence < threshold go into ``_candidates`` for manual review.
    - Manual entries (with ``_manual: true`` in the pattern dict) are preserved.
    - Tests not found in scan are kept as-is.
    """
    scanned_map = {a["test_id"]: a for a in scanned}
    merged = []

    for entry in existing:
        tid = entry["test_id"]
        if tid in scanned_map:
            scanned_entry = scanned_map.pop(tid)
            # Preserve ONLY manual patterns; auto patterns come from fresh scan
            manual_patterns = [p for p in entry.get("patterns", []) if p.get("_manual")]
            # Split fresh scanned patterns by confidence
            high_conf = [
                p
                for p in scanned_entry["patterns"]
                if p.get("confidence", 0) >= AUTO_CONFIDENCE_THRESHOLD
            ]
            low_conf = [
                p
                for p in scanned_entry["patterns"]
                if p.get("confidence", 0) < AUTO_CONFIDENCE_THRESHOLD
            ]
            # Replace patterns entirely with fresh scan + preserved manual
            entry["patterns"] = high_conf + manual_patterns
            if low_conf:
                entry["_candidates"] = low_conf
            else:
                entry.pop("_candidates", None)
            entry["test_class"] = scanned_entry.get("test_class") or entry.get(
                "test_class", ""
            )
            entry["_auto_detected"] = True
            # If no patterns at all (not even candidates), remove auto-detected flag
            if not entry["patterns"] and not entry.get("_candidates"):
                entry.pop("_auto_detected", None)
        else:
            # Test not in fresh scan — remove auto flag, keep manual patterns only
            entry["patterns"] = [
                p for p in entry.get("patterns", []) if p.get("_manual")
            ]
            entry.pop("_auto_detected", None)
            entry.pop("_candidates", None)
        merged.append(entry)

    for tid, entry in scanned_map.items():
        high_conf = [
            p
            for p in entry["patterns"]
            if p.get("confidence", 0) >= AUTO_CONFIDENCE_THRESHOLD
        ]
        low_conf = [
            p
            for p in entry["patterns"]
            if p.get("confidence", 0) < AUTO_CONFIDENCE_THRESHOLD
        ]
        if high_conf:
            entry["patterns"] = high_conf
            entry["_auto_detected"] = True
            merged.append(entry)
        if low_conf:
            entry.setdefault("_candidates", []).extend(low_conf)
            if not high_conf:
                entry["patterns"] = []
                entry["_auto_detected"] = True
                merged.append(entry)

    # Remove entries with no patterns, no candidates, and no manual flag
    merged = [
        e
        for e in merged
        if e.get("patterns")
        or e.get("_candidates")
        or any(p.get("_manual") for p in e.get("patterns", []))
    ]

    return merged


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect oracle patterns in test files via AST analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, do not update test_patterns.json",
    )
    parser.add_argument(
        "--test", type=str, metavar="PATH", help="Scan a single test file"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current assignments from test_patterns.json",
    )
    args = parser.parse_args()

    if args.list:
        tp = load_patterns()
        assignments = tp.get("test_pattern_assignments", [])
        for a in assignments:
            patterns = ", ".join(
                f"{p['pattern_id']}({p.get('role','?')})" for p in a.get("patterns", [])
            )
            auto = "[auto]" if a.get("_auto_detected") else "[manual]"
            print(f"{auto} {a['test_id']}")
            print(f"       class={a.get('test_class','?')}  patterns={patterns}")
        sys.exit(0)

    # Scan
    print("Scanning test files for oracle patterns...")
    assignments = scan_tests(single_file=args.test)

    # Summary
    pattern_counts = {}
    for a in assignments:
        for p in a["patterns"]:
            pid = p["pattern_id"]
            pattern_counts[pid] = pattern_counts.get(pid, 0) + 1

    print(f"\nDetected {len(assignments)} tests with oracle patterns:")
    for pid, count in sorted(pattern_counts.items()):
        print(f"  {pid:30s}  {count} tests")

    if args.test:
        # Single file mode: just print the detection result
        print(f"\n{args.test}:")
        for a in assignments:
            for p in a["patterns"]:
                print(f"  {p['pattern_id']} (weight={p['weight']}, {p['detail']})")
        sys.exit(0)

    if args.dry_run:
        print("\n[DRY RUN] test_patterns.json NOT updated.")
        print("Re-run without --dry-run to apply changes.")
        sys.exit(0)

    # Merge into test_patterns.json
    tp = load_patterns()
    existing = tp.get("test_pattern_assignments", [])
    merged = merge_assignments(existing, assignments)
    tp["test_pattern_assignments"] = merged

    # Update statistics in _meta
    tp.setdefault("_meta", {})["last_auto_scan"] = True
    tp["_meta"]["scanned_test_count"] = len(assignments)
    tp["_meta"]["total_assignment_count"] = len(merged)

    save_patterns(tp)
    print(
        f"\nMerged: {len(existing)} existing + {len(assignments)} detected "
        f"= {len(merged)} total assignments"
    )
    print("Run '/npu-risk-graph build' to rebuild graph with updated patterns.")
