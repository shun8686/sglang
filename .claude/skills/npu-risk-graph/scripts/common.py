"""
Shared utilities for the NPU risk testing framework.

Usage:
    from common import (
        compute_fingerprint, load_json, save_json,
        get_changed_files, affected_features, RiskLevel, factor_score
    )
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---- Paths ----
# Find the repo root, then locate .sglang-risk/ (data directory).
# common.py may live under .claude/skills/, not .sglang-risk/scripts/.
_script_path = Path(__file__).resolve()
_repo_root = _script_path.parent
while not (_repo_root / ".git").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent

REPO_ROOT = _repo_root
ROOT = REPO_ROOT / ".sglang-risk"
BASELINE_DIR = ROOT / "baselines" / "latest"
DELTA_DIR = ROOT / "deltas"
TREND_DIR = ROOT / "trends"
DB_DIR = ROOT / "db"
GRAPH_DIR = ROOT / "graph"
SCHEMA_DIR = ROOT / "schemas"

# ---- Fingerprint ----


def compute_fingerprint(file_paths: list[str]) -> str:
    """SHA256 of normalized (whitespace-stripped) source content.
    Returns same hash for semantically identical code regardless of formatting.
    """
    h = hashlib.sha256()
    for fp in sorted(file_paths):
        p = REPO_ROOT / fp
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        # Normalize: strip trailing whitespace per line, skip empty lines
        normalized = "\n".join(
            line.rstrip() for line in content.splitlines() if line.strip()
        )
        h.update(normalized.encode())
    return h.hexdigest()


# ---- JSON I/O ----


def load_json(path: Path) -> dict:
    """Load and return parsed JSON. Returns {} if file missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data, indent: int = 2):
    """Atomically write JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


# ---- Git helpers ----


def get_changed_files(base_ref: str = "HEAD~1", head_ref: str = "HEAD") -> list[str]:
    """Return list of files changed between two git refs.

    Returns empty list on git failure, but prints a warning to stderr so
    the caller can distinguish "no changes" from "git command failed"
    (e.g. missing remote, detached HEAD).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, head_ref],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=REPO_ROOT,
            timeout=30,
        )
        if result.returncode != 0:
            print(
                f"  [WARN] git diff failed (exit {result.returncode}): {result.stderr.strip()[:200]}",
                file=sys.stderr,
            )
            return []
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"  [WARN] git diff error: {e}", file=sys.stderr)
        return []


def get_current_commit() -> str:
    """Return current HEAD SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_commits_since(since_sha: str) -> list[dict]:
    """Return commit list between since_sha..HEAD."""
    try:
        result = subprocess.run(
            ["git", "log", f"{since_sha}..HEAD", "--format=%H|%ai|%s"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=REPO_ROOT,
            timeout=30,
        )
        commits = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({"sha": parts[0], "date": parts[1], "message": parts[2]})
        return commits
    except Exception:
        return []


# ---- Dependency Map ----


def load_dependency_map() -> dict:
    """Load dependency_map.json from latest baseline.

    Extracts and returns the inner ``dependency_map`` dict (file→feature
    mappings), not the wrapper JSON that includes generated_at / git_commit.
    """
    data = load_json(BASELINE_DIR / "dependency_map.json")
    return data.get("dependency_map", {})


# ---- Feature name registry (single source of truth: features.json) ----

_features_name_cache: set[str] | None = None


def load_feature_names() -> set[str]:
    """Return the set of canonical feature names from features.json.
    Cached after first call — features.json is treated as immutable within a run.
    """
    global _features_name_cache
    if _features_name_cache is not None:
        return _features_name_cache
    data = load_json(BASELINE_DIR / "features.json")
    _features_name_cache = {f["name"] for f in data.get("features", [])}
    return _features_name_cache


def validate_feature_name(name: str, caller: str = "") -> bool:
    """Check if `name` exists in features.json. Prints warning if not.
    Returns True if valid, False otherwise.
    """
    valid = load_feature_names()
    if name not in valid:
        ctx = f" [{caller}]" if caller else ""
        print(
            f"  [WARN]{ctx} feature name '{name}' not found in features.json — may need update"
        )
        return False
    return True


def resolve_feature_name(target: str, caller: str = "") -> str | None:
    """Resolve a possibly-stale feature name against features.json.
    Tries exact match → case-insensitive → substring. Returns canonical name or None.
    """
    valid = load_feature_names()
    if target in valid:
        return target
    key = target.lower().replace("_", "")
    name_lower = {n.lower().replace("_", ""): n for n in valid}
    if key in name_lower:
        return name_lower[key]
    for canon in sorted(valid):
        if key in canon.lower().replace("_", ""):
            ctx = f" [{caller}]" if caller else ""
            print(
                f"  [WARN]{ctx} '{target}' fuzzy-resolved to '{canon}' — update the reference"
            )
            return canon
    ctx = f" [{caller}]" if caller else ""
    print(f"  [WARN]{ctx} '{target}' not found in features.json")
    return None


def affected_features(changed_files: list[str], dep_map: dict = None) -> set[str]:
    """Given list of changed files, return set of affected feature names.
    Uses 1-hop file→feature mapping from dependency_map.
    """
    if dep_map is None:
        dep_map = load_dependency_map()
    features = set()
    for f in changed_files:
        # Normalize path: strip leading ./ and make relative to repo root
        f_norm = f.lstrip("./")
        # Try exact match first, then suffix match
        for path, info in dep_map.items():
            if f_norm.endswith(path) or path.endswith(f_norm) or f_norm == path:
                for feat in info.get("features", []):
                    features.add(feat)
                break
    return features


# ---- Risk Level ----


class RiskLevel:
    """Risk level thresholds for composite_score (max 45)."""

    CRITICAL = 28  # >= 28
    HIGH = 20  # >= 20
    MEDIUM = 12  # >= 12
    LOW = 0  # < 12

    @classmethod
    def from_score(cls, score: int) -> str:
        if score >= cls.CRITICAL:
            return "critical"
        if score >= cls.HIGH:
            return "high"
        if score >= cls.MEDIUM:
            return "medium"
        return "low"

    @classmethod
    def test_depth(cls, level: str) -> str:
        return {
            "critical": "L0",
            "high": "L1",
            "medium": "L2",
            "low": "L3",
        }.get(level, "L3")


def factor_score(
    score: int,
    max_score: int = 5,
    rationale: str = "",
    source: str = "manual",
    details: dict = None,
) -> dict:
    """Create a standardized factor score entry.

    Clamps to [0, max_score] — historical_defects can legitimately be 0
    (clean feature with no past bugs), unlike other factors which use a
    1-5 scale.
    """
    return {
        "score": max(0, min(max_score, score)),
        "rationale": rationale,
        "source": source,
        "details": details or {},
    }


# ---- Timestamp ----


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Initialization ----


def init_directories():
    """Ensure all required directories exist."""
    for d in [
        BASELINE_DIR,
        DELTA_DIR,
        TREND_DIR,
        DB_DIR,
        GRAPH_DIR,
        GRAPH_DIR / "snapshots",
        ROOT / "defect_mining",
        SCHEMA_DIR,
        ROOT / "dashboard",
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init_directories()
    print(f"Framework root: {ROOT}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Current commit: {get_current_commit()}")
    print("Directories initialized.")
