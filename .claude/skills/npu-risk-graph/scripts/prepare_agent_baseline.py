#!/usr/bin/env python3
"""
Prepare Agent Baseline Batch — chunks NPU test files into Agent-sized batches
for Workflow-driven test-to-feature mapping.

Produces: baselines/latest/agent_baseline_batch.json

Usage:
    python .claude/skills/npu-risk-graph/scripts/prepare_agent_baseline.py [--batch-size 7]
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add this script's directory for shared imports (common.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BASELINE_DIR,
    DB_DIR,
    REPO_ROOT,
    ROOT,
    get_current_commit,
    load_json,
    now_iso,
    save_json,
)

# ============================================================
# Test file discovery
# ============================================================


def discover_test_files() -> list[Path]:
    """Discover all NPU test files under test/registered/ascend/."""
    test_root = REPO_ROOT / "test" / "registered" / "ascend"
    if not test_root.exists():
        print(f"WARNING: Test directory not found: {test_root}")
        return []
    return sorted(test_root.rglob("test_*.py"))


# ============================================================
# Context extraction per test file
# ============================================================


def _parse_registrations(file_path: Path) -> list[dict]:
    """Extract register_npu_ci() call kwargs."""
    registrations = []
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "register_npu_ci":
                    kwargs = {}
                    for kw in node.keywords:
                        if isinstance(kw.value, ast.Constant):
                            kwargs[kw.arg] = kw.value.value
                        elif isinstance(kw.value, ast.Name) and kw.value.id in (
                            "True",
                            "False",
                        ):
                            kwargs[kw.arg] = kw.value.id == "True"
                    registrations.append(kwargs)
    except SyntaxError:
        pass
    return registrations


def _parse_classes(file_path: Path) -> list[dict]:
    """Extract test class names, methods, and base classes."""
    classes = []
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    item.name
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and item.name.startswith("test_")
                ]
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                if methods or bases:
                    # Extract docstring [tags]
                    docstring = ast.get_docstring(node) or ""
                    tags = {}
                    for line in docstring.split("\n"):
                        m = re.match(r"\[(\w+\s*\w*)\]\s*(.*)", line.strip())
                        if m:
                            tags[m.group(1)] = m.group(2).strip()
                    classes.append(
                        {
                            "name": node.name,
                            "methods": methods,
                            "bases": bases,
                            "docstring_tags": tags,
                        }
                    )
    except SyntaxError:
        pass
    return classes


def _extract_model(content: str) -> str:
    """Extract cls.model assignment."""
    m = re.search(r'cls\.model\s*=\s*["\'](.+?)["\']', content)
    if m:
        return m.group(1)
    # Also check TEST_MODEL_MATRIX keys
    m = re.search(r"TEST_MODEL_MATRIX\s*=\s*\{", content)
    if m:
        return "(TEST_MODEL_MATRIX)"
    return ""


def _extract_other_args(content: str) -> list[str]:
    """Extract other_args list content."""
    m = re.search(r"other_args\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not m:
        m = re.search(r"common_args\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if m:
        # Extract individual string arguments
        args = re.findall(r'"([^"]+)"', m.group(1))
        return args
    return []


def _extract_env_vars(content: str) -> dict[str, str]:
    """Extract cls.env or os.environ assignments."""
    env = {}
    for m in re.finditer(
        r'(?:cls\.env|os\.environ)\[["\'](\w+)["\']\]\s*=\s*["\']?([^"\'\n]+)["\']?',
        content,
    ):
        env[m.group(1)] = m.group(2)
    return env


def extract_test_context(file_path: Path) -> dict:
    """Extract structured context from a single test file."""
    rel_path = str(file_path.relative_to(REPO_ROOT)).replace("\\", "/")
    registrations = _parse_registrations(file_path)
    classes = _parse_classes(file_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        content = ""

    model = _extract_model(content)
    other_args = _extract_other_args(content)
    env_vars = _extract_env_vars(content)

    # Merge all class info
    all_class_names = [c["name"] for c in classes]
    all_bases = list(set(b for c in classes for b in c["bases"]))
    all_methods = list(set(m for c in classes for m in c["methods"]))
    all_tags = {}
    for c in classes:
        all_tags.update(c.get("docstring_tags", {}))

    return {
        "path": rel_path,
        "class_names": all_class_names,
        "base_classes": all_bases,
        "model": model,
        "other_args": other_args,
        "env_vars": env_vars,
        "registrations": registrations,
        "test_methods": all_methods,
        "docstring_tags": all_tags,
        "content": content,
    }


# ============================================================
# Chunking
# ============================================================


def chunk_test_files(test_contexts: list[dict], batch_size: int = 7) -> list[dict]:
    """Split test files into batches, grouping by directory.

    Files in the same directory test related features, so grouping them
    together helps the Agent maintain context.
    """
    # Group by parent directory
    dir_groups = defaultdict(list)
    for ctx in test_contexts:
        parent = str(Path(ctx["path"]).parent)
        dir_groups[parent].append(ctx)

    # Flatten: keep directory groups together, fill batches
    batches = []
    current_batch = []
    for group in dir_groups.values():
        for ctx in group:
            current_batch.append(ctx)
            if len(current_batch) >= batch_size:
                batches.append({"test_files": current_batch})
                current_batch = []

    # Remaining
    if current_batch:
        batches.append({"test_files": current_batch})

    return batches


# ============================================================
# Main
# ============================================================


def prepare(batch_size: int = 7, output_path: Path = None):
    """Prepare the Agent baseline batch JSON."""
    if output_path is None:
        output_path = BASELINE_DIR / "agent_baseline_batch.json"

    # Load features
    features_data = load_json(BASELINE_DIR / "features.json")
    features = features_data.get("features", [])
    if not features:
        raise RuntimeError("features.json is empty")

    # Discover and extract context from all test files
    print(f"Discovering test files...")
    test_files = discover_test_files()
    print(f"  Found {len(test_files)} test files")

    print(f"Extracting context from each file...")
    contexts = []
    for tf in test_files:
        ctx = extract_test_context(tf)
        contexts.append(ctx)

    # Chunk
    batches = chunk_test_files(contexts, batch_size=batch_size)
    print(f"  Chunked into {len(batches)} batches (~{batch_size} files each)")

    # Build output
    output = {
        "generated_at": now_iso(),
        "git_commit": get_current_commit(),
        "total_test_files": len(test_files),
        "total_batches": len(batches),
        "batch_size": batch_size,
        "features": features,
        "batches": [
            {
                "batch_id": i,
                "count": len(b["test_files"]),
                "test_files": b["test_files"],
            }
            for i, b in enumerate(batches)
        ],
    }

    save_json(output_path, output)
    print(f"  Saved: {output_path}")

    # Print summary
    for b in output["batches"]:
        dirs = set(str(Path(tf["path"]).parent) for tf in b["test_files"])
        print(f"  Batch {b['batch_id']}: {b['count']} files from {len(dirs)} dirs")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Agent baseline batch")
    parser.add_argument(
        "--batch-size", type=int, default=7, help="Files per batch (default: 7)"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    prepare(
        batch_size=args.batch_size,
        output_path=Path(args.output) if args.output else None,
    )
