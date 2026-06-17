#!/usr/bin/env python3
"""
Knowledge Graph Builder for NPU Risk Analysis.

Converts JSON baseline (features.json + tests.json + risk_profiles.json +
dependency_map.json + defect_db.json) into a NetworkX property graph.

Usage:
    python .claude/skills/npu-risk-graph/scripts/build_graph.py [--baseline latest] [--output latest_graph.json]
"""

import sys
from pathlib import Path

import networkx as nx
from common import (
    BASELINE_DIR,
    DB_DIR,
    GRAPH_DIR,
    REPO_ROOT,
    ROOT,
    get_current_commit,
    load_json,
    now_iso,
    save_json,
)

# ============================================================
# Schema Normalization
# ============================================================

# New defect_db.json schema (Agent workflow output) differs from what
# build_graph.py originally expected. This normalization layer infers
# the missing fields so downstream graph construction stays unchanged.
#
# New → Old field mapping:
#   category     → failure_mode   (direct map)
#   category     → is_silent      (inferred: precision_loss + perf_regression)
#   files_fixed  → features_affected (via dependency_map + features.json)
#   pr_title     → title          (with encoding sanitization)

CATEGORY_TO_FAILURE_MODE = {
    "crash": "crash",
    "precision_loss": "silent_precision_loss",
    "compatibility": "compatibility_issue",
    "perf_regression": "performance_regression",
    "compile_error": "build_error",
}

CATEGORY_IS_SILENT = {
    "precision_loss": True,
    "perf_regression": True,
    "crash": False,
    "compile_error": False,
    "compatibility": False,
}


def _normalize_defect_schema(
    defects: list[dict],
    dep_map: dict,
    features: list[dict],
) -> list[dict]:
    """Add inferred fields (features_affected, failure_mode, is_silent, title)
    to each defect so the rest of build_graph.py can consume them transparently.

    The new Agent-workflow schema has ``files_fixed`` and ``category`` but not
    ``features_affected``, ``failure_mode``, or ``is_silent``.  This function
    bridges that gap by:

    * building a file-path → feature-names reverse index from both
      ``dependency_map.json`` and ``features.json`` source_files; and
    * mapping ``category`` values to ``failure_mode`` / ``is_silent``.

    Defects whose ``files_fixed`` cannot be mapped to any known feature are
    kept (their ``features_affected`` list will be empty — they still become
    Defect nodes, just without AFFECTED_BY edges).
    """
    # ---- file → features reverse index ---------------------------------
    file_to_features: dict[str, set[str]] = {}

    # Primary source: dependency_map.json (richer, pre-computed)
    for file_path, entry in dep_map.items():
        file_to_features.setdefault(file_path, set()).update(entry.get("features", []))

    # Supplementary: features.json source_files (catches NPU files that
    # aren't in dependency_map, e.g. newly added or docs/config files)
    for f in features:
        feat_name = f["name"]
        for sf in f.get("source_files", []):
            sf_norm = sf.replace("\\", "/")
            file_to_features.setdefault(sf_norm, set()).add(feat_name)

    # ---- identify hub files (configuration / orchestration files that   -
    # map to too many features — we exclude them from features_affected
    # inference because a bug in a hub file doesn't indicate a shared
    # defect *pattern* between the features it touches).
    total_features = len(features)
    hub_threshold = max(5, int(total_features * 0.25))  # >25% of features = hub
    hub_files: set[str] = {
        fp for fp, feats in file_to_features.items() if len(feats) > hub_threshold
    }

    # ---- normalize each defect -----------------------------------------
    normalized: list[dict] = []
    for d in defects:
        nd = dict(d)  # shallow copy — we only add keys, don't mutate originals

        cat = d.get("category", "unknown")

        # failure_mode — use persisted value if already backfilled
        nd["failure_mode"] = d.get("failure_mode") or CATEGORY_TO_FAILURE_MODE.get(
            cat, cat
        )

        # is_silent — use persisted value if already backfilled
        nd["is_silent"] = (
            d.get("is_silent")
            if "is_silent" in d
            else CATEGORY_IS_SILENT.get(cat, False)
        )

        # title (sanitize garbled Unicode from PR titles)
        raw_title = d.get("pr_title", "") or ""
        try:
            nd["title"] = raw_title.encode("utf-8", errors="replace").decode(
                "utf-8", errors="replace"
            )
        except Exception:
            nd["title"] = raw_title

        # features_affected — use persisted value if already backfilled,
        # otherwise infer from files_fixed (excluding hub files)
        if "features_affected" in d and d["features_affected"]:
            nd["features_affected"] = d["features_affected"]
        else:
            features_affected: set[str] = set()
            for fp in d.get("files_fixed", []):
                fp_norm = fp.replace("\\", "/")
                if fp_norm in hub_files:
                    continue  # skip hub files — too broad to be meaningful
                if fp_norm in file_to_features:
                    features_affected.update(file_to_features[fp_norm])
            nd["features_affected"] = sorted(features_affected)

        normalized.append(nd)

    return normalized


def _integrate_test_patterns(G: nx.MultiDiGraph, patterns_path: Path) -> None:
    """Load test_patterns.json and wire OraclePattern / FailureMode nodes
    plus TEST_USES_PATTERN / PATTERN_DETECTS / PATTERN_BLIND_SPOT edges
    into the existing graph.

    This is a non-destructive add-on: when test_patterns.json is absent
    the graph is built without testing-knowledge nodes (backward compat).
    """
    if not patterns_path.exists():
        print(
            "  (test_patterns.json not found — skipping testing-knowledge integration)"
        )
        return

    tp = load_json(patterns_path)

    # ── FailureMode nodes ────────────────────────────────────────
    for fm in tp.get("failure_modes", []):
        fm_id = fm["id"]
        if not G.has_node(fm_id):
            G.add_node(
                fm_id,
                type="FailureMode",
                name=fm.get("name", fm_id),
                description=fm.get("description", ""),
                is_silent=fm.get("is_silent", False),
                detectable_by_oracle=fm.get("detectable_by_oracle", True),
            )

    # ── OraclePattern nodes ──────────────────────────────────────
    for op in tp.get("oracle_patterns", []):
        op_id = op["id"]
        if not G.has_node(op_id):
            G.add_node(
                op_id,
                type="OraclePattern",
                name=op.get("name", op_id),
                category=op.get("category", "unknown"),
                max_safe_tokens=op.get("max_safe_tokens"),
                tolerance=op.get("tolerance"),
                severity=op.get("severity", "approximate"),
                requires_single_server=op.get("requires_single_server", False),
                npu_validated=op.get("npu_validated", False),
                npu_supported=op.get("npu_supported", True),
                reference_test=op.get("reference_test", ""),
                description=op.get("description", ""),
                limitation_detail=op.get("limitation_detail", ""),
            )

    # ── PATTERN_DETECTS edges ────────────────────────────────────
    for pd in tp.get("pattern_detects", []):
        src, tgt = pd["source"], pd["target"]
        if G.has_node(src) and G.has_node(tgt):
            # Avoid duplicates (idempotent on re-build)
            if not any(
                d.get("type") == "PATTERN_DETECTS"
                for u, v, d in G.out_edges(src, data=True)
                if v == tgt
            ):
                G.add_edge(
                    src,
                    tgt,
                    type="PATTERN_DETECTS",
                    confidence=pd.get("confidence", 1.0),
                    evidence=pd.get("evidence", ""),
                )

    # ── PATTERN_BLIND_SPOT edges ─────────────────────────────────
    for pb in tp.get("pattern_blind_spots", []):
        src, tgt = pb["source"], pb["target"]
        if G.has_node(src) and G.has_node(tgt):
            if not any(
                d.get("type") == "PATTERN_BLIND_SPOT"
                for u, v, d in G.out_edges(src, data=True)
                if v == tgt
            ):
                G.add_edge(
                    src,
                    tgt,
                    type="PATTERN_BLIND_SPOT",
                    reason=pb.get("reason", ""),
                    mitigation=pb.get("mitigation", ""),
                )

    # ── TEST_USES_PATTERN edges ──────────────────────────────────
    # Wired by test_file path — matches TestCase node IDs in the graph.
    for tpa in tp.get("test_pattern_assignments", []):
        test_id = tpa["test_id"]
        if not G.has_node(test_id):
            continue  # test not in this build (e.g. CUDA-only tests on NPU build)
        for p in tpa.get("patterns", []):
            pid = p["pattern_id"]
            if not G.has_node(pid):
                continue
            if not any(
                d.get("type") == "TEST_USES_PATTERN"
                for u, v, d in G.out_edges(test_id, data=True)
                if v == pid
            ):
                G.add_edge(
                    test_id,
                    pid,
                    type="TEST_USES_PATTERN",
                    weight=p.get("weight", 1.0),
                    role=p.get("role", "primary"),
                    detail=p.get("detail", ""),
                )

    n_oracle = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "OraclePattern")
    n_fmode = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "FailureMode")
    print(
        f"  Test patterns integrated: {n_oracle} oracle patterns, {n_fmode} failure modes"
    )


# ============================================================
# Graph Construction
# ============================================================


def build_risk_graph(baseline_dir: Path = None) -> nx.MultiDiGraph:
    """Build the complete NPU risk knowledge graph from all baseline data sources.

    Returns a MultiDiGraph with 5 node types and 7 edge types.
    """
    if baseline_dir is None:
        baseline_dir = BASELINE_DIR

    G = nx.MultiDiGraph()
    G.graph["name"] = "NPU Risk Knowledge Graph"
    G.graph["built_at"] = now_iso()
    G.graph["git_commit"] = get_current_commit()

    # Load all data sources
    features_data = load_json(baseline_dir / "features.json")
    tests_data = load_json(baseline_dir / "tests.json")
    profiles_data = load_json(baseline_dir / "risk_profiles.json")
    dep_map_data = load_json(baseline_dir / "dependency_map.json")
    defect_db = load_json(DB_DIR / "defect_db.json")

    features = features_data.get("features", [])
    tests = tests_data.get("tests", [])
    profiles = {p["feature"]: p for p in profiles_data.get("profiles", [])}
    dep_map = dep_map_data.get("dependency_map", {})
    defects_raw = defect_db.get("defects", [])
    near_misses = defect_db.get("near_misses", [])

    # Normalize: new Agent-workflow schema → format expected by Steps 4-6
    defects = _normalize_defect_schema(defects_raw, dep_map, features)

    # ============================================================
    # Step 1: Create Feature nodes
    # ============================================================
    for f in features:
        name = f["name"]
        p = profiles.get(name, {})
        G.add_node(
            name,
            type="Feature",
            category=f.get("category", "unknown"),
            risk_score=p.get("composite_score", 0),
            probability=p.get("probability", {}).get("total", 0),
            impact=p.get("impact", {}).get("total", 0),
            level=p.get("level", "low"),
            test_depth=p.get("test_depth", "L3"),
            fingerprint=f.get("fingerprint", ""),
            n_tests=sum(1 for t in tests if name in t.get("features_tested", [])),
            n_source_files=len(f.get("source_files", [])),
            last_modified=f.get("last_modified", ""),
            complexity=f.get("complexity", 1),
        )

    # ============================================================
    # Step 2: Create SourceFile nodes + IMPLEMENTS_IN edges
    # ============================================================
    source_files_seen = set()
    for f in features:
        name = f["name"]
        for sf in f.get("source_files", []):
            sf_norm = sf.replace("\\", "/")
            file_id = sf_norm.split("/")[-1]

            # Use full path as node ID for uniqueness
            node_id = sf_norm

            if node_id not in source_files_seen:
                # Infer file_type from extension
                ext = Path(sf_norm).suffix.lower()
                _ft_map = {
                    ".py": "python",
                    ".cpp": "cpp",
                    ".h": "header",
                    ".cu": "cuda",
                    ".sh": "shell",
                    ".yaml": "yaml",
                    ".yml": "yaml",
                    ".json": "json",
                    ".md": "markdown",
                }
                _file_type = _ft_map.get(ext, ext.lstrip(".") if ext else "unknown")

                # Count lines of code (best-effort, skip if file missing)
                _loc = 0
                _file_path = REPO_ROOT / sf_norm
                try:
                    if _file_path.is_file():
                        _loc = sum(
                            1
                            for _ in open(
                                _file_path, encoding="utf-8", errors="replace"
                            )
                        )
                except Exception:
                    pass

                G.add_node(
                    node_id,
                    type="SourceFile",
                    file_name=file_id,
                    file_type=_file_type,
                    lines_of_code=_loc,
                    complexity=f.get("complexity", 1),
                    platform_divergence=(
                        "fully_npu_private"
                        if "hardware_backend/npu" in sf_norm
                        else "partial"
                    ),
                )
                source_files_seen.add(node_id)

            # IMPLEMENTS_IN: Feature → SourceFile
            weight = 1.0 / len(f["source_files"]) if f["source_files"] else 1.0
            G.add_edge(
                name,
                node_id,
                type="IMPLEMENTS_IN",
                weight=round(weight, 2),
            )

    # ============================================================
    # Step 3: Create TestCase nodes + TESTED_BY / COVERS edges
    # ============================================================
    for t in tests:
        test_id = t["test_file"]
        # Total estimated time from ci_registrations (take max if multiple)
        _registrations = t.get("ci_registrations") or [{"est_time": 0}]
        _est_time = max((r.get("est_time", 0) for r in _registrations), default=0)
        G.add_node(
            test_id,
            type="TestCase",
            test_type=(
                t["test_methods"][0]["type"] if t.get("test_methods") else "unknown"
            ),
            quality_score=t.get("quality_score", 0),
            assertion_type=(
                t["test_methods"][0].get("assertion_type", "threshold")
                if t.get("test_methods")
                else "unknown"
            ),
            has_oracle=any(
                m.get("has_reference_oracle", False) for m in t.get("test_methods", [])
            ),
            ci_suite=(
                t["ci_registrations"][0].get("suite", "unknown")
                if t.get("ci_registrations")
                else "unknown"
            ),
            n_methods=len(t.get("test_methods", [])),
            test_class=t.get("test_class", ""),
            est_time_seconds=_est_time,
        )

        # TESTED_BY: Feature → TestCase
        for feat_name in t.get("features_tested", []):
            if G.has_node(feat_name):
                G.add_edge(
                    feat_name,
                    test_id,
                    type="TESTED_BY",
                    weight=t.get("quality_score", 1),
                )

        # COVERS: TestCase → SourceFile (via TESTED_BY → IMPLEMENTS_IN chain)
        # A test covers the source files that implement the features it tests.
        for feat_name in t.get("features_tested", []):
            if G.has_node(feat_name):
                for u, v, k, d in G.out_edges(feat_name, keys=True, data=True):
                    if d.get("type") == "IMPLEMENTS_IN":
                        G.add_edge(
                            test_id,
                            v,
                            type="COVERS",
                            weight=round(
                                d.get("weight", 0.5) * t.get("quality_score", 1) / 5, 2
                            ),
                        )

    # ============================================================
    # Step 3.5: Integrate Test Pattern Knowledge (OraclePattern +
    # FailureMode nodes, TEST_USES_PATTERN / PATTERN_DETECTS /
    # PATTERN_BLIND_SPOT edges)
    #
    # Loads test_patterns.json which captures reusable oracle
    # patterns (exact-match, logprob rescore, token oracle, etc.)
    # and their detection capabilities.  This makes the graph
    # queryable for test design guidance: "which oracle pattern
    # can detect silent precision loss?"  "what is my test blind
    # to?"
    # ============================================================
    _integrate_test_patterns(G, DB_DIR / "test_patterns.json")

    # ============================================================
    # Step 4: Create Defect nodes + AFFECTED_BY / FIXED_IN edges
    # ============================================================
    for d in defects:
        bug_id = d["bug_id"]
        G.add_node(
            bug_id,
            type="Defect",
            title=d.get("title", ""),
            root_cause=d.get("root_cause", ""),
            category=d.get("category", "unknown"),
            severity=d.get("severity", 3),
            is_silent=d.get("is_silent", False),
            failure_mode=d.get("failure_mode", "unknown"),
            date_fixed=d.get("date_fixed", ""),
            source=d.get("source", "commit"),
            confidence=d.get("confidence", 0.5),
        )

        # AFFECTED_BY: Feature ← Defect (defect affects feature)
        for feat_name in d.get("features_affected", []):
            if G.has_node(feat_name):
                G.add_edge(
                    bug_id,
                    feat_name,
                    type="AFFECTED_BY",
                    weight=d.get("severity", 3),
                )

        # FIXED_IN: Defect → SourceFile
        for file_path in d.get("files_fixed", []):
            file_norm = file_path.replace("\\", "/")
            if G.has_node(file_norm) and G.nodes[file_norm].get("type") == "SourceFile":
                G.add_edge(
                    bug_id,
                    file_norm,
                    type="FIXED_IN",
                    weight=round(1.0 / max(1, len(d.get("files_fixed", []))), 2),
                )

    # ============================================================
    # Step 5: Near-Miss nodes (low-weight risk signals from PR reviews)
    # ============================================================
    for nm in near_misses:
        nm_id = f"NEAR_MISS_PR{nm.get('source_pr', 'unknown')}"
        G.add_node(
            nm_id,
            type="NearMiss",
            risk_pattern=nm.get("risk_pattern", ""),
            module=nm.get("module", ""),
            symptom=nm.get("symptom_if_not_caught", ""),
            weight=nm.get("weight", 0.3),
        )
        for feat_name in nm.get("would_affect_features", []):
            if G.has_node(feat_name):
                G.add_edge(
                    nm_id,
                    feat_name,
                    type="WARNS_OF",
                    weight=nm.get("weight", 0.3),
                )

    # ============================================================
    # Step 6: Infer SHARES_DEFECT_PATTERN edges (Feature — Feature)
    # ============================================================
    # When two features are affected by defects that share the same
    # failure_mode, they form a systemic risk coupling. This is the
    # edge that makes cross-feature defect propagation possible.
    shared_defect_counts = {}  # (feat_a, feat_b) → count
    shared_modes = {}  # (feat_a, feat_b) → set of failure_modes

    for d in defects:
        affected = [fn for fn in d.get("features_affected", []) if G.has_node(fn)]
        if len(affected) < 2:
            continue
        mode = d.get("failure_mode", "unknown")
        for i, a in enumerate(affected):
            for b in affected[i + 1 :]:
                key = tuple(sorted([a, b]))
                shared_defect_counts[key] = shared_defect_counts.get(key, 0) + 1
                shared_modes.setdefault(key, set()).add(mode)

    for (a, b), count in shared_defect_counts.items():
        weight = min(5, count)  # cap at 5
        G.add_edge(
            a,
            b,
            type="SHARES_DEFECT_PATTERN",
            weight=weight,
            shared_defect_count=count,
            shared_failure_modes=sorted(shared_modes.get((a, b), [])),
        )
        G.add_edge(
            b,
            a,
            type="SHARES_DEFECT_PATTERN",
            weight=weight,
            shared_defect_count=count,
            shared_failure_modes=sorted(shared_modes.get((a, b), [])),
        )

    # ============================================================
    # Step 7: Infer DEPENDS_ON edges (Feature → Feature)
    # ============================================================
    # From explicit dependent_features
    for f in features:
        for dep in f.get("dependent_features", []):
            if G.has_node(dep):
                G.add_edge(
                    f["name"],
                    dep,
                    type="DEPENDS_ON",
                    weight=3,
                    via="explicit_dependency",
                )

    # From shared source files → indirect coupling
    file_to_features = {}
    for u, v, data in G.edges(data=True):
        if data.get("type") == "IMPLEMENTS_IN":
            file_to_features.setdefault(v, []).append(u)

    for file_id, feat_list in file_to_features.items():
        if len(feat_list) > 1:
            for i, a in enumerate(feat_list):
                for b in feat_list[i + 1 :]:
                    if not G.has_edge(a, b):
                        G.add_edge(
                            a,
                            b,
                            type="DEPENDS_ON",
                            weight=2,
                            via=f"shared_file:{file_id.split('/')[-1]}",
                        )
                        G.add_edge(
                            b,
                            a,
                            type="DEPENDS_ON",
                            weight=2,
                            via=f"shared_file:{file_id.split('/')[-1]}",
                        )

    # ============================================================
    # Step 8: Create external Dependency nodes
    # ============================================================
    ext_deps = [
        {"name": "CANN", "version": "8.5.0", "dep_type": "runtime", "upgrade_risk": 4},
        {
            "name": "torch_npu",
            "version": "2.5.1",
            "dep_type": "python_pkg",
            "upgrade_risk": 3,
        },
        {
            "name": "sgl_kernel_npu",
            "version": "2026.05",
            "dep_type": "kernel",
            "upgrade_risk": 4,
        },
        {
            "name": "HCCL",
            "version": "8.5.0",
            "dep_type": "communication",
            "upgrade_risk": 3,
        },
    ]
    for dep in ext_deps:
        G.add_node(dep["name"], type="Dependency", **dep)
        # All NPU features depend on CANN and torch_npu
        for f in features:
            coupling = 5 if dep["name"] in ("CANN", "torch_npu") else 3
            G.add_edge(
                f["name"],
                dep["name"],
                type="DEPENDS_ON",
                weight=coupling,
            )

    # ============================================================
    # Step 9: Populate computed node properties
    # ============================================================

    # SourceFile.n_bugs: count of FIXED_IN edges pointing to this file
    file_bug_counts = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("type") == "FIXED_IN":
            file_bug_counts[v] = file_bug_counts.get(v, 0) + 1
    for node, data in G.nodes(data=True):
        if data.get("type") == "SourceFile":
            data["n_bugs"] = file_bug_counts.get(node, 0)

    # Dependency.n_consumers: count of Features depending on this dep
    dep_consumers = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("type") == "DEPENDS_ON" and G.nodes[v].get("type") == "Dependency":
            dep_consumers[v] = dep_consumers.get(v, 0) + 1
    for node, data in G.nodes(data=True):
        if data.get("type") == "Dependency":
            data["n_consumers"] = dep_consumers.get(node, 0)
            data["provider"] = {
                "CANN": "Huawei",
                "torch_npu": "Huawei",
                "sgl_kernel_npu": "SGLang",
                "HCCL": "Huawei",
            }.get(node, "unknown")

    # Feature.n_effective_tests: TESTED_BY edges with quality_score >= 3
    # (tests that provide meaningful coverage, not just smoke/accuracy checks)
    feat_effective = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("type") == "TESTED_BY" and G.nodes[v].get("quality_score", 0) >= 3:
            feat_effective[u] = feat_effective.get(u, 0) + 1
    for node, data in G.nodes(data=True):
        if data.get("type") == "Feature":
            data["n_effective_tests"] = feat_effective.get(node, 0)

    # ============================================================
    # Step 10: Compute graph metrics
    # ============================================================
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)

    # Betweenness centrality (undirected approximation)
    try:
        G_undirected = nx.Graph()
        for u, v, key in G.edges(keys=True):
            G_undirected.add_edge(u, v)
        if G_undirected.number_of_nodes() > 1:
            bc = nx.betweenness_centrality(G_undirected)
            for node in G.nodes():
                G.nodes[node]["betweenness"] = bc.get(node, 0.0)
    except Exception:
        for node in G.nodes():
            G.nodes[node]["betweenness"] = 0.0

    return G


# ============================================================
# Export
# ============================================================


def export_graph(G: nx.MultiDiGraph, output_path: Path = None):
    """Export graph as node-link JSON for visualization and persistence."""
    if output_path is None:
        output_path = GRAPH_DIR / "latest_graph.json"

    # Convert MultiDiGraph to node-link format
    data = nx.node_link_data(G, edges="edges")

    # Add metadata
    data["graph_metadata"] = {
        "name": G.graph.get("name", ""),
        "built_at": G.graph.get("built_at", ""),
        "git_commit": G.graph.get("git_commit", ""),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }

    # Statistics
    node_types = {}
    edge_types = {}
    for n, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    for u, v, d in G.edges(data=True):
        t = d.get("type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1
    data["statistics"] = {
        "node_types": node_types,
        "edge_types": edge_types,
    }

    save_json(output_path, data)
    print(f"Graph exported: {output_path}")
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"  Node types: {node_types}")
    print(f"  Edge types: {edge_types}")

    return data


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build NPU Risk Knowledge Graph")
    parser.add_argument(
        "--baseline", default="latest", help="Baseline directory name under baselines/"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: graph/latest_graph.json)",
    )
    args = parser.parse_args()

    baseline_dir = ROOT / "baselines" / args.baseline
    if not baseline_dir.exists():
        print(f"ERROR: Baseline not found: {baseline_dir}")
        print(
            "Run `/npu-risk-graph baseline` first (or: python .claude/skills/npu-risk-graph/scripts/run_full.py)."
        )
        sys.exit(1)

    print(f"Building graph from baseline: {baseline_dir}")
    G = build_risk_graph(baseline_dir)

    output = Path(args.output) if args.output else None
    export_graph(G, output)

    print("\nGraph built successfully.")
    print("Use queries.py for risk analysis queries.")
