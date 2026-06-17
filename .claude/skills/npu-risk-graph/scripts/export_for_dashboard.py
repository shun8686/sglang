#!/usr/bin/env python3
"""
Dashboard Export — transforms NPU risk graph into CI-friendly JSON formats.

Output modes:
    --format summary   : Top-level risk summary (for CI badge / PR comment)
    --format nodes     : Flat node list with key properties (for table views)
    --format d3        : D3.js force-graph JSON (for web embedding)
    --format risks     : Risk-ranked feature list (for priority dashboard)

Usage:
    python .claude/skills/npu-risk-graph/scripts/export_for_dashboard.py --format summary
    python .claude/skills/npu-risk-graph/scripts/export_for_dashboard.py --format nodes --output nodes.json
    python .claude/skills/npu-risk-graph/scripts/export_for_dashboard.py --format d3 --max-nodes 80
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx
from common import GRAPH_DIR, load_json, save_json


def load_graph(graph_path: Path = None) -> nx.MultiDiGraph:
    if graph_path is None:
        graph_path = GRAPH_DIR / "latest_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found: {graph_path}\n"
            "Run '/npu-risk-graph build' first "
            "(python .claude/skills/npu-risk-graph/scripts/build_graph.py)."
        )
    data = load_json(graph_path)
    G = nx.node_link_graph(data, edges="edges")
    if not isinstance(G, nx.MultiDiGraph):
        G2 = nx.MultiDiGraph()
        G2.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            G2.add_edge(u, v, **d)
        G = G2
    return G


# ============================================================
# Format: summary
# ============================================================


def export_summary(G: nx.MultiDiGraph) -> dict:
    """Top-level risk summary for CI badges and PR comments."""
    features = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "Feature"]

    risk_levels = Counter(d.get("level", "unknown") for _, d in features)
    total_tests = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "TestCase")
    total_defects = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Defect")
    silent_defects = sum(
        1
        for _, d in G.nodes(data=True)
        if d.get("type") == "Defect" and d.get("is_silent")
    )

    # Top 5 riskiest features
    by_risk = sorted(features, key=lambda x: x[1].get("risk_score", 0), reverse=True)
    top_risks = [
        {
            "feature": n,
            "risk_score": d.get("risk_score", 0),
            "level": d.get("level", "?"),
            "n_tests": d.get("n_tests", 0),
            "n_effective_tests": d.get("n_effective_tests", 0),
        }
        for n, d in by_risk[:5]
    ]

    # Top 5 priority features (biggest gap)
    def _test_gap(feat_node):
        weights = [
            ed[3].get("weight", 0)
            for ed in G.out_edges(feat_node, keys=True, data=True)
            if ed[3].get("type") == "TESTED_BY"
        ]
        best = max(weights) if weights else 0
        return 5 - best

    by_gap = sorted(
        features,
        key=lambda x: x[1].get("risk_score", 0) * _test_gap(x[0]),
        reverse=True,
    )
    top_gaps = [
        {
            "feature": n,
            "risk_score": d.get("risk_score", 0),
            "n_tests": d.get("n_tests", 0),
        }
        for n, d in by_gap[:5]
    ]

    return {
        "graph": G.graph.get("name", ""),
        "commit": G.graph.get("git_commit", ""),
        "updated": G.graph.get("last_updated_at") or G.graph.get("built_at", ""),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "risk_distribution": dict(risk_levels),
        "total_tests": total_tests,
        "total_defects": total_defects,
        "silent_defects": silent_defects,
        "silent_ratio": (
            round(silent_defects / total_defects, 3) if total_defects else 0
        ),
        "top_risks": top_risks,
        "top_test_gaps": top_gaps,
    }


# ============================================================
# Format: nodes
# ============================================================


def export_nodes(G: nx.MultiDiGraph, node_types: list[str] = None) -> list[dict]:
    """Flat node list filtered by type, with key properties."""
    if node_types is None:
        node_types = ["Feature", "SourceFile", "TestCase", "Defect", "Dependency"]

    result = []
    for n, d in G.nodes(data=True):
        ntype = d.get("type", "unknown")
        if ntype not in node_types:
            continue

        entry = {"id": n, "type": ntype}
        if ntype == "Feature":
            entry.update(
                {
                    "risk_score": d.get("risk_score", 0),
                    "level": d.get("level", "?"),
                    "category": d.get("category", "?"),
                    "n_tests": d.get("n_tests", 0),
                    "n_effective_tests": d.get("n_effective_tests", 0),
                    "betweenness": round(d.get("betweenness", 0), 4),
                }
            )
        elif ntype == "SourceFile":
            entry.update(
                {
                    "file_name": d.get("file_name", ""),
                    "lines_of_code": d.get("lines_of_code", 0),
                    "n_bugs": d.get("n_bugs", 0),
                    "platform_divergence": d.get("platform_divergence", "?"),
                }
            )
        elif ntype == "TestCase":
            entry.update(
                {
                    "quality_score": d.get("quality_score", 0),
                    "has_oracle": d.get("has_oracle", False),
                    "ci_suite": d.get("ci_suite", "?"),
                    "est_time_seconds": d.get("est_time_seconds", 0),
                    "test_class": d.get("test_class", ""),
                }
            )
        elif ntype == "Defect":
            entry.update(
                {
                    "severity": d.get("severity", 0),
                    "is_silent": d.get("is_silent", False),
                    "failure_mode": d.get("failure_mode", "?"),
                    "category": d.get("category", "?"),
                    "confidence": d.get("confidence", 0),
                }
            )
        elif ntype == "Dependency":
            entry.update(
                {
                    "version": d.get("version", "?"),
                    "upgrade_risk": d.get("upgrade_risk", 0),
                    "n_consumers": d.get("n_consumers", 0),
                    "provider": d.get("provider", "?"),
                }
            )

        result.append(entry)

    return result


# ============================================================
# Format: d3
# ============================================================


def export_d3(G: nx.MultiDiGraph, max_nodes: int = 0) -> dict:
    """D3.js force-graph compatible JSON with color/size encoding."""
    NODE_COLORS = {
        "Feature": "#4da6ff",
        "SourceFile": "#3fb950",
        "TestCase": "#d29922",
        "Defect": "#f85149",
        "NearMiss": "#f778ba",
        "Dependency": "#a371f7",
    }

    # Prioritize important nodes
    nodes = []
    type_priority = [
        "Feature",
        "SourceFile",
        "Defect",
        "Dependency",
        "TestCase",
        "NearMiss",
    ]
    for ntype in type_priority:
        for n, d in G.nodes(data=True):
            if d.get("type") == ntype:
                nodes.append((n, d))
    if max_nodes and len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    node_ids = {n for n, _ in nodes}

    node_data = []
    for n, d in nodes:
        ntype = d.get("type", "unknown")
        node_data.append(
            {
                "id": n,
                "group": ntype,
                "color": NODE_COLORS.get(ntype, "#888"),
                "size": d.get("degree", 1),
                "risk_score": d.get("risk_score"),
                "level": d.get("level"),
                "label": n.split("/")[-1] if ntype == "SourceFile" else n,
            }
        )

    edge_data = []
    for u, v, d in G.edges(data=True):
        if u in node_ids and v in node_ids:
            edge_data.append(
                {
                    "source": u,
                    "target": v,
                    "type": d.get("type", "unknown"),
                    "weight": d.get("weight", 1),
                }
            )

    return {"nodes": node_data, "links": edge_data}


# ============================================================
# Format: risks
# ============================================================


def export_risks(G: nx.MultiDiGraph) -> list[dict]:
    """Risk-ranked feature list for priority dashboard."""
    features = []
    for n, d in G.nodes(data=True):
        if d.get("type") != "Feature":
            continue
        # Best test quality from TESTED_BY edges
        best_quality = max(
            (
                ed[3].get("weight", 0)
                for ed in G.out_edges(n, keys=True, data=True)
                if ed[3].get("type") == "TESTED_BY"
            ),
            default=0,
        )
        gap = 5 - best_quality
        priority_score = d.get("risk_score", 0) * gap
        features.append(
            {
                "feature": n,
                "risk_score": d.get("risk_score", 0),
                "level": d.get("level", "?"),
                "category": d.get("category", "?"),
                "probability": d.get("probability", 0),
                "impact": d.get("impact", 0),
                "n_tests": d.get("n_tests", 0),
                "n_effective_tests": d.get("n_effective_tests", 0),
                "best_test_quality": best_quality,
                "test_gap": gap,
                "priority_score": priority_score,
                "n_bugs": sum(
                    1
                    for _, _, _, ed in G.in_edges(n, keys=True, data=True)
                    if ed.get("type") == "AFFECTED_BY"
                ),
                "betweenness": round(d.get("betweenness", 0), 4),
            }
        )

    return sorted(features, key=lambda x: x["priority_score"], reverse=True)


# ============================================================
# CLI
# ============================================================

EXPORTERS = {
    "summary": export_summary,
    "nodes": export_nodes,
    "d3": export_d3,
    "risks": export_risks,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export NPU risk graph for dashboards")
    parser.add_argument("--graph", default=None, help="Graph JSON path")
    parser.add_argument(
        "--format",
        choices=list(EXPORTERS.keys()),
        default="summary",
        help="Export format",
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON path (default: stdout)"
    )
    parser.add_argument(
        "--max-nodes", type=int, default=0, help="Max nodes (d3 format)"
    )
    parser.add_argument(
        "--node-types", nargs="*", help="Filter node types (nodes format)"
    )
    args = parser.parse_args()

    G = load_graph(Path(args.graph) if args.graph else None)
    exporter = EXPORTERS[args.format]

    if args.format == "d3":
        result = exporter(G, max_nodes=args.max_nodes)
    elif args.format == "nodes":
        result = exporter(G, node_types=args.node_types)
    else:
        result = exporter(G)

    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Exported ({args.format}): {args.output}")
    else:
        print(json_str)
