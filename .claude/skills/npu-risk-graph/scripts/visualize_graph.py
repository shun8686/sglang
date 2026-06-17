#!/usr/bin/env python3
"""
NPU Risk Knowledge Graph — Interactive HTML Visualization

Generates a self-contained HTML file with a D3.js force-directed graph.
Color-coded by node type, sized by degree, with hover tooltips, search, and legend.

Usage:
    python .claude/skills/npu-risk-graph/scripts/visualize_graph.py                    # default output
    python .claude/skills/npu-risk-graph/scripts/visualize_graph.py --output risk.html # custom path
    python .claude/skills/npu-risk-graph/scripts/visualize_graph.py --max-nodes 100    # limit nodes
"""

import json
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
from common import GRAPH_DIR, load_json

# ============================================================
# Color scheme — matches design document §2.1 node type colors
# ============================================================

NODE_COLORS = {
    "Feature": "#4da6ff",  # blue
    "SourceFile": "#3fb950",  # green
    "TestCase": "#d29922",  # yellow
    "Defect": "#f85149",  # red
    "NearMiss": "#f778ba",  # pink
    "Dependency": "#a371f7",  # purple
}

RISK_LEVEL_COLORS = {
    "critical": "#f85149",
    "high": "#f0883e",
    "medium": "#d29922",
    "low": "#3fb950",
}

EDGE_COLORS = {
    "IMPLEMENTS_IN": "#3fb950",
    "TESTED_BY": "#d29922",
    "DEPENDS_ON": "#4da6ff",
    "AFFECTED_BY": "#f85149",
    "FIXED_IN": "#f0883e",
    "COVERS": "#39d2c0",
    "SHARES_DEFECT_PATTERN": "#a371f7",
    "WARNS_OF": "#f778ba",
}


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


def _short_label(node_id: str, node_type: str) -> str:
    """Human-readable short label for display."""
    if node_type == "SourceFile":
        return node_id.split("/")[-1]
    if node_type == "TestCase":
        parts = node_id.replace("\\", "/").split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    if node_type in ("Defect", "NearMiss"):
        return node_id
    return node_id


def _node_radius(data: dict) -> float:
    """Node radius based on type and metrics."""
    ntype = data.get("type", "unknown")
    if ntype == "Feature":
        risk = data.get("risk_score", 0)
        return 5 + risk / 5  # 5-14px
    if ntype == "SourceFile":
        bugs = data.get("n_bugs", 0)
        return 4 + bugs * 0.8  # 4-22px
    if ntype == "Defect":
        sev = data.get("severity", 3)
        return 3 + sev * 0.5
    if ntype == "Dependency":
        return 8
    if ntype == "TestCase":
        return 3.5
    return 4


def _node_tooltip(node_id: str, data: dict) -> str:
    """HTML tooltip content for a node."""
    ntype = data.get("type", "unknown")
    lines = [f"<b>{_short_label(node_id, ntype)}</b>", f"<i>{ntype}</i>"]

    if ntype == "Feature":
        lines += [
            f"Risk: {data.get('risk_score', '?')} ({data.get('level', '?')})",
            f"Tests: {data.get('n_tests', 0)} (eff: {data.get('n_effective_tests', 0)})",
            f"Category: {data.get('category', '?')}",
        ]
    elif ntype == "SourceFile":
        lines += [
            f"LOC: {data.get('lines_of_code', '?')}",
            f"Bugs: {data.get('n_bugs', 0)}",
            f"Divergence: {data.get('platform_divergence', '?')}",
        ]
    elif ntype == "TestCase":
        lines += [
            f"Quality: {data.get('quality_score', '?')}/5",
            f"Oracle: {'yes' if data.get('has_oracle') else 'no'}",
            f"Suite: {data.get('ci_suite', '?')}",
        ]
    elif ntype == "Defect":
        lines += [
            f"Severity: {data.get('severity', '?')}",
            f"Silent: {'yes' if data.get('is_silent') else 'no'}",
            f"Mode: {data.get('failure_mode', '?')}",
        ]
    elif ntype == "Dependency":
        lines += [
            f"Version: {data.get('version', '?')}",
            f"Consumers: {data.get('n_consumers', '?')}",
            f"Risk: {data.get('upgrade_risk', '?')}",
        ]

    return "<br>".join(lines)


def generate_html(
    G: nx.MultiDiGraph,
    output_path: Path,
    max_nodes: int = 0,
    title: str = "NPU Risk Knowledge Graph",
):
    """Generate a self-contained D3.js interactive graph HTML file."""

    # Build node list (filter by type priority if max_nodes set)
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

    # Build edge list (only edges where both endpoints are in the node set)
    edges = []
    for u, v, d in G.edges(data=True):
        if u in node_ids and v in node_ids:
            edges.append((u, v, d))

    # Serialize to JSON for embedding
    graph_data = {
        "nodes": [
            {
                "id": n,
                "label": _short_label(n, d.get("type", "")),
                "type": d.get("type", "unknown"),
                "color": NODE_COLORS.get(d.get("type", ""), "#888"),
                "radius": _node_radius(d),
                "tooltip": _node_tooltip(n, d),
                "risk_level": d.get("level", ""),
                "risk_score": d.get("risk_score", 0),
                "degree": d.get("degree", 0),
            }
            for n, d in nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "type": d.get("type", "unknown"),
                "color": EDGE_COLORS.get(d.get("type", ""), "#888"),
                "weight": d.get("weight", 1),
            }
            for u, v, d in edges
        ],
    }

    graph_json = json.dumps(graph_data, ensure_ascii=False)
    edge_colors_json = json.dumps(EDGE_COLORS, ensure_ascii=False)
    node_colors_json = json.dumps(NODE_COLORS, ensure_ascii=False)

    # Statistics
    node_types = {}
    edge_types = {}
    for nd in graph_data["nodes"]:
        node_types[nd["type"]] = node_types.get(nd["type"], 0) + 1
    for ed in graph_data["edges"]:
        edge_types[ed["type"]] = edge_types.get(ed["type"], 0) + 1

    stats_html = "<br>".join(
        f'<span style="color:{NODE_COLORS.get(t,"#888")}">{t}: {c}</span>'
        for t, c in sorted(node_types.items())
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0d1117; color:#c9d1d9; overflow:hidden; }}
#graph {{ width:100vw; height:100vh; }}
#panel {{ position:absolute; top:12px; left:12px; background:rgba(22,27,34,.94); border:1px solid #30363d; border-radius:8px; padding:14px 18px; font-size:13px; max-width:280px; z-index:10; }}
#panel h2 {{ font-size:15px; margin-bottom:6px; color:#e6edf3; }}
#panel .stats {{ font-size:11px; line-height:1.6; color:#8b949e; margin-bottom:8px; }}
#search {{ width:100%; padding:5px 8px; border:1px solid #30363d; border-radius:4px; background:#0d1117; color:#c9d1d9; font-size:12px; margin-bottom:8px; }}
#legend {{ font-size:11px; line-height:1.8; }}
#legend span {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }}
#tooltip {{ position:absolute; background:rgba(22,27,34,.96); border:1px solid #30363d; border-radius:6px; padding:8px 12px; font-size:11px; pointer-events:none; opacity:0; z-index:20; max-width:260px; line-height:1.5; }}
.links line {{ stroke-opacity:0.25; }}
.links line:hover {{ stroke-opacity:0.7; }}
.nodes circle {{ stroke:#0d1117; stroke-width:1.2; cursor:pointer; }}
.nodes circle:hover {{ stroke:#e6edf3; stroke-width:2; }}
.nodes text {{ font-size:8px; fill:#8b949e; pointer-events:none; }}
</style>
</head>
<body>
<div id="panel">
  <h2>{title}</h2>
  <div class="stats">{stats_html}<br>Edges: {len(edges)}</div>
  <input id="search" type="text" placeholder="Search nodes..." oninput="filterNodes(this.value)">
  <div id="legend"></div>
</div>
<div id="tooltip"></div>
<svg id="graph"></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {graph_json};
const EDGE_COLORS = {edge_colors_json};
const NODE_COLORS = {node_colors_json};

const svg = d3.select("#graph"),
      width = window.innerWidth,
      height = window.innerHeight;

const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1, 4]).on("zoom", (event) => g.attr("transform", event.transform)));

const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.edges).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-120))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide().radius(d => d.radius + 2));

const link = g.append("g").attr("class", "links").selectAll("line")
  .data(data.edges).join("line")
  .attr("stroke", d => d.color)
  .attr("stroke-width", d => Math.max(0.5, Math.min(3, d.weight / 2)));

const node = g.append("g").attr("class", "nodes").selectAll("g")
  .data(data.nodes).join("g").call(d3.drag()
    .on("start", (event,d) => {{ if(!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on("drag", (event,d) => {{ d.fx=event.x; d.fy=event.y; }})
    .on("end", (event,d) => {{ if(!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}));

node.append("circle")
  .attr("r", d => d.radius)
  .attr("fill", d => d.risk_level ? (d.risk_level=="critical"?"#f85149":d.risk_level=="high"?"#f0883e":d.color) : d.color)
  .on("mouseover", (event, d) => {{
    d3.select("#tooltip").style("opacity",1).html(d.tooltip)
      .style("left",(event.pageX+12)+"px").style("top",(event.pageY-10)+"px");
    link.style("stroke-opacity", e => (e.source.id===d.id||e.target.id===d.id) ? 0.7 : 0.08);
  }})
  .on("mouseout", () => {{
    d3.select("#tooltip").style("opacity",0);
    link.style("stroke-opacity", 0.25);
  }})
  .on("click", (event, d) => {{
    d3.select("#search").property("value", d.label);
    filterNodes(d.label);
  }});

node.append("text").text(d => d.label).attr("dx", d => d.radius+3).attr("dy", 3);

simulation.on("tick", () => {{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

// Legend
const legend = d3.select("#legend");
Object.entries(NODE_COLORS).forEach(([type, color]) => {{
  const cnt = data.nodes.filter(n => n.type===type).length;
  if(cnt>0) legend.append("div").html(`<span style="background:${{color}}"></span> ${{type}} (${{cnt}})`);
}});

// Search/filter
function filterNodes(q) {{
  if(!q) {{
    node.style("opacity",1);
    link.style("display","");
    return;
  }}
  const lq = q.toLowerCase();
  node.style("opacity", d => (d.id.toLowerCase().includes(lq)||d.label.toLowerCase().includes(lq)||d.type.toLowerCase().includes(lq)) ? 1 : 0.12);
  link.style("display", e => {{
    const s = e.source.id||e.source, t = e.target.id||e.target;
    return (s.toLowerCase().includes(lq)||t.toLowerCase().includes(lq)) ? "" : "none";
  }});
}}

window.addEventListener("resize", () => location.reload());
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interactive NPU risk graph HTML"
    )
    parser.add_argument("--graph", default=None, help="Graph JSON path")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument(
        "--max-nodes", type=int, default=0, help="Limit node count (0 = all)"
    )
    args = parser.parse_args()

    graph_path = Path(args.graph) if args.graph else None
    G = load_graph(graph_path)

    output = Path(args.output) if args.output else (GRAPH_DIR / "risk_graph.html")
    generate_html(G, output, max_nodes=args.max_nodes)
    print(f"Interactive graph saved: {output}")
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"  Open in browser: file:///{output.resolve()}")
