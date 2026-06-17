# npu-risk-graph

NPU risk knowledge graph — build, update, query, and visualize. All scripts live under `.claude/skills/npu-risk-graph/scripts/`.

## Trigger

```
/npu-risk-graph                  → health check + risk overview
/npu-risk-graph features         → generate/update features.json from sglang docs + graphify
/npu-risk-graph baseline         → regenerate full baseline (run_full.py)
/npu-risk-graph build            → rebuild graph from baseline
/npu-risk-graph update           → incremental update after code changes
/npu-risk-graph backfill         → full backfill of features_affected into defect_db.json
/npu-risk-graph impact <file>    → change impact analysis
/npu-risk-graph pattern scan     → auto-detect oracle patterns in existing tests
/npu-risk-graph pattern list     → list all test→pattern assignments
/npu-risk-graph visualize        → generate interactive HTML
/npu-risk-graph query <name>     → feature/sub-feature risk report (Agent auto-scoping, see table below)
```

**`.sglang-risk/testcases/`** — design staging area. Generated test scripts go here; this directory is outside both the sglang source tree and the graph build process. To deploy: copy to `test/registered/`.

```

## Workflow

All commands run from the repo root.

### No args — Health check + risk overview

```bash
python .claude/skills/npu-risk-graph/scripts/export_for_dashboard.py --format summary 2>&1
```

Present a summary:
- Graph age: compare `updated` / `built_at` against HEAD. If mismatch: "Graph may be stale. Run `/npu-risk-graph update` or `/npu-risk-graph build`."
- Risk distribution (critical / high / medium / low)
- Silent defect ratio — if > 15%, flag as "high silent risk"
- Top 3 riskiest features with test counts
- Top 3 test gaps (highest risk with 0 effective tests)

If `latest_graph.json` missing: "No graph found. Run `/npu-risk-graph build` first."
If `baselines/latest/` missing: "No baseline data found. Run `/npu-risk-graph features` then `/npu-risk-graph baseline`."
If `features.json` missing but `baselines/latest/` exists: "Run `/npu-risk-graph features` first."

### features — Generate/update features.json

基于 sglang 官方特性文档 + graphify 知识图谱，生成或增量更新 NPU 特性定义文件。

- `features.json` **不存在** → 全量生成（Phase 1-6）
- `features.json` **已存在** → 增量更新（Step 1-6），保留人工字段

**设计原则：**
1. 特性名使用 sglang 官方名称，不加 `npu_`/`ascend_` 前缀
2. NPU 参与度决定纳入（strong / medium / weak / platform_agnostic / not_supported），不取决于代码量
3. 平台无关特性标记 `platform_agnostic`，NPU 不支持特性标记 `not_supported`（需逐个文档阅读确认）
4. 全部关联文件在单一 `source_files` 字段中

**详细工作流：** [`references/features-workflow.md`](references/features-workflow.md)
**格式定义：** [`references/features_schema.md`](references/features_schema.md)
**文档覆盖检测：** `python .claude/skills/npu-risk-graph/scripts/detect_new_docs.py`
**输出：** `.sglang-risk/baselines/latest/features.json`

### baseline — Regenerate full baseline

Agent Workflow mode (batch semantic test→feature mapping). Steps:

1. Prepare batch:
   ```bash
   python .claude/skills/npu-risk-graph/scripts/run_full.py 2>&1
   ```
   If output says "Found cached Agent results", skip to step 3. Otherwise note the batch count.

2. Run the Workflow:
   ```
   Workflow({scriptPath: ".claude/skills/npu-risk-graph/workflows/baseline-mapping.js", args: {batchCount: <N>}})
   ```
   Replace `<N>` with the batch count from step 1 (default: 16). The script defines `MAPPING_SCHEMA`, spawns one Agent per batch, and returns `{mappings: [...]}` to an output JSON file.

   Save the Workflow's returned JSON output (the `{mappings: [...]}` object) to a file, e.g. `.sglang-risk/workflow_baseline_result.json`.

3. Apply results (use `--workflow-results` to bridge the Workflow output):
   ```bash
   python .claude/skills/npu-risk-graph/scripts/run_full.py --workflow-results .sglang-risk/workflow_baseline_result.json 2>&1
   ```
   The `--workflow-results` flag copies the Workflow output to `agent_baseline_results.json` and applies it immediately. Omit `--workflow-results` if cached results already exist from a previous run.

Regenerates: `tests.json`, `risk_profiles.json`, `dependency_map.json`. (Uses existing `features.json` as input — run `/npu-risk-graph features` first to update features.)

Auto-backfills: missing or stale `features_affected` in defect_db.

Pipeline: Load features (+backfill failure_mode) → test→feature mapping (Agent, quality: 1-5) → Dependency map (file→feature reverse index) → Risk profiling (6 probability + 4 impact factors, +auto-backfill features_affected).

After completion, run `/npu-risk-graph build` to rebuild the graph.

### build — Rebuild graph from baseline JSON

```bash
python .claude/skills/npu-risk-graph/scripts/build_graph.py --baseline latest 2>&1
```

Requires `baselines/latest/`. Then run `/npu-risk-graph` (no args) to verify graph health.

### update — Incremental update after code changes

```bash
python .claude/skills/npu-risk-graph/scripts/run_delta.py --mode full_delta 2>&1
```

Pipeline: git diff → JSON direct mapping → KG n-hop propagation → fingerprint filter → incremental rescore → delta report → graph update.

Summarize: changed files, directly affected features, propagated features, risk changes (up/down), recommendation (BLOCK / WARNING / MERGABLE).

If no NPU changes: "No NPU-related changes found. Graph is up to date."

### backfill — Full backfill of features_affected into defect_db.json

Two-step: mechanical file→feature matching + Agent semantic mapping for unmatched defects.

1. Mechanical match:
   ```bash
   python .claude/skills/npu-risk-graph/scripts/backfill_defect_features.py 2>&1
   ```
   If output says "Found cached Agent defect mapping results", skip to step 3.

2. Run the Workflow for unmatched defects:
   ```
   Workflow({scriptPath: ".claude/skills/npu-risk-graph/workflows/defect-backfill.js", args: {batchCount: <N>}})
   ```
   Save the Workflow's returned JSON output to a file, e.g. `.sglang-risk/workflow_backfill_result.json`.

3. Apply results (use `--workflow-results` to bridge the Workflow output):
   ```bash
   python .claude/skills/npu-risk-graph/scripts/backfill_defect_features.py --workflow-results .sglang-risk/workflow_backfill_result.json 2>&1
   ```

Rebuilds `features_affected`, `failure_mode`, and `is_silent` for **every** defect. Unlike the auto-trigger in `run_full.py` (which now also detects stale feature name references), this does a **full forced refresh** — overwriting all existing values.

Use when:
- Manually added/edited defects in `defect_db.json`
- `features.json` was regenerated and feature names changed
- `dependency_map.json` was rebuilt and file→feature mappings shifted

After backfill, run `/npu-risk-graph build` to propagate the updated defect→feature links into the graph.

### impact \<file\> — Change impact analysis

```bash
python .claude/skills/npu-risk-graph/scripts/queries.py --impact <file> 2>&1
```

Shows direct + 1-hop + 2-hop propagation via `impacted_features()`.

For shared-defect coupling detail, use the import preamble below then:

```python
from queries import load_graph, impacted_features
G = load_graph()
r = impacted_features(G, "<file>")
for feat in r["direct"]:
    sdp = [(v, d.get("shared_defect_count", 0)) for u, v, k, d in G.out_edges(feat, keys=True, data=True) if d.get("type") == "SHARES_DEFECT_PATTERN"]
    if sdp:
        top = sorted(sdp, key=lambda x: x[1], reverse=True)[:3]
        print(f"{feat}: shares defects with {[(t[0], t[1]) for t in top]}")
```

Summarize: direct features, 1-hop, 2-hop, total regression scope, key shared-defect couplings.

### visualize — Generate interactive HTML

```bash
python .claude/skills/npu-risk-graph/scripts/visualize_graph.py --output .sglang-risk/graph/risk_graph.html 2>&1
```

Output path; note: "Open in browser. Drag nodes, scroll to zoom, hover for details."

### query \<name\> — Run a specific query

**CLI (use Bash):**

| Name | Command |
|------|---------|
| `feature <name>` | See "Feature / Sub-Feature Query" below |
| `report` | `python .claude/skills/npu-risk-graph/scripts/queries.py --report` |
| `blind` / `blind-spots` | `python .claude/skills/npu-risk-graph/scripts/queries.py --blind-spots` |
| `storm` / `perfect-storm` | `python .claude/skills/npu-risk-graph/scripts/queries.py --perfect-storm` |
| `hotspot` / `defect-hotspots` | `python .claude/skills/npu-risk-graph/scripts/queries.py --defect-hotspots` |
| `priority` / `test-priority` | `python .claude/skills/npu-risk-graph/scripts/queries.py --test-priority` |
| `orphans` / `orphan-tests` | `python .claude/skills/npu-risk-graph/scripts/queries.py --orphans` |
| `oracle-blind <test>` | `python .claude/skills/npu-risk-graph/scripts/queries.py --oracle-blind-spots <test>` |
| `oracle-rec <feat>` | `python .claude/skills/npu-risk-graph/scripts/queries.py --recommend-oracle <feat>` |
| `oracle-cov` | `python .claude/skills/npu-risk-graph/scripts/queries.py --pattern-coverage` |
| `backfill` | `python .claude/skills/npu-risk-graph/scripts/backfill_defect_features.py` |

### Test generation (within query flow)

Test code generation is NOT a separate command — it happens as a natural continuation
of the query conversation after the user reviews the risk report.

**Flow**:

1. `/npu-risk-graph query <feature>` produces a risk report with scored, actionable
   recommendations (e.g. `[T-001]` through `[T-004]`).  Each item names the defect it
   covers, the oracle pattern to use, and key parameters.

2. The user reviews the report and selects which items to implement (e.g. "write tests
   for T-001 and T-003, skip T-002").

3. The skill generates test code covering only the selected items, matching oracle
   patterns to TestCase methods:
   - `exact_match_short` → `test_output_token_identity` (16 tokens, dual-server)
   - `logprob_rescore` → `test_logprob_rescore_match` (64 tokens, single-server, tolerance 0.255)
   - PlanStream isolation → `test_output_identity_no_planstream` (env override)
   - Accept-rate sanity → `test_speculation_is_active` (server metrics check)

4. Output is written to `.sglang-risk/testcases/<test_file>.py`.  This directory is
   outside both the sglang source tree and the knowledge graph build process — copy to
   `test/registered/` to deploy.

#### Feature / Sub-Feature Query (Agent auto-scoping)

All `--feature` queries route through Agent semantic scoping — the script
**always** exits with code 2 to signal the skill layer, regardless of whether the
feature name matches exactly.

```
python .claude/skills/npu-risk-graph/scripts/queries.py --feature <name> 2>&1
```

**Step 1 — Find parent feature**: `prepare_agent_bundle()` locates the best
matching Feature node.  Exact name matches (including space→underscore
normalization) use the feature itself as parent.  Partial / variant names (e.g.
"eagle3", "Speculative decode") use keyword matching with a stem fallback
("eagle3" → "eagle" → matches source files).  The parent's full source_files /
defects / tests and risk profile are exported as a bundle JSON to
`.sglang-risk/graph/agent_bundle_<keyword>.json`.  Script exits with code 2.

If no parent can be found, the script prints an error with the list of available
feature names.

**Step 2 — Agent classification**: the skill layer reads the bundle JSON
and calls an Agent with this structured-output schema:

   ```json
   {
     "type": "object",
     "required": ["source_files", "defects", "tests", "summary"],
     "properties": {
       "source_files": {"type": "array", "items": {
         "type": "object", "required": ["path", "is_scoped", "rationale"],
         "properties": {
           "path": {"type": "string"},
           "is_scoped": {"type": "boolean"},
           "rationale": {"type": "string"}
         }
       }},
       "defects": {"type": "array", "items": {
         "type": "object", "required": ["bug_id", "is_scoped", "rationale"],
         "properties": {
           "bug_id": {"type": "string"},
           "is_scoped": {"type": "boolean"},
           "rationale": {"type": "string"}
         }
       }},
       "tests": {"type": "array", "items": {
         "type": "object", "required": ["path", "is_scoped", "rationale"],
         "properties": {
           "path": {"type": "string"},
           "is_scoped": {"type": "boolean"},
           "rationale": {"type": "string"}
         }
       }},
       "summary": {"type": "object", "required": ["key_risk_insight"], "properties": {
         "key_risk_insight": {"type": "string"}
       }}
     }
   }
   ```

3. Feed the Agent's output JSON back:

   ```
   python .claude/skills/npu-risk-graph/scripts/queries.py --apply-agent <agent_result.json> --keyword <name>
   ```

The Agent path catches items that simple keyword filtering misses (e.g. a defect
whose title says "spec decode" but never mentions "eagle", or a shared utility file
that Eagle3 is the primary NPU consumer of).

**Direct Agent bundle (skip Tier 1):**

```
python .claude/skills/npu-risk-graph/scripts/queries.py --prepare-agent <name>
```

**Apply a cached Agent result:**

```
python .claude/skills/npu-risk-graph/scripts/queries.py --apply-agent <result.json> --keyword <name>
```

**Presentation**: when the report has a `scoping` key, present it as a scoped
sub-feature view — show the parent feature name, the keyword, the filter method
("agent_semantic_classification"), and the scoped counts in each section header
(e.g. "Defect History (4 scoped from 8 total)").

**Python-only queries** — use this preamble, then call the function:

```python
import sys; sys.path.insert(0, ".claude/skills/npu-risk-graph/scripts")
import networkx as nx
from queries import load_graph, cann_upgrade_blast_radius, high_coupling_no_test, n_hop_impact, cross_feature_defect_patterns, test_redundancy_check, longest_dependency_chain, oracle_blind_spots, recommend_oracle_for_feature, test_pattern_coverage
G = load_graph()
```

| Name | Function | What it answers |
|------|----------|----------------|
| `blast` | `cann_upgrade_blast_radius(G)` | CANN upgrade blast radius |
| `coupling` | `high_coupling_no_test(G)` | High out-degree files with zero COVERS |
| `nhop <f>` | `n_hop_impact(G, "<f>", hops=2)` | 2-hop BFS from feature |
| `patterns` | `cross_feature_defect_patterns(G)` | Failure modes affecting >= 2 features |
| `redundancy` | `test_redundancy_check(G)` | Test pairs covering same features |
| `chain` | `longest_dependency_chain(G)` | Deepest DEPENDS_ON chains |
| `blindspot` | `oracle_blind_spots(G)` | What failure modes each test is blind to |
| `oracle` | `recommend_oracle_for_feature(G, "<f>")` | Oracle patterns to cover feature's defect modes |
| `patcov` | `test_pattern_coverage(G)` | Oracle pattern usage across all tests |

For Python-only queries, present results in a readable table. For `report`, print the full markdown.

## What the KG adds beyond JSON

When presenting results, highlight insights that JSON analysis alone cannot provide:

- **Shared defect patterns**: which other features share the same bug types → test strategies can be reused
- **Cascade impact**: n-hop propagation along DEPENDS_ON / SHARES_DEFECT_PATTERN edges
- **Blind spots**: high-betweenness, low-risk nodes (e.g., CANN has risk=0 but all features depend on it — a CANN upgrade's blast radius is invisible to JSON risk scores)
- **Defect density per file**: SourceFile.n_bugs / SourceFile.lines_of_code → bug-prone files
- **Structural coupling**: betweenness centrality identifies hidden hubs that JSON risk scores miss

## File reference

```
.claude/skills/npu-risk-graph/
├── SKILL.md
├── references/
│   ├── features-workflow.md
│   └── features_schema.md
├── workflows/
│   ├── baseline-mapping.js        # Agent Workflow: test→feature mapping
│   └── defect-backfill.js         # Agent Workflow: defect→feature mapping
└── scripts/
    ├── common.py                 # shared utilities (paths, JSON I/O, etc.)
    ├── run_full.py               # full baseline analysis (Agent Workflow, two-step)
    ├── run_delta.py              # incremental delta engine
    ├── prepare_agent_baseline.py # AST-parses test files → batches
    ├── apply_agent_baseline.py   # validates + merges Agent results
    ├── baseline_prompts.py       # Agent prompt templates
    ├── backfill_defect_features.py
    ├── detect_new_docs.py        # doc coverage checker
    ├── detect_oracle_patterns.py # AST scan tests → auto-detect oracle patterns
    ├── validate_features.py      # schema validation for features.json
    ├── build_graph.py            # JSON baseline → NetworkX graph (+ test patterns)
    ├── update_graph.py           # delta report → graph mutation
    ├── queries.py                # 15 query functions + CLI (incl. test-knowledge)
    ├── visualize_graph.py        # D3.js HTML generation
    └── export_for_dashboard.py   # summary / nodes / d3 / risks

.sglang-risk/                      # data directory (outputs, not scripts)
├── testcases/                    # generated test design artifacts (NOT in KG build)
│   └── test_npu_eagle3_oracle.py # canonical dual-oracle eagle3 NPU test
├── graph/
│   ├── latest_graph.json         # current graph (node-link format)
│   ├── snapshots/                # pre-mutation graph backups
│   └── agent_bundle_*.json       # Agent semantic scoping bundles
├── baselines/latest/             # baseline JSONs (run_full.py output)
├── deltas/                       # delta records
├── db/
│   ├── defect_db.json            # defect database
│   └── test_patterns.json        # oracle patterns + failure modes + archetypes
├── schemas/
│   └── risk_graph_schema.json    # LPG schema (8 node types, 11 edge types)
├── prompts/
│   ├── baseline_batches/         # per-batch Agent prompts (auto-generated)
│   └── defect_backfill_batches/  # defect mapping Agent prompts
└── workflow_*.json               # Workflow bridge files
```

## Troubleshooting

- **"No baseline data"**: Run `/npu-risk-graph baseline` first
- **"No graph found"**: Run `/npu-risk-graph build`
- **Features are outdated**: Run `/npu-risk-graph features` → `/npu-risk-graph baseline` → `/npu-risk-graph build`
- **Graph is stale**: Run `/npu-risk-graph update` or rebuild
- **Unicode errors on Windows**: All scripts use `encoding='utf-8', errors='replace'`
- **NetworkX import errors**: Requires NetworkX 3.6+ (`pip install networkx`)
- **build_graph.py crashes**: Check all baseline JSONs exist in `baselines/latest/`
- **defect_db.json 更新后**: Run `/npu-risk-graph backfill` (full forced refresh of features_affected)
- **Few features have tests**: Some features lack dedicated test coverage — this is a test gap signal, not a data quality issue
- **Many tests have empty features_tested**: Interface/API tests correctly map to no specific feature
- **Unsure if docs cover all features**: Run `python .claude/skills/npu-risk-graph/scripts/detect_new_docs.py`
- **_meta counts look wrong**: Run `/npu-risk-graph features` — auto-repairs on next update
- **Test knowledge queries show 0 patterns**: Rebuild graph with `build_graph.py` — test_patterns.json is loaded at build time
- **Generated test not in KG**: `.sglang-risk/testcases/` is excluded from graph builds. Copy to `test/registered/` to include it, then re-run `baseline` → `build`
