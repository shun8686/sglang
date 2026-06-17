# npu-defect-mine

NPU defect database (`defect_db.json`) lifecycle management: PR-first extraction from origin/main, Agent semantic classification, data integrity enforcement, and periodic audit.

## Data model

**Agent-produced fields** (5 fields — audited for quality):
`category` — `crash` | `compatibility` | `precision_loss` | `compile_error` | `perf_regression`
`severity` — `10` (Critical: crash/OOM/hang) | `3` (Major: function error/precision loss) | `1` (Minor: perf/compile/boundary)
`root_cause` — free-form text, one sentence, specific (not generic like "cann_api")
`rationale` — Agent's reasoning for the classification
`confidence` — 0.0-1.0 (≥0.9 high, ≥0.7 medium, <0.7 needs_review)

**Script-produced fields** (filled automatically from GitHub):
`bug_id`, `pr_title`, `pr_body`, `files_fixed`, `source`, `pr_number`, `commit_sha`, `date_fixed`

**Audit tracking**:
`audit_count` — number of times this defect has been audited (script increments after each audit pass)

**Flag**:
`needs_review` — set by integrity check (data issues) or Agent (low confidence)

**Extraction filters** — skipped at source:
- Revert PRs (`title.startswith("Revert")`)
- Pure documentation (all files under `docs/` or `.md/.rst/.txt`)
- Non-product code (no files under `python/`, `sgl-kernel/`, `sgl-router/`)
- **Merge-status**: Only filtered in `init`/`update` modes. `analyze-pr --pr <N>` skips this check, allowing unmerged/OPEN PRs.

**Unmerged PR tracking**:
- When a PR is extracted via `--pr` before merge, the defect record gets `_unmerged: true`
- Downstream queries or reports can optionally filter on this flag to exclude unmerged entries from production dashboards

**False-positive prevention** — regex must use word boundaries (`\b`):
- `\bnpu\b` NOT `npu` — prevents matching `input` (i-n-p-u-t), `nonpublic`, `unplug`
- `\bcann\b` NOT `cann` — prevents matching `cannot`, `scanning`, `scanner`
- `\bascend\b` NOT `ascend` — prevents matching within boilerplate launch commands
- Post-extraction `is_genuinely_npu_defect()` validator checks: label **or** title tag **or** file path **or** substantive body discussion (strips Accuracy/Speed Tests boilerplate before check)
- Known false-positive source: launch command `--device npu --attention-backend ascend` in PR "Accuracy Tests" sections is boilerplate, not substantive NPU discussion

## Trigger

```
/npu-defect-mine init              # 首次建库
/npu-defect-mine update            # 增量更新
/npu-defect-mine analyze-pr <num>  # 分析指定 PR（不限制合入状态）
/npu-defect-mine maintain          # 质量维护（日常抽样 ~20 条）
/npu-defect-mine maintain --full   # 质量维护（全量审计）
/npu-defect-mine status            # 查看状态
```

Or natural language: "初始化缺陷库", "更新缺陷数据", "分析 PR #23685", "维护缺陷库", "检查缺陷库状态".

---

## Architecture

```
                    ┌──────────────┐
                    │ defect_db.json │ ← 核心资产
                    └──────┬───────┘
           ┌───────────────┼───────────────┬──────────┐
           │               │               │          │
        (init)         (update)        (maintain)  (status)
      首次建库         增量追加         质量修复     只读查询
```

Claude drives each sub-workflow: Python scripts for data preparation and merging, Workflow tool for Agent classification.

---

## Shared utilities

### PR body enrichment

When defects have `pr_number` but lack `pr_body`, fetch from GitHub API. Needed when defects were imported without full PR data (e.g., legacy entries, heuristic-only runs). PR-first extraction already embeds PR data at extraction time.

```bash
python -c "
import subprocess, json, time, sys
sys.path.insert(0, '.claude/skills/npu-defect-mine/scripts')
from common import load_json, save_json, DB_DIR

db = load_json(DB_DIR / 'defect_db.json')
targets = [d for d in db['defects']
    if d.get('pr_number') and (not d.get('pr_body') or len(str(d.get('pr_body','')))<100)]
if not targets:
    print('All PR defects already have body text — skipping')
    exit()

print(f'Fetching PR body for {len(targets)} defects...')
for i, d in enumerate(targets):
    if i>0 and i%8==0: time.sleep(2)  # rate limit
    try:
        r = subprocess.run(['gh','api',f'repos/sgl-project/sglang/pulls/{d[\"pr_number\"]}',
            '--jq','{body,title,labels:[.labels[].name]}'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
        if r.returncode==0 and r.stdout.strip():
            pr = json.loads(r.stdout)
            d['pr_body'] = (pr.get('body') or '')[:3000]
            d['pr_title'] = pr.get('title') or d.get('title','')
            d['pr_labels'] = pr.get('labels',[])
    except Exception as e: print(f'  Error: {e}')
    if (i+1)%30==0: print(f'  {i+1}/{len(targets)}')

db['_meta']['pr_body_enriched_at'] = __import__('datetime').datetime.now().isoformat()
save_json(DB_DIR/'defect_db.json', db)
enriched = sum(1 for d in targets if d.get('pr_body') and len(str(d['pr_body']))>=100)
print(f'Enriched: {enriched}/{len(targets)}')
"
```

### Agent Audit — 分类质量审计

抽样检查 Agent 分类是否正确。只审计 5 个 Agent 字段（category/severity/root_cause/rationale/confidence），脚本字段不审。

```bash
# Step 1: 抽样或全量提取
python .claude/skills/npu-defect-mine/scripts/audit_sample.py --sample-size 3 --split 15
python .claude/skills/npu-defect-mine/scripts/audit_sample.py --all --split 15
→ db/audit_batches/ (每个 batch 一个 JSON 文件)

# Step 2: 并行审计 — 每个 batch 文件 spawn 一个 Audit agent
# Prompt 规定见 audit_prompt.txt。关键要求：
#   - bug_id 必须原样抄写，不得改动（曾出现 2025→2026 抄错）
#   - action=accept 时 changes 为空，只写 evidence
#   - action=change 时 changes 包含所有修正字段
# Agent 读 batch 文件 → 挑战分类 → 写 audit_batch_NN_result.json
# 输出格式: {summary: {total, accepted, changed, deleted}, findings: [...]}

# Step 3: 主 Agent 分组确认
# 根据变更数量选择策略：
#   - <10 条 change: 逐条确认
#   - ≥10 条 change: 按类型分组
#       A 组 (category 变更): 逐条确认 ← 影响最大
#       B 组 (root_cause/rationale 重写): 批量采纳
#       C 组 (severity/confidence 微调): 批量采纳

# Step 3.5: 争议解决
# 对 A 组中主 Agent 存疑的 category 变更：
#   1. spawn 1 个独立 Audit agent（不告知争议内容，只给 PR body + category 定义）
#   2. 比较：原始分类 vs 第一次审计建议 vs 独立审计意见
#   3. 主 Agent 做最终判决（不做投票，看论据质量）

# Step 4: 主 Agent 将确认后的 findings 写入 .sglang-risk/db/audit_report.json
# 格式与 audit agent 产出相同: {findings: [{bug_id, action, evidence, changes, confidence}]}
# apply_agent_results.py 自动完成 findings → classification 转换 + 字段补全

# Step 5: 应用确认的修正 + 审计计数
python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py --input .sglang-risk/db/audit_report.json
python .claude/skills/npu-defect-mine/scripts/audit_finalize.py
```

抽样策略（按 audit_count ASC 排序，审计最少的优先）：
- 所有 needs_review 条目（最高优先级，不排入计数限制）
- 每类 audit_count 最低的 N 条（低 confidence 优先）+ audit_count 最低的 N 条（高 confidence 优先）
- `compatibility` 双倍采样
- `--all` 全量审计，建议周期：首次 Agent 分类后 + 每月

### Agent classification

The shared step used by `init`, `update`, and `maintain`. Claude reads `pending_agent_batch.json`, calls the `Workflow` tool, saves the return value, and runs `apply_agent_results.py`:

1. Read `.sglang-risk/db/pending_agent_batch.json`, extract `pr_batches` and `commit_batch`
2. Call `Workflow` tool with inline script and `args`
3. Workflow internally:
   - `pipeline(pr_batches, classify, verify)` — 15 defects/batch, adversarial verify for confidence <0.7
   - `agent()` for commit batch — forced `needs_review: true`, confidence ≤0.7
4. Write Workflow return value to `.sglang-risk/db/agent_results.json`
5. Run `python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py`

Workflow prompt must include category definitions, severity scale, and root_cause guidelines (free-form, specific).

### Status check

```bash
python .claude/skills/npu-defect-mine/scripts/status_check.py
```

---

## Sub-workflow 1: `init` — 首次建库

When `defect_db.json` doesn't exist or needs full rebuild.

**Architecture: PR-first extraction.** Search merged PRs with NPU labels + keywords → `is_genuinely_npu_defect()` validates NPU relevance (label/title/files/body) → skip reverts + doc-only → fetch full PR body/files → Agent classifies. 100% PR body coverage. Commit-only fallback via `gh search prs` per SHA. NPU regex uses word boundaries (`\bnpu\b`) to prevent substring false positives like `input`.

```
Step A: PR 提取
  python .claude/skills/npu-defect-mine/scripts/extract_prs.py --since 2025-01-01
  → pr_defects.json（每个缺陷已含 body/labels/reviews/files/merge_commit）

Step B: 写入 defect_db + 准备 Agent 批次
  python .claude/skills/npu-defect-mine/scripts/seed_defect_db.py --from-prs --agent
  → defect_db.json + pending_agent_batch.json

Step C: Agent 语义分类
  Run "Agent classification" (see shared utilities)

Step D: 合并
  python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py

Step E: Agent Audit（建议首次分类后全量审计）
  python .claude/skills/npu-defect-mine/scripts/audit_sample.py --all --split 15
  → 并行 spawn Agent → 主 Agent 分组确认 → apply + finalize

Step F (optional): Run consistency check.
```

6 steps (A-F). PR body embedded at extraction — no separate enrichment or batch formatting.

---

## Sub-workflow 2: `update` — 增量更新

Daily use: scan for newly merged PRs with NPU labels since last scan. PR-first only（commit fallback 已移除）.

```
Step A: 扫描 PR + 写入 DB + 准备批次
  python .claude/skills/npu-defect-mine/scripts/append_daily.py --agent
  → gh search prs merged:>={last_scan_at} + label:npu/ascend
  → NPU label/title/files/body 四维校验 + doc/non-product 过滤
  → pr_number/commit_sha 去重后写入 defect_db.json
  → 生成 pending_agent_batch.json

Step B: Agent 分类
Step C: 合并
Step D (optional): 一致性检查
```

0-5 new defects per run. Without `--agent`, defects appended with `category=unknown`.

---
## Sub-workflow 2b: `analyze-pr` — 分析指定 PR（支持未合入 PR）

When you need to analyze a specific PR regardless of its merge status — for example, a PR that is still OPEN but you want to pre-classify it into the defect database.

```
Step A: 提取 PR 数据 + 写入 DB
  python .claude/skills/npu-defect-mine/scripts/append_daily.py --pr <number> --agent
  → gh api repos/sgl-project/sglang/pulls/<number>
  → **No merge-status filter** — works for OPEN/MERGED/CLOSED PRs
  → NPU relevance validation (labels/title/files/body)
  → doc-only + non-product filtering (still applied)
  → revert detection (still applied)
  → Deduplicates against existing DB entries
  → Writes to defect_db.json with _unmerged flag
  → Prepares pending_agent_batch.json

Step B: Agent 语义分类
  Run "Agent classification" (see shared utilities)

Step C: 合并
  python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py
```

Key differences from `update`:
- **No merge requirement**: PR can be OPEN, CLOSED (unmerged), or MERGED
- **No NPU signal filter**: User explicitly chose the PR; NPU labels/title/files are logged but not required
- **No date filter**: Fetches PR by exact number
- **`_unmerged` flag**: Defect record includes `_unmerged: true` when `merged_at` is null, so downstream queries can optionally filter
- **Dedup**: Skips if PR number already exists in DB (can be forced by removing the existing entry first)

Also works via `extract_prs.py` for standalone extraction without DB append:
```
python .claude/skills/npu-defect-mine/scripts/extract_prs.py --pr 23685
→ pr_defects.json (single entry, can be manually inspected before seeding)
```

---

## Sub-workflow 3: `maintain` — 质量维护

When `status` shows `needs_review > 0` or consistency violations exist — or periodically.

```
Step A: 数据完整性检查（快速，脚本）
  python .claude/skills/npu-defect-mine/scripts/consistency_check.py
  → 必填字段、格式校验、唯一性、枚举值、值域、分类状态、孤儿 commit

Step B: Agent Audit（核心，必走）
  日常: python .claude/skills/npu-defect-mine/scripts/audit_sample.py --sample-size 3 --split 15（~30 条）
        = 5 categories × 2 confidence bands (low+high) × 3 samples
        + compatibility 双倍（+3）+ needs_review（不计入限额）
  --full: python .claude/skills/npu-defect-mine/scripts/audit_sample.py --all --split 15（全量）
  → 并行 spawn Audit Agent（7-8 个）→ 各写 audit_batch_NN_result.json
  → 主 Agent 汇总分组确认:
      A 组 (category 变更): 逐条确认 ← 影响最大
      B 组 (root_cause/rationale 重写): 批量采纳
      C 组 (severity/confidence 微调): 批量采纳
      D 组 (is_not_npu=true): 主 Agent 判断升级为 delete 还是保留 needs_review
  → 主 Agent 将确认的 findings 写入 .sglang-risk/db/audit_report.json

Step C: 应用修正
  python .claude/skills/npu-defect-mine/scripts/apply_agent_results.py --input .sglang-risk/db/audit_report.json
  （自动: accept→保留分类+清理needs_review, change→合并修正,
          delete→跳过, is_not_npu→转为needs_review标记）

Step D: 删除非 NPU 缺陷
  主 Agent 从 defect_db.json 中移除 action=delete 的条目，
  更新 _meta.last_audit_deletions

Step E: 审计计数 + 重跑完整性
  python .claude/skills/npu-defect-mine/scripts/audit_finalize.py
  python .claude/skills/npu-defect-mine/scripts/consistency_check.py
  ├─ 0 issues → clean，结束
  └─ 有 issues → 修复后重跑 Step E
```

Note: `apply_agent_results.py` 只处理 classification 修正，不删除条目。
删除由主 Agent 在 Step D 中手动执行。

Maintain 完成后清理：`.sglang-risk/db/audit_batches/` 下的 batch JSON 和 result JSON 可删除，
`.sglang-risk/db/audit_report.json` 建议保留作为审计记录。

---

## Sub-workflow 4: `status` — 查看状态

Run the status check script from shared utilities. Read-only.

---

## Key files

| File | Role |
|------|------|
| `.sglang-risk/db/defect_db.json` | 核心缺陷数据库 |
| `.claude/skills/npu-defect-mine/scripts/extract_prs.py` | init: PR-first 提取 |
| `.claude/skills/npu-defect-mine/scripts/seed_defect_db.py` | init: 创建 seed DB |
| `.claude/skills/npu-defect-mine/scripts/classify_pr.py` | init/maintain: PR 缺陷 → Agent 批次 |
| `.claude/skills/npu-defect-mine/scripts/classify_commit.py` | init/maintain: commit-only → Agent 批次 |
| `.claude/skills/npu-defect-mine/scripts/append_daily.py` | update: PR-first 增量 |
| `.claude/skills/npu-defect-mine/scripts/apply_agent_results.py` | 所有流程: schema 验证 + 合并 |
| `.claude/skills/npu-defect-mine/scripts/consistency_check.py` | maintain: 数据完整性检查（格式/唯一性/值域/孤儿）|
| `.claude/skills/npu-defect-mine/scripts/audit_sample.py` | audit: 抽取审计样本（抽样/全量，按 audit_count 排序）|
| `.claude/skills/npu-defect-mine/scripts/audit_finalize.py` | audit: 审计完成后 audit_count +1 |
| `.claude/skills/npu-defect-mine/scripts/status_check.py` | status: 数据库状态快照 |
| `.claude/skills/npu-defect-mine/scripts/prompts.py` | Agent prompt 模板 |
| `.claude/skills/npu-defect-mine/scripts/audit_prompt.txt` | audit: Agent 审计 prompt 规范 |
| `.claude/skills/npu-defect-mine/scripts/defect_db_schema.json` | JSON Schema 验证规范 |

## Fallback

Without Claude Code Workflow, use heuristic keyword matching (confidence 0.4-0.6):
```bash
python .claude/skills/npu-defect-mine/scripts/seed_defect_db.py          # no --agent
python .claude/skills/npu-defect-mine/scripts/append_daily.py            # no --agent (PR-first, category=unknown)
```
