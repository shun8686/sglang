# features 分支工作流

完整的工作流文档。SKILL.md 中仅保留摘要，详细逻辑在此文件。
生成/更新前必须先读取 `references/features_schema.md`，所有输出必须符合格式约束。

## 设计原则

1. **特性名参考 sglang 官方名称** — 来自 `docs_new/docs/advanced_features/`。
2. **NPU 参与度决定纳入，不取决于代码量** — 5 行配置的 `hicache` 也是特性。
3. **平台无关特性也要纳入** — NPU 无需适配但需要测试覆盖，标记 `npu_participation: platform_agnostic`。
4. **NPU 不支持的特性也纳入** — 标记 `npu_participation: not_supported`，区分于 `platform_agnostic`。要求逐个文档阅读确认（不能仅凭"无 NPU 代码信号"推断）。
5. **单一 `source_files` 字段** — NPU 实现、共享文件 NPU 分支、平台无关实现、不支持特性的通用实现，全部在一个列表。
6. **代码特性也纳入** — Phase 2 社区检测中发现的 NPU 实现密集型模块，即使无对应 sglang 文档，也作为特性候选。

## 全量生成（from scratch）

从零开始生成 features.json，覆盖任何已有文件。

### Phase 1 — 从官方文档建立 sglang 特性清单

读取 `docs_new/docs/advanced_features/` 获取全部 sglang 官方特性（~31 个）。

### Phase 2 — Graphify NPU 边界子图 + 社区检测

确保 `graphify-out/graph.json` 存在。若不存在，提示用户运行 `/graphify` 生成，暂停等待确认。

graphify 扫描范围（NPU 相关目录 + 共享目录）：

```
/graphify python/sglang/srt/hardware_backend/npu/ python/sglang/srt/layers/ \
  python/sglang/srt/models/ python/sglang/srt/managers/ \
  python/sglang/srt/model_executor/ python/sglang/srt/distributed/ \
  python/sglang/srt/mem_cache/ python/sglang/srt/disaggregation/ \
  python/sglang/srt/speculative/ python/sglang/srt/compilation/ \
  python/sglang/srt/lora/ python/sglang/srt/multimodal/
```

从图中提取 NPU 边界子图：

```
Layer 0 (NPU seeds): source_file 含 "hardware_backend/npu"
Layer 1 (1-hop):     从 Layer 0 出发的 call/import/inherit 边
Layer 2 (2-hop):     若总数 < 5000 节点则纳入，否则截断
```

在子图上跑 Louvain 社区检测。按节点组成分类：
- Pure NPU (>= 90% Layer 0)
- Cross-boundary (30-90%)
- NPU-adjacent (< 30%)

### Phase 3 — 交叉验证 NPU 参与度

#### 3.1 文档特性匹配

对 Phase 1 的每个 sglang 特性，从四个维度检查 NPU 参与：

| 维度 | 方法 |
|------|------|
| NPU 实现文件 | 在 graphify 中搜索匹配特性关键词的 NPU 节点（关键词取特性名 + 核心文件名，如 attention_backend → attention/ascend_backend） |
| 共享代码 NPU 分支 | Grep 特性对应模块目录中的 `is_npu()` |
| NPU 专属配置 | 检查 `server_args.py` 中的 `--feature-backend ascend` 模式 |
| NPU 测试 | 检查 `test/registered/ascend/` 中是否有特性相关测试 |

按 `features_schema.md § npu_participation` 判定标准分类。判定流程：

```
四个维度全部无命中 → 阅读文档内容判断平台可行性
  → 平台可用   → platform_agnostic
  → NPU 不支持 → not_supported（包含在 features.json 中，标注不支持原因）
```

#### 3.2 代码特性发现（无文档特性）

Phase 2 社区检测中的 Pure NPU 集群中，**过滤掉已在 Phase 3.1 分配到特性的文件后**，剩余未分配文件按以下条件评估：

- >= 2 个 `hardware_backend/npu/` 下的 `.py` 文件（NPU 实现文件）
- 或 >= 5 处 `is_npu()` 分支在共享文件中（非 `models/`、非 `test/`）

不满足条件者合并到现有相关特性（如单个 `allocator_npu.py` → `kv_cache_pool`）。

命名取模块功能名（如 `cmo.py` 对应 `weight_prefetch`），不强行拼凑 sglang 文档名。

### Phase 4 — 提取模型覆盖

找到所有含 `is_npu()` 调用的模型文件：

```bash
grep -rn "is_npu()" python/sglang/srt/models/ --include="*.py" -l
```

按模型类型分类（MoE / DeepSeek-MLA / VL / 通用），为每个特性映射使用的模型：

1. 对每个特性，grep 其 `source_files` 中的 NPU 文件名在模型文件中的引用:
   ```bash
   grep -rn "<npu_file_basename>" python/sglang/srt/models/ --include="*.py" -l
   ```
2. 结合特性关键词（如 `moe`、`attention`、`quantization`）在模型文件中的引用交叉验证
3. 基础设施特性（所有模型都依赖，如 `memory`、`scheduling`）→ `"All models on NPU"`
4. 平台无关特性 → `"All models"`

取值约定见 `features_schema.md § models_using`。

### Phase 5 — 生成 features.json

所有字段必须符合 `references/features_schema.md` 定义。`category` 枚举：`attention` / `inference_engine` / `memory` / `quantization` / `distributed` / `parallelism` / `interface` / `platform`。

`git_commit` 记录当前 HEAD hash（`git rev-parse HEAD`），用于增量更新 diff 的基线。

`cross_cutting` 标记条件：特性是基础设施层，其风险通过其他 feature 间接表达。当前适用：`hardware_backend`（平台适配层）、`scheduling`（请求调度）。

`failure_mode` 不在此阶段设置——由人工标注。缺失时风险模型默认 3。增量更新时保留（Step 3）。

### Phase 6 — 校验

```bash
python .claude/skills/npu-risk-graph/scripts/validate_features.py
```

校验项：必填字段完整性、category/npu_participation 枚举值、complexity 1-5 范围、source_files/models_using 非空、fingerprint SHA256 hex 格式、路径前缀合法性、无重复特性名、failure_mode 若存在必须 1-5。

保存到 `.sglang-risk/baselines/latest/features.json`。

## 增量更新（incremental）

当 `features.json` 已存在时运行。保留人工编辑（description、complexity、models_using），更新结构性变化。

### Step 1 — 检测变化

```
A. 新增 NPU 源文件（两层检测 + 假阳性过滤）
   → Layer 1: 扫 hardware_backend/npu/ 中不在任何 feature source_files 里的 .py 文件
   → Layer 2: 扫共享目录中 is_npu() 分支文件
        grep -rn "is_npu()" python/sglang/srt/ --include="*.py" -l | grep -v "hardware_backend/npu"
   → 排除: models/*.py（由 models_using 覆盖）、graphify-out/、test/、__pycache__/
   → 假阳性过滤: 去除仅含 @torch.compile(disable=_is_npu) 或仅 import is_npu 无分支使用的文件
      (保留: 有 if is_npu(): / if _is_npu: / torch_npu.xxx() / sgl_kernel_npu.xxx() 调用的文件)

B. 新增/变化 sglang 特性文档
   → python .claude/skills/npu-risk-graph/scripts/detect_new_docs.py  # auto-classify docs → covered/skip/subsumed/candidate; then diff docs_new/docs/advanced_features/ 与已知文档列表
   → 文档重命名/拆分检测: 若同时出现旧文档移除 + 新文档新增，阅读两者内容确认是否覆盖同一功能
     - 确认重命名 → 更新 features.json 中 name，保留其他字段
     - 确认拆分 → 旧 feature 可能被多个新 feature 替代，逐一评估
   → 对每个未覆盖的文档：
       a. 先检查 NPU 代码显式参与（is_npu(), torch_npu, SGLANG_NPU*）→ strong/medium/weak
       b. 无 NPU 代码时，阅读文档内容→判断平台可行性
       c. 平台可用→platform_agnostic；不可用→not_supported
       d. 跳过以下文档（非独立特性）：
          - overview（索引页）
          - 调优/最佳实践指南（如 hyperparameter_tuning）
          - 对比指南（如 dp_dpa_smg_guide）
          - 已融入其他特性的文档（如 vlm_query 已并入 multimodal）
          不确定时默认纳入（宁可多不可漏）

C. 新增模型文件
   → 使用上一次更新时的 commit hash 作为基线（记录在 features.json 的 `git_commit` 字段）:
        git diff <last_git_commit> HEAD --name-only -- python/sglang/srt/models/ | grep "\.py$"
   → 检查其中含 is_npu() 的文件
   → 按新模型的架构类型（MoE/MLA/VL/通用）匹配相关特性，追加到 models_using

D. 删除/移动的文件
   → 逐一检查 features.json 中 source_files 是否仍在磁盘上
   → 文件移动：更新路径；文件删除：移除引用

E. 新社区集群
   → 仅在大版本更新或用户明确要求时重跑 Phase 2（因 graphify 成本高）
   → 日常增量更新可跳过此步，依赖 Step A 的文件级检测覆盖
   → 若重跑：新群集如符合 Phase 3.2 条件，作为候选特性
```

### Step 2 — 分类处理

| 变化 | 操作 |
|------|------|
| 新 NPU 文件 → 现有特性 | 启发式匹配（文件名 + 父目录关键词 vs 现有特性名 + source_files 归属），匹配成功则加入 `source_files`。若无明确匹配，人工判断归属或标记待定 |
| 新 NPU 文件 → 无匹配特性 | 候选：按 Phase 3.2 代码特性条件评估，满足则提出新特性 |
| 新 sglang 文档 → NPU 参与 | 作为新特性加入（跑 Phase 3-4） |
| 新 sglang 文档 → NPU 不参与 | 按 Phase 3 判定 → platform_agnostic 或 not_supported |
| 新模型文件 | 更新相关特性 `models_using` |
| 文件删除/移动 | 更新 `source_files` 路径，无替代时标记人工审核 |
| 新社区集群 | 提取群集核心文件 → 按文件名聚类 → 与现有特性交叉对照 → 符合条件提出新特性或合并建议 |

### Step 3 — 保留人工字段

**增量更新永不覆盖：**
- `description` — 保留用户编写文本
- `complexity` — 保留人工调整。仅在 source_files 变化 >= +3 时，自动 +1（上限 5）
- `models_using` — 保留，除非检测到新增/移除模型（Step 1C）
- `failure_mode` — 人工标注，永不覆盖

**始终更新：**
- `source_files` — 反映实际文件状态
- `fingerprint` — 任何变化时重新生成
- `last_modified` — 更新时间戳
- `npu_participation` — 新增/移除 `hardware_backend/npu/` 下的文件时重新计算（共享文件的 is_npu() 分支变化不影响参与度级别）

### Step 4 — Diff 报告

```
## Incremental Update

### New Features (N)
- <feature>: <reason> (npu_participation=<level>)

### Updated Features (N)
- <feature>: +2 source_files, -1 source_file (moved)

### Removed Features (N)
- <feature>: source files deleted, no replacement

### Unchanged (N)
```

询问："Apply these changes? (Y/N / select)"

### Step 5 — 应用并保存

写入更新后的 features.json，更新 `git_commit`（`git rev-parse HEAD`）、`generated_at`、`_meta.changes`，备份到 `features.json.bak.{ISO timestamp}`。

### Step 6 — 校验

运行 Phase 6 的校验脚本。确认通过。

### 兜底

`features.json` 不存在 → 回退到**全量生成**模式（Phase 1-6）。
