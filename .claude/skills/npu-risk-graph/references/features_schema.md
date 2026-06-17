# features.json 字段定义

NPU 测试风险知识图谱的特性定义文件

## 顶层结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `generated_at` | string | ✓ | 生成时间戳 (ISO 8601) |
| `git_commit` | string | ✓ | 生成时的 git commit |
| `total_features` | int | ✓ | 特性总数（必须等于 `features` 数组长度） |
| `features` | array | ✓ | 特性列表 |

## 特性字段 (features[])

### 字段总览

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✓ | 特性名称 |
| `category` | string | ✓ | 特性分类（8 选 1） |
| `source_files` | array | ✓ | 关联源码文件路径列表 |
| `complexity` | int | ✓ | 复杂度 1-5 |
| `npu_participation` | string | ✓ | NPU 参与度（5 选 1） |
| `description` | string | ✓ | 特性描述 |
| `models_using` | array | ✓ | 使用该特性的模型列表 |
| `fingerprint` | string | ✓ | source_files 的 SHA256 指纹 |
| `last_modified` | string | ✓ | 最后修改时间 (ISO 8601) |
| `source` | string | ✓ | 数据来源标识 |
| `cross_cutting` | bool | — | 可选：横切基础设施标记 |

### name

**特性名称**。使用 sglang 官方特性名，不加 `npu_`/`ascend_` 前缀。参考 `docs_new/docs/advanced_features/` 中的特性名。

例外：`weight_prefetch` 为 Ascend 硬件独有的 CMO 特性，`sgl_model_gateway` 为独立 Rust 项目。

### category

**特性分类**，8 个有效值：

| 值 | 含义 | 典型文件 |
|----|------|---------|
| `attention` | 注意力机制 | `layers/attention/` |
| `inference_engine` | 推理执行 | `graph_runner/`, `moe/`, `sampler/`, `speculative/`, `multimodal/` |
| `memory` | 内存/缓存管理 | `mem_cache/`, `model_loader/` |
| `quantization` | 模型量化 | `layers/quantization/` |
| `distributed` | 多节点 disaggregation | `disaggregation/`, `checkpoint_engine/`, `model_loader/remote` |
| `parallelism` | 并行策略 | TP/DP/DPA/EP/PP/CP/EPLB |
| `interface` | 接口/协议层 | 平台无关 API |
| `platform` | 平台适配 | `hardware_backend/npu/` |

### source_files

**关联源码文件列表**。所有路径以 `python/sglang/srt/` 为根。一个文件可被多个 feature 共享。

| npu_participation | source_files 要求 | 说明 |
|-------------------|------------------|------|
| `strong` | >=1 个 `hardware_backend/npu/` 下的文件 | NPU 独立实现 |
| `medium` | >=1 个文件（含 NPU 代码分支） | NPU 代码在共享文件中 |
| `weak` | >=0（通常 1-5 个共享文件） | 仅有 is_npu() 分支或配置 |
| `platform_agnostic` | >=1 个共享文件 | 无 NPU 代码，但需追踪变更 |
| `not_supported` | >=1 个通用文件 | 追踪底层依赖间接影响 |

### complexity

**复杂度**，1-5 整数：

| 值 | 含义 |
|----|------|
| 1 | 纯配置/转发，无独立逻辑 |
| 2 | 薄适配层，少量代码 |
| 3 | 独立实现，有多文件/多后端 |
| 4 | 复杂实现，涉及多个子系统 |
| 5 | 高风险核心特性，多文件 + 多模型 |

### npu_participation

**NPU 参与度**，5 个有效值：

| 值 | 含义 | 判定标准 |
|----|------|---------|
| `strong` | 有独立 NPU 实现文件 | `source_files` 中含 >=1 个 `hardware_backend/npu/` 下文件 |
| `medium` | 有 NPU 代码但较薄 | 共享文件中有 `is_npu()` 分支，但无独立 NPU 文件 |
| `weak` | 仅配置/import | 仅有 `is_npu()` 类型转换或 `@torch.compile(disable=_is_npu)` |
| `platform_agnostic` | NPU 上可用，无需适配 | 无任何 NPU 代码信号，但平台可运行 |
| `not_supported` | NPU 不支持 | 依赖的 backend/kernel NPU 无等效替代 |

测试优先级：`strong` > `medium` > `weak` > `platform_agnostic`。`not_supported` 不参与排序，仅标记能力边界。

**判定流程：** 四个维度（NPU 实现文件 / `is_npu()` 分支 / NPU 配置 / NPU 测试）全部无命中 → 阅读文档内容判断平台可行性 → `platform_agnostic` 或 `not_supported`。

### cross_cutting

**横切特性标记**（可选）。`true` 时表示该特性是基础设施层，不参与 per-feature 风险排序。当前标记为 `cross_cutting` 的特性：`hardware_backend`、`scheduling`。

### description

**特性描述**。根据 `npu_participation` 不同，内容侧重不同：
- `strong` / `medium`：描述 NPU 侧实现、使用的 NPU 算子、集成的共享模块
- `weak` / `platform_agnostic`：描述特性功能，说明 NPU 为何能运行
- `not_supported`：描述特性功能，**必须写明 NPU 不支持的具体原因**（缺少哪个 backend/kernel/依赖）

### models_using

**使用该特性的模型列表**。取值约定：

| 值 | 适用场景 |
|----|---------|
| `["All models"]` | `platform_agnostic` 特性，与设备无关 |
| `["All models on NPU"]` | NPU 基础设施特性，所有模型 NPU 推理时默认启用 |
| `["Not applicable (NPU unsupported)"]` | `not_supported` 特性 |
| `["ModelA", "ModelB", ...]` | 特定模型列表 |

### failure_mode (可选)

**失败模式严重程度**，1-5 整数。优先于此字段，未标注时由风险模型从 `category` 推断。

| 值 | 含义 | 典型 category |
|----|------|-------------|
| 1 | 表面问题：日志噪音、微小性能退化 | — |
| 2 | 功能断裂：crash、error，用户立刻感知 | inference_engine, interface |
| 3 | 质量退化：精度下降但不静默 | attention, memory |
| 4 | 静默损坏：超时/hang/错误结果但不 crash | distributed, parallelism |
| 5 | 静默精度损失：输出看起来正常但数值错误 | quantization |

### fingerprint / last_modified / source

| 字段 | 说明 |
|------|------|
| `fingerprint` | 内容指纹 (SHA256 of sorted source_files)，用于增量更新时检测文件变更 |
| `last_modified` | 最后修改时间 (ISO 8601) |
| `last_modified_reason` | 可选：变更原因简述（如 "description: merged xxx doc" 或 "source_files: +2 files"），区分文件变更与描述更新 |
| `source` | 数据来源标识，当前为 `sglang_docs_graphify_v6`。增量更新时统一标准化为此值 |
