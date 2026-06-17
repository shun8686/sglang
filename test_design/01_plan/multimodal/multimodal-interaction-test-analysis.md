# 多模态功能交互测试分析报告

> 生成日期: 2026-06-10
> 审阅日期: 2026-06-11
> 数据来源: graphify 知识图谱 (143K nodes, 279K edges, 11K communities) + features.json (39 features)
> 分析方法: 图社区聚类 + 跨特性边统计 + 源代码审计（对抗验证）
> 审阅: 3 个子 agent 并行对抗验证 + 1 轮人工正确性审查

---

## 1. 概述

### 1.1 分析对象

`multimodal` 特性（`test_design/baseline/features.json` 定义）：

- **源文件**: 15 个
- **图节点**: 846 个（分布在 53 个社区）
- **NPU 参与度**: strong
- **支持模型**: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM-4V/4.5V/4.6V, InternVL, IDEFICS2, MiniCPMV-ViT, Dots-VLM-ViT, PaddleOCR-VL（注：Ernie4.5-VL processor 文件存在但未列入 models_using）

### 1.2 交互拓扑（与哪些特性存在耦合）

共有 27 个特性与 multimodal 存在图级或代码级交互（含 §4 覆盖矩阵中引用但原始边统计遗漏的 5 个特性），按连接强度排序：

| 排名 | 特性 | 边数 | 交互本质 |
|------|---------|------|---------|
| 1 | attention_backend | 95 | 图像编码走哪个注意力后端 |
| 2 | hardware_backend | 69 | NPU 平台适配（线性层、归一化、CMO 预取） |
| 3 | tensor_parallelism | 53 | 模型分片时 ViT 与 LLM 的并行策略 |
| 4 | graph_compilation | 36 | ViT 的图捕获与回放 |
| 5 | speculative_decoding | 23 | 投机解码的草稿 token 在多模态输入下是否有效 |
| 6 | hicache | 18 | 分层 KV 缓存如何对待图像 token |
| 7 | scheduling | 16 | 调度器如何混合纯文本和图像请求 |
| 8 | chunked_prefill | 16 | 分块预填充是否破坏图像特征缓冲区 |
| 9 | pd_disaggregation | 15 | 预填充-解码分离时图像嵌入如何传输 |
| 10 | moe | 10 | MoE 专家路由在图像 token 下是否有偏差 |
| 11 | dp_for_multi_modal_encoder | 8 | ViT 数据并行编码（直接多模态特性，§6.1 补充） |
| 12 | dp_attention | 6 | 数据并行注意力在多模态下的行为（§6.1 补充） |
| 13 | lora | 5 | LoRA 适配器对 ViT/LLM 的影响范围（§6.1 补充） |
| 14 | structured_outputs | 3 | JSON Schema 约束对图像 token 输出的影响（§6.1 补充） |
| 15 | tool_parser | 2 | 工具调用中 image 参数的传递（§6.1 补充） |
| 16~27 | deterministic_inference, quantization, quantized_kv_cache, offloading, hisparse, kv_cache_pool, separate_reasoning, context_parallelism, data_parallelism, weight_prefetch, eplb, expert_parallelism | 1~7 | 间接交互或特定标志触发 |

> **审阅注 (§6.1)**: dp_for_multi_modal_encoder (ViT DP 编码)、dp_attention、lora、structured_outputs、tool_parser 在原始图边统计中因图分区边界被遗漏，但 §4 覆盖矩阵中已引用，此处补充。dp_for_multi_modal_encoder 是专门针对多模态编码器的数据并行特性，与 multimodal 直接相关，建议后续增加专项测试用例。

### 1.3 交互拓扑图

```
                         ┌─────────────────┐
                         │   用户请求       │
                         │ (图像 + 文本)    │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌────▼────┐ ┌─────▼──────┐
              │ ViT 编码器 │ │调度器   │ │ Tokenizer  │
              │ (attention │ │(sched-  │ │(server_    │
              │  _backend) │ │ uling)  │ │ args)      │
              └─────┬─────┘ └────┬────┘ └─────┬──────┘
                    │            │             │
         ┌──────────┼────────────┼─────────────┼──────────┐
         │          │            │             │          │
    ┌────▼────┐ ┌───▼────┐ ┌────▼────┐ ┌──────▼────┐ ┌──▼──────┐
    │图编译   │ │TP/DP   │ │分块预填 │ │ PD 分离   │ │投机解码 │
    │(graph   │ │(tensor │ │(chunked │ │(pd_disag │ │(specula │
    │_compil) │ │_paral) │ │_prefill)│ │gregation)│ │tive)    │
    └────┬────┘ └───┬────┘ └────┬────┘ └──────┬────┘ └────┬────┘
         │          │           │             │           │
         └──────────┼───────────┼─────────────┼───────────┘
                    │           │             │
              ┌─────▼────┐ ┌───▼──────┐ ┌────▼──────────┐
              │ LLM 解码 │ │ KV 缓存  │ │ 输出/日志      │
              │ (moe,    │ │ (hicache,│ │ (observability,│
              │  quant)  │ │  kv_pool)│ │  separate_     │
              │          │ │          │ │  reasoning)    │
              └──────────┘ └──────────┘ └───────────────┘
```

---

## 2. E2E 测试用例清单

> 以下用例从业务/用户视角描述，不涉及实现细节。每条用例的"关联特性"列出该场景覆盖到的特性交互，实现时可参考对应代码路径。
>
> ### 能力标签说明
>
> 用例通过"所需能力"描述前置条件，而非绑定具体模型。§3 模型矩阵定义每个模型具备哪些能力。
>
> | 能力标签 | 含义 | 代表模型 |
> |----------|------|----------|
> | `vlm` | 任意多模态模型（ViT 编码器 + 文本解码器） | Qwen3-VL, Qwen2.5-VL, ... |
> | `vlm-moe` | MoE 多模态模型 | Qwen3-VL, Qwen3.5-MoE |
> | `vlm-lora` | 支持 LoRA 适配器的多模态模型 | Qwen3-VL |
> | `vlm-reasoning` | 支持 reasoning parser 的多模态模型 | GLM-4.6V, Qwen3-VL |
> | `vlm-mtp` | 支持 MTP/EAGLE 投机解码的多模态模型 | Qwen3-VL（待确认），DeepSeek-V2（仅文本） |
> | `vlm-gdn` | 使用 GDN 线性注意力文本骨干的多模态模型 | Qwen3.5, Qwen3.5-MoE |
> | `vlm-pp` | 支持流水线并行的多模态模型 | Qwen3-VL |

### 2.1 P0 用例（核心业务流程，每次变更必须通过）

#### P0-001: 单图 + 文本 → 描述图片内容

| 项目 | 内容 |
|------|------|
| **场景** | 用户上传一张图片，要求模型描述图片内容 |
| **输入** | 1 张 PNG 图片 + 文本 "请描述这张图片" |
| **期望结果** | 模型输出与图片内容相关的文字描述，无乱码、无截断 |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. TTFT < 阈值 |
| **关联特性** | attention_backend (图像编码)，scheduling (请求调度) |
| **所需能力** | `vlm` |
| **用例价值** | **10/10** — 最基础的多模态 smoke test。覆盖 ViT 编码→LLM 解码全链路，是所有多模态功能的前置条件。任何代码变更后第一个应执行的测试。失败意味着多模态功能完全不可用。 |
| **价值描述** | 核心路径必经点；失败面极大（所有多模态用户受影响）；调试成本低（输入简单、根因定位快）。**必须每次 CI 通过。** |

#### P0-002: 同图多次请求 → 缓存命中

| 项目 | 内容 |
|------|------|
| **场景** | 同一张图片被两次请求使用（如图片是系统提示的一部分），第二次应更快 |
| **输入** | 请求1: 图片A + "描述图片" / 请求2: 同一张图片A + "描述图片中的颜色" |
| **期望结果** | 请求2 的 TTFT 显著低于请求1（前缀缓存命中） |
| **验证点** | 1. 两次输出均正确 2. 请求2 TTFT < 请求1 TTFT * 0.6 |
| **关联特性** | Radix Cache (前缀缓存)，kv_cache_pool (缓存池) |
| **所需能力** | `vlm` |
| **用例价值** | **8/10** — 验证多模态场景下 Radix Cache 对图像 token 前缀的缓存能力。缓存命中直接影响生产环境吞吐和延迟，但不命中不会导致功能错误。 |
| **价值描述** | 性能关键路径（生产环境多轮对话依赖前缀缓存）；失败面中等（仅影响性能不阻塞功能）；验证阈值 0.6× 可能过严——若 ViT 编码时间占比高则缓存收益有限，建议分 ViT 时间占比设不同阈值或仅验证 "TTFT₂ < TTFT₁"。 |

#### P0-003: 并发纯文本 + 图像请求 → 隔离性

| 项目 | 内容 |
|------|------|
| **场景** | 同时发送 10 个纯文本请求和 5 个图像请求，验证两类请求互不干扰 |
| **输入** | 并发发送：10 个纯文本 "hello" 请求 + 5 个图片 "描述图片" 请求 |
| **期望结果** | 所有请求均正确返回，纯文本输出和图像输出均正确 |
| **验证点** | 1. 所有 15 个请求成功 2. 纯文本输出不受图像 token 污染 3. 无崩溃 |
| **关联特性** | scheduling, attention_backend |
| **所需能力** | `vlm` |
| **用例价值** | **9/10** — 验证调度器在混合负载下的 batch 隔离性。图像 token 数量远大于文本 token，跨请求 KV cache 污染是静默正确性杀手。此类 bug 线上极难排查。 |
| **价值描述** | 高隐蔽性故障模式（token 污染不会崩溃但输出错误）；线上影响面大（混合负载是常态）；调试成本极高（需对比单请求/并发请求输出差异）。唯一缺点是并发数偏小（10+5），建议增加一轮高并发 (50+50) 作为 P1 补充。 |

#### P0-004: 多图 → 比较两张图片

| 项目 | 内容 |
|------|------|
| **场景** | 用户上传两张图片，要求比较它们的异同 |
| **输入** | 2 张不同的 PNG 图片 + 文本 "请比较这两张图片" |
| **期望结果** | 模型输出内容涉及两张图片的特征比较 |
| **验证点** | 1. 返回码 200 2. 输出同时涉及两张图片 3. 无 OOM |
| **关联特性** | attention_backend (多模态 token 填充), scheduling |
| **所需能力** | `vlm` |
| **用例价值** | **8/10** — 多图场景是多模态模型的真实高频用例（图片对比、多页文档理解）。覆盖 image token 在 batch 中多次拼接的正确性，以及 KV cache 在多图序列下的内存管理。 |
| **价值描述** | 用户高频场景（对比类 prompt 占比高）；验证点 "输出同时涉及两张图片" 偏主观——建议改为结构化验证（如要求模型列出两张图片的差异列表并检查列表项数≥1）。 |

#### P0-005: 变尺寸图片 → 不同分辨率处理

| 项目 | 内容 |
|------|------|
| **场景** | 依次发送 3 张不同尺寸的图片（小 128x128、中 640x480、大 1920x1080），验证模型均能正确处理 |
| **输入** | 3 次请求，每次 1 张不同尺寸图片 + "描述图片" |
| **期望结果** | 所有尺寸图片均正确返回描述 |
| **验证点** | 1. 全部成功 2. 大分辨率不导致 OOM 3. 小分辨率不产生退化输出 |
| **关联特性** | graph_compilation (图重捕获)，attention_backend |
| **所需能力** | `vlm` |
| **用例价值** | **9/10** — 不同分辨率触发不同的 ViT 图编译路径（patch 数量变化导致计算图 shape 变化），是 NPU Graph 最常见的崩溃来源。大分辨率还验证 KV cache 内存分配的弹性。 |
| **价值描述** | NPU 高危场景（图重捕获在多 shape 下频繁触发）；线上真实存在（用户上传图片分辨率不可控）；小分辨率退化是隐蔽的正确性问题（patch 过少导致图像理解失败但不会报错）。建议增加一轮 `max_pixels` 边界值（如 1×1 和 4096×4096）作为补充。 |

#### P0-006: 长文本 + 图片 → 分块预填充

| 项目 | 内容 |
|------|------|
| **场景** | 用户在图片前附加一段长系统提示（触发分块预填充），验证图像特征不被截断 |
| **输入** | 3K token 前缀文本 + 1 张图片 + "总结以上" |
| **期望结果** | 模型输出正确总结文本和图片内容 |
| **验证点** | 1. 输出涉及图片内容（图像特征未被忽略）2. 输出涉及前缀文本 3. 无截断 |
| **关联特性** | chunked_prefill (分块预填充跨块状态), scheduling |
| **所需能力** | `vlm` |
| **用例价值** | **8/10** — 分块预填充将长序列切分为多个 chunk 分别计算 attention。图像 token 若恰好落在 chunk 边界，其 KV cache 可能丢失或错位。这是 chunked_prefill 与多模态的经典交互 bug。 |
| **价值描述** | 隐蔽性高（只在特定 token 长度触发）；chunked_prefill 是生产默认开启的特性；3K token 阈值可能不够精确——建议用参数化方式测试多个前缀长度（刚好 chunk_size、chunk_size+1、2×chunk_size），确保跨 chunk 边界的图像 token 安全。 |

#### P0-007: 图编译开启下的多模态推理

| 项目 | 内容 |
|------|------|
| **场景** | 启用图编译（不传 `--disable-cuda-graph`），发送图像请求，验证推理正确 |
| **输入** | 图片 + "描述图片" |
| **期望结果** | 输出正确，不崩溃 |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. **不使用 `--disable-cuda-graph`** |
| **关联特性** | graph_compilation (NPU ViT graph runner) |
| **所需能力** | `vlm` |
| **用例价值** | **9/10** — NPU 上 ViT 图编译是最复杂的代码路径之一（vit_npu_graph_runner.py）。图捕获失败或回放 shape 不匹配会导致静默错误输出而非崩溃，排查困难。 |
| **价值描述** | NPU 独有高风险路径（GPU 使用 CUDA Graph，NPU 使用独立 vit_npu_graph_runner）；图回放 shape 不匹配时可能输出语义错误的描述（幻觉）而非报错——需要对比 `--disable-cuda-graph` 的输出来验证一致性。建议验证点增加 "输出与 `--disable-cuda-graph` 时语义一致"。 |

#### P0-008: LoRA 适配器 + 多模态模型 → 仅作用于 LLM

| 项目 | 内容 |
|------|------|
| **场景** | 加载 LoRA 适配器后，发送图像请求，验证 LoRA 不影响图像理解能力 |
| **输入** | 带 LoRA 适配器的模型 + 图片 + "描述图片" |
| **期望结果** | 输出受 LoRA 风格影响但内容仍基于图片（LoRA 未破坏 ViT 编码器） |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. 风格与 baseline 不同（LoRA 生效）4. 不崩溃 |
| **关联特性** | LoRA |
| **所需能力** | `vlm-lora` |
| **用例价值** | **7/10** — LoRA 与多模态的组合在生产中日益常见（微调 VLM 的文本解码风格）。验证 LoRA 权重注入不意外修改 ViT 编码器参数（LoRA 应仅作用于 LLM 部分）。 |
| **价值描述** | 用户增长场景（VLM 定制化微调需求上升）；LoRA on NPU 仅 1 个源文件（ascend_backend.py）、npu_participation=medium，NPU 路径成熟度低于 GPU；验证点 "风格不同" 较主观——建议用定量指标（如输出长度分布、词汇多样性）辅助判断。**降级理由**: LoRA 失败仅影响微调用户，不影响基础多模态功能，P0 定位偏高，建议降为 P1。 |

---

#### P0-009: GDN 线性注意力 + 视觉编码器 → 多模态推理正确

| 项目 | 内容 |
|------|------|
| **场景** | 使用 GDN 线性注意力文本骨干的多模态模型（Qwen3.5-MoE），验证图像理解能力不受 GDN 循环状态影响 |
| **输入** | 图片 + "描述这张图片" |
| **期望结果** | 输出与图片内容相关，与同等规模的 Flash-Attention 文本骨干的多模态模型精度一致 |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. 无 NaN / 无静默回退至 FlashInfer |
| **关联特性** | attention_backend (GDN hybrid_linear), graph_compilation |
| **所需能力** | `vlm-gdn` |
| **用例价值** | **10/10** — GDN 线性注意力维护循环状态（recurrent state），图像 token 插入可能破坏状态连续性。Qwen3.5-MoE 是 GDN+ViT 唯一代码路径，无替代方案。NaN 或静默回退是已观察到的故障模式。 |
| **价值描述** | 唯一代码路径（无可替代模型验证同一交互）；GDN 循环状态是线性注意力的核心正确性前提；线上风险高——NaN 会导致整个 batch 输出损坏（非单请求级故障）。Qwen3.5-MoE 官方支持标记为 ❌，需确认测试环境可用性。 |

#### P0-010: GDN + MoE + 视觉编码器 → 多模态推理正确

| 项目 | 内容 |
|------|------|
| **场景** | 同时使用 GDN 线性注意力和 MoE 专家路由的多模态模型（Qwen3.5-MoE），验证专家路由在 GDN 循环状态下不被图像 token 偏斜 |
| **输入** | 图片 + "描述这张图片" |
| **期望结果** | 输出与图片相关，各专家利用率与纯文本请求分布一致 |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. 专家利用率无异常集中 |
| **关联特性** | attention_backend (GDN), moe, eplb |
| **所需能力** | `vlm-gdn` + `vlm-moe` |
| **用例价值** | **9/10** — GDN + MoE + ViT 三重交互是最复杂的多模态代码路径。GDN 循环状态影响 hidden states → hidden states 影响 MoE router → router 偏斜导致专家负载不均衡 → 部分专家过载影响吞吐。链路长、级联故障风险高。 |
| **价值描述** | 三重交互（唯一覆盖 GDN×MoE×ViT 的用例）；级联故障模式——GDN 状态偏差 → router 偏斜 → 专家过载 → 延迟飙升，每个环节的根因定位都困难。"专家利用率无异常集中"需要定义基线分布对比；建议同时监控 P0-009 的 GDN 状态健康度。 |

---

### 2.2 P1 用例（重要场景，涉及相关标志变更时执行）

#### P1-001: 投机解码 + 图像 → 加速不破坏正确性

| 项目 | 内容 |
|------|------|
| **场景** | 启用投机解码（EAGLE/EAGLE3），发送图像请求，验证输出正确且延迟降低 |
| **输入** | 图片 + "描述图片中的每个物体" |
| **期望结果** | 输出与不启用投机解码时一致，TTFT 更低 |
| **验证点** | 1. 输出内容与 baseline 一致（语义等价）2. TTFT < baseline TTFT |
| **关联特性** | speculative_decoding |
| **所需能力** | `vlm-mtp` |
| **用例价值** | **7/10** — 投机解码是性能关键特性，但在多模态下的正确性保障不足。草稿模型可能对图像 token 产生偏差的草稿 token，若验证阶段未正确拒绝则导致输出错误。 |
| **价值描述** | 性能与正确性的权衡型测试（投机解码可能引入 subtle 正确性损失）；**高风险**: `vlm-mtp` 能力标记为 tbc，如果 Qwen3-VL 不支持 MTP，P1-001 和 P1-010 无有效执行模型，需在上线前确认或替换测试模型。TTFT 降低的验证在 CI 环境可能不稳定（取决于机器负载），建议用 "TTFT ≤ baseline TTFT × 1.05 且吞吐 ≥ baseline 吞吐" 替代严格低于。 |

#### P1-002: PD 分离 + 图像请求 → 预填充节点传输正确

| 项目 | 内容 |
|------|------|
| **场景** | 在 PD 分离部署中，预填充节点处理图像编码，解码节点接收嵌入进行生成 |
| **输入** | 图片 + "描述图片" → 预填充节点 |
| **期望结果** | 解码节点正确接受图像嵌入并生成正确输出 |
| **验证点** | 1. 返回码 200 2. 输出与单节点部署一致 3. 无传输错误 |
| **关联特性** | pd_disaggregation |
| **所需能力** | `vlm` |
| **用例价值** | **8/10** — PD 分离是生产级部署的核心架构。图像嵌入（ViT 输出 hidden states）体积远大于文本 token embedding，传输层（Mooncake/zmq/GRPC）的序列化/反序列化是此场景独有风险。 |
| **价值描述** | 生产架构必经路径；图像嵌入传输体积大（可能触发传输层的未知边界条件）；验证 "输出与单节点一致" 需要确保两端使用相同的采样参数（seed、temperature）。建议增加 EPD（encoder-only prefill disaggregation）变体——pd_disaggregation 的 features.json 描述已提及 EPD 支持 VLM。 |

#### P1-003: TP 并行 + 图像 → 多卡推理正确

| 项目 | 内容 |
|------|------|
| **场景** | 使用张量并行（TP=2 或 TP=4）部署多模态模型，发送图像请求 |
| **输入** | 图片 + "描述图片" |
| **期望结果** | 输出正确，与单卡一致 |
| **验证点** | 1. 输出与 TP=1 时语义一致 2. 无 NCCL/HCCL 通信错误 |
| **关联特性** | tensor_parallelism |
| **所需能力** | `vlm` |
| **用例价值** | **8/10** — TP 是多卡部署的默认并行策略。ViT 编码器通常不参与 TP 分片（仅 LLM 部分分片），但 ViT 输出的 hidden states 需要 broadcast 到所有 TP rank，通信模式与纯文本不同。 |
| **价值描述** | 多卡部署基础验证；通信模式差异（ViT hidden states broadcast vs 文本 token embedding broadcast）可能触发 HCCL 的 corner case；TP=4 时通信开销显著增加，建议同时验证 TP=2 和 TP=4 的延迟退化是否线性可预测。 |

#### P1-004: DP-attention + 图像 → 高并发吞吐提升

| 项目 | 内容 |
|------|------|
| **场景** | 启用数据并行注意力（`--enable-dp-attention`），高并发图像请求 |
| **输入** | 50 并发图片 + "简短描述" 请求 |
| **期望结果** | 全部正确返回，吞吐量高于不启用时 |
| **验证点** | 1. 50 个请求全部成功 2. 吞吐量 > baseline 3. 无精度损失 |
| **关联特性** | dp_attention, tensor_parallelism |
| **所需能力** | `vlm` |
| **用例价值** | **7/10** — DP-attention 是高并发场景的吞吐优化。dp_attention 的 NPU 参与度标记为 weak（仅一个 import），NPU 路径成熟度低。50 并发验证了 DP 分片注意力在多模态 batch 下的正确性。 |
| **价值描述** | 高并发场景覆盖（补充 P0-003 的低并发）；dp_attention NPU participation=weak，该特性在 NPU 上可能退化或未充分测试；50 并发在 CI 环境可能需要较大 GPU 资源，建议根据 CI 机器规模调整并发数。 |

#### P1-005: 确定性推理 + 图像 → 两次运行输出一致

| 项目 | 内容 |
|------|------|
| **场景** | 启用确定性推理后，同一图片请求发送两次，验证输出完全一致 |
| **输入** | 同一图片 + "描述图片" × 2 次（相同 seed） |
| **期望结果** | 两次输出逐 token 一致 |
| **验证点** | 1. 两次输出完全相同 2. VLM 缓存正确被禁用 |
| **关联特性** | deterministic_inference |
| **所需能力** | `vlm` |
| **用例价值** | **5/10** — deterministic_inference 的 NPU 参与度标记为 `not_supported`（仅支持 FlashInfer/FA3/Triton 后端，不含任何 NPU attention backend）。**本用例在 NPU 上不可执行，仅适用于 GPU。** |
| **价值描述** | **NPU 阻塞**: features.json 明确记录 deterministic_inference 不支持 NPU。如果项目以 NPU 为主要平台，此用例应标注为 GPU-only 或降级为 P2（未来 NPU 支持后再启用）。GPU 上价值中等——确定性推理主要用于调试和基准测试，非生产必需。 |

#### P1-006: 结构化输出 + 图像 → JSON Schema 约束

| 项目 | 内容 |
|------|------|
| **场景** | 用户要求从图片中提取结构化信息（JSON 格式） |
| **输入** | 发票图片 + "提取金额、日期、商户名为 JSON" + `response_format: json_schema` |
| **期望结果** | 输出为合法 JSON，包含指定字段，字段值来源于图片 |
| **验证点** | 1. 输出为合法 JSON 2. 提取信息与图片内容吻合 3. 不崩溃 |
| **关联特性** | structured_outputs (xgrammar) |
| **所需能力** | `vlm` |
| **用例价值** | **6/10** — xgrammar 是 CPU 侧的语法约束引擎（platform_agnostic），与 ViT 编码无直接内核交互。结构化输出对图像 token 的约束本质上与文本 token 相同。交互风险低但用户场景常见（票据识别、表单提取）。 |
| **价值描述** | 业务场景高频（OCR + 结构化提取是 VLM 的核心应用）；技术交互风险低（xgrammar 处理 logits 约束，不感知 token 来源）；JSON 合法性是硬验证条件，易于自动化。P1 定位合理——语法引擎变更时执行即可。 |

#### P1-007: 工具调用 + 图像参数 → 函数参数中传递图片

| 项目 | 内容 |
|------|------|
| **场景** | 模型使用工具调用，其中某工具的 `image_url` 参数来源于输入图片 |
| **输入** | 图片 + "分析这张图片，调用 analyze_image 工具"，注册 `analyze_image({"image": "..."})` |
| **期望结果** | 模型正确输出工具调用，`image` 参数不丢失不损坏 |
| **验证点** | 1. 工具调用格式正确 2. image 参数包含有效的 base64 或 URL 3. 不崩溃 |
| **关联特性** | tool_parser |
| **所需能力** | `vlm` |
| **用例价值** | **6/10** — tool_parser 是 CPU 侧文本解析（platform_agnostic），与 ViT 编码无直接交互。多模态 Agent 场景（如 "看图调用工具"）在生产中有实际需求，但技术交互链路短。 |
| **价值描述** | Agent 场景覆盖（多模态 Agent 是趋势）；技术风险低（tool_parser 仅解析 LLM 输出文本，不感知输入来源）；image 参数验证（base64/URL）是确定性的格式检查，自动化友好。P1 定位合理。 |

#### P1-008: 分块预填充 + Offloading → 嵌入在块间持久化

| 项目 | 内容 |
|------|------|
| **场景** | 启用 CPU offloading，发送长文本 + 图片请求（触发分块预填充），验证图像嵌入在块间正确传递 |
| **输入** | 5K token 前缀 + 图片 + "总结" / `--cpu-offload-gb=4` |
| **期望结果** | 输出涉及图片内容，证明图像特征在 offload 后仍可用 |
| **验证点** | 1. 输出与图片相关 2. GPU 内存使用 < baseline 3. 不崩溃 |
| **关联特性** | offloading, chunked_prefill |
| **所需能力** | `vlm` |
| **用例价值** | **8/10** — offloading 将模型权重在 GPU/CPU 间换入换出，与 chunked_prefill 的跨 chunk 状态管理叠加后，图像嵌入可能在 GPU↔CPU 传输中丢失或损坏。三重交互（offload + chunk + multimodal）是真实的内存受限场景。 |
| **价值描述** | 内存受限部署场景（小 GPU 大模型）的关键路径；三重交互（offload × chunk × multimodal），任何一层出问题都导致图像理解失败；GPU 内存使用 < baseline 是可机器验证的硬指标。建议增加多轮请求（验证 offload 重复换入换出后的稳定性）。 |

#### P1-009: DP-attention + DP LM Head + 图像 → LM 头分片不影响图像 token 投影

| 项目 | 内容 |
|------|------|
| **场景** | 启用 DP-attention 和 DP LM Head 分片，发送图像请求，验证词汇投影在 LM 头分片后仍正确 |
| **输入** | 图片 + "描述图片" / `--enable-dp-attention` + `--enable-dp-lm-head` |
| **期望结果** | 输出与未启用 `--enable-dp-lm-head` 时语义一致，图像 token 的 logits 投影不被分片截断 |
| **验证点** | 1. 输出语义等价于 baseline 2. 无 NaN / 无截断 3. 吞吐 > baseline |
| **关联特性** | dp_attention, hardware_backend |
| **所需能力** | `vlm` |
| **来源** | ascend_npu_optimization.mdx: DP LM Head — compatibility matrix shows 🟠 with EP, PD, Quantization, Chunked Prefill, NPU Graph |
| **用例价值** | **7/10** — DP LM Head 是 NPU 特有的性能优化（将 LM head 分片到各 DP rank）。兼容矩阵标记 🟠 与多项特性，说明已知存在交互风险。图像 token 的 logits 投影若在分片边界截断会产生错误的 token 分布。 |
| **价值描述** | NPU 独有路径（GPU 无 DP LM Head 概念）；兼容矩阵 🟠 表明该特性已知与其他特性存在部分不兼容，测试必要性高；吞吐 > baseline 验证了优化的有效性。验证点 "无 NaN/截断" 是硬指标，易于判定。 |

#### P1-010: Overlap Schedule + 投机解码 + 图像 → 重叠调度不破坏多模态草稿

| 项目 | 内容 |
|------|------|
| **场景** | 启用投机解码和重叠调度，发送图像请求，验证草稿生成与验证的重叠不会破坏多模态上下文 |
| **输入** | 图片 + "描述图片" / `--speculative-algorithm NEXTN` + `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` + `SGLANG_ENABLE_SPEC_V2=1` |
| **期望结果** | 输出与不启用重叠调度时一致，TTFT 更低 |
| **验证点** | 1. 输出语义等价于 baseline 2. TTFT < baseline TTFT 3. 无崩溃 |
| **关联特性** | speculative_decoding |
| **所需能力** | `vlm-mtp` |
| **来源** | ascend_npu_optimization.mdx: Overlap Schedule — compatibility matrix shows 🟠 with Chunked Prefill, NPU Graph, Quantization |
| **用例价值** | **6/10** — Overlap Schedule 是多流并行优化的实验性特性（需手动设置 env var），兼容矩阵 🟠 多项。依赖 `vlm-mtp`（tbc），且投机解码本身与 NPU Graph 的兼容性也是 🟠。执行前置条件复杂，CI 稳定性存疑。 |
| **价值描述** | 高级优化场景（多流重叠在极端吞吐要求下有价值）；执行风险高——需同时满足投机解码可用 + MTP 模型可用 + overlap env var 正确；**与 P1-001 共享 vlm-mtp 风险**。建议在 P1-001 稳定通过后再执行此用例。 |

---

### 2.3 P2 用例（边界场景，回归套件中执行）

#### P2-001: MoE 模型 + 图像 → 专家路由无偏差

| 项目 | 内容 |
|------|------|
| **场景** | 使用 MoE 多模态模型（Qwen3-VL-MoE），分别发送 20 个纯文本和 20 个图像请求，比对专家利用率 |
| **输入** | 20 纯文本 + 20 图像请求，观察 MoE 专家负载分布 |
| **期望结果** | 图像请求的专家分布不出现极端集中（与纯文本分布相似） |
| **验证点** | 1. 所有请求成功 2. 专家利用率标准差在合理范围内 |
| **关联特性** | moe, eplb |
| **所需能力** | `vlm-moe` |
| **用例价值** | **5/10** — 验证图像 token 不会系统性偏斜 MoE router（如图像 patch token 全部路由到同一位专家）。对正确性无直接影响（router 偏斜主要影响吞吐），属性能/公平性验证。 |
| **价值描述** | 性能优化型验证（router 偏斜导致专家负载不均→吞吐下降）；20+20 样本量偏小，统计显著性不足——建议增至 100+100 或持续发送直到分布稳定。"合理范围"需量化（如标准差<均值的 30%）。P2 定位合理。 |

#### P2-002: 上下文并行 + 长图像序列 → 长上下文正确

| 项目 | 内容 |
|------|------|
| **场景** | 在长上下文场景中启用上下文并行（`--attn-cp-size=2`），发送需多图像的长序列请求 |
| **输入** | 10 张图片 + 长文本交错排列 + "按顺序描述每张图片" |
| **期望结果** | 输出按顺序正确描述每张图片，不丢失上下文 |
| **验证点** | 1. 输出包含所有 10 张图片的描述 2. 顺序正确 3. 不 OOM |
| **关联特性** | context_parallelism |
| **所需能力** | `vlm` |
| **用例价值** | **5/10** — context_parallelism 的 NPU 参与度为 weak（仅通过共享 HCCL process group），无 NPU 专用 CP 代码。10 张图片产生大量 image tokens，CP 分片后跨 rank 的 attention 计算可能遗漏某些图片 token。 |
| **价值描述** | 长上下文 + 多图是真实场景（文档理解、视频帧分析）；context_parallelism on NPU 成熟度低（weak participation）；10 张图片的 token 数可能非常大（取决于图片分辨率），需注意 CI 环境的显存容量。P2 定位合理。 |

#### P2-003: HiCache + 图像 → 分层缓存不影响精度

| 项目 | 内容 |
|------|------|
| **场景** | 启用 HiCache 分层 KV 缓存，发送图像请求 |
| **输入** | 图片 + "描述图片"，启用 `--enable-hicache` |
| **期望结果** | 输出与不启用 HiCache 时一致 |
| **验证点** | 1. 输出语义不变 2. 缓存层正确工作 3. 不崩溃 |
| **关联特性** | hicache |
| **所需能力** | `vlm` |
| **用例价值** | **4/10** — HiCache 的 NPU 参与度标记为 weak（仅配置层），NPU IO backend 仅处理 memory layout 选择。图像 token 的 KV cache 在 GPU→host→disk 层级间移动时可能丢失，但 NPU 上实际不执行磁盘 offload。 |
| **价值描述** | NPU 上交互风险极低（weak participation, 无磁盘层）；GPU 上价值更高但非 NPU 平台重点。P2 定位合理，可考虑在 NPU 平台上标记为 "仅 GPU 执行" 或跳过。验证点 "缓存层正确工作" 过于模糊——建议改为验证请求 2（相同前缀）TTFT 降低。 |

#### P2-004: 量化模型 + 图像 → FP8 精度不丢失

| 项目 | 内容 |
|------|------|
| **场景** | 使用 FP8 量化模型部署，发送图像请求，验证精度可接受 |
| **输入** | 量化模型 + 图片 + "描述图片" |
| **期望结果** | 输出与 FP16 baseline 语义一致 |
| **验证点** | 1. 输出语义等价于 FP16 2. 吞吐更高 3. 不崩溃 |
| **关联特性** | quantization |
| **所需能力** | `vlm` |
| **用例价值** | **6/10** — 量化是生产部署的标配（降低显存、提升吞吐）。量化误差在 ViT 编码阶段累积后影响图像特征质量，进而影响 LLM 的图像理解。量化 + 多模态的精度损失模式与纯文本不同（视觉特征对量化噪声更敏感）。 |
| **价值描述** | 生产部署必需（量化模型占比高）；ViT 编码阶段的量化误差传播链路长（ViT 量化 → hidden states 偏差 → LLM 理解偏差）；"语义等价" 判定主观——建议对比量化/非量化模型在同一图片上的 token 级别 logits 差异（KL divergence）。P2 定位合理但建议提升为 P1（量化用户群体大）。 |

#### P2-005: KV Cache 量化 + 图像 → 精度可接受

| 项目 | 内容 |
|------|------|
| **场景** | 启用 KV Cache 量化（`--kv-cache-dtype fp8_e4m3`），发送图像请求 |
| **输入** | 图片 + "描述图片" |
| **期望结果** | GPU: 输出语义不退化。NPU: 量化 KV 缓存不支持（`npu_participation: not_supported`），应优雅拒绝或回退 |
| **验证点** | 1. GPU: 输出与未量化一致 2. NPU: 启动时报错或自动回退，不静默产生错误结果 3. 不崩溃 |
| **关联特性** | quantized_kv_cache |
| **所需能力** | `vlm` |
| **用例价值** | **5/10** — quantized_kv_cache 的 NPU 参与度标记为 `not_supported`（NPU attention backend 缺少 fused dequant+attention kernel）。GPU 上价值中等；NPU 上的"优雅拒绝"验证有安全价值——防止用户误开启导致静默精度损失。 |
| **价值描述** | NPU 上主要验证错误处理（非功能验证）；"优雅拒绝"是重要的安全网——避免用户误配置后得到错误结果而不自知。P2 定位合理。建议将此用例拆分为 GPU 子用例（精度验证）和 NPU 子用例（错误处理验证）。 |

#### P2-006: Pipeline Parallelism + 图像 → 流水线正确

| 项目 | 内容 |
|------|------|
| **场景** | 使用 PP=2 部署支持 PP 的多模态模型，发送图像请求 |
| **输入** | 图片 + "描述图片" |
| **期望结果** | 输出与 PP=1 一致 |
| **验证点** | 1. 输出语义一致 2. 不崩溃 |
| **关联特性** | pipeline_parallelism |
| **所需能力** | `vlm-pp` |
| **用例价值** | **5/10** — PP 用于超大规模模型跨节点部署。ViT 编码器通常在 PP stage 0，图像 embedding 需要跨 PP stage 传输。传输正确性依赖于 send/recv 的 metadata 同步。 |
| **价值描述** | 大模型部署场景（PP 使用率低于 TP）；技术风险中等（ViT 在 stage 0 → embedding 经 P2P 传输到后续 stage）；需要 `vlm-pp` 能力标签——仅 Qwen3-VL 标记支持 PP，需确认 CI 环境是否有 PP=2 的 GPU 资源。P2 定位合理。 |

#### P2-007: 权重预取 + 图像 → CMO 不影响 ViT

| 项目 | 内容 |
|------|------|
| **场景** | 在 NPU 上启用 CMO 权重预取，发送图像请求 |
| **输入** | 图片 + "描述图片" / NPU + `--enable-cmo-prefetch` |
| **期望结果** | 输出正确，ViT 权重预取不影响图像编码 |
| **验证点** | 1. 输出正确 2. 吞吐可能提升 3. 不崩溃 |
| **关联特性** | weight_prefetch |
| **所需能力** | `vlm` |
| **用例价值** | **4/10** — CMO 权重预取是 Ascend 硬件独有特性（NPU L2 cache prefetch）。ViT 权重通常较小，预取收益有限。与 ViT 编码的并发 stream 交互是主要风险（预取 stream 和编码 stream 的同步）。 |
| **价值描述** | NPU 独有特性（GPU 无对应功能）；ViT 权重体积小，预取对 ViT 编码的加速效果有限，但预取 stream 与 ViT 编码 stream 的同步 bug 可能导致静默数据损坏；P2 定位合理，"吞吐可能提升" 验证较弱——建议明确为 "不导致吞吐下降"（防御性验证）。 |

#### P2-008: EPLB + 图像 → MoE 专家负载均衡不受影响

| 项目 | 内容 |
|------|------|
| **场景** | 启用 EPLB 负载均衡的 MoE 多模态模型，连续发送图像请求 |
| **输入** | 启用 `--enable-eplb` + 50 并发图片请求 |
| **期望结果** | EPLB 弹性重分布不影响推理正确性 |
| **验证点** | 1. 全部请求成功 2. EPLB 统计方法不因图像 token 而偏斜 3. 不崩溃 |
| **关联特性** | eplb |
| **所需能力** | `vlm-moe` |
| **用例价值** | **4/10** — EPLB 是纯算法层（platform_agnostic），基于 expert activation 统计进行负载均衡。图像 token 的 activation pattern 与文本不同，EPLB 统计若未考虑多模态分布可能导致重分布决策次优，但不影响单请求正确性。 |
| **价值描述** | 负载均衡算法验证（非功能正确性）；图像 token activation pattern 可能触发 EPLB 不必要的重分布（抖动），影响长期稳定吞吐；50 并发在 CI 环境执行成本高。P2 定位合理。 |

#### P2-009: Reasoning Parser + 图像 → 不破坏思维链

| 项目 | 内容 |
|------|------|
| **场景** | 使用推理模型 + 图像输入，要求模型显示思维链 |
| **输入** | 图片 + "请逐步分析这张图的内容" / 启用 `--reasoning-parser` |
| **期望结果** | 推理模型的思维链（`&lt;think&gt;...&lt;/think&gt;`）在 response 中完整保留，不因图像 token 被截断 |
| **验证点** | 1. 存在 `&lt;think&gt;` 和 `&lt;/think&gt;` 标签对 2. 思维链内容与图片相关 3. 不崩溃 |
| **关联特性** | separate_reasoning |
| **所需能力** | `vlm-reasoning` |
| **用例价值** | **6/10** — Reasoning parser 是 CPU 侧文本解析（platform_agnostic），与 ViT 编码无直接交互。但思维链模式（`<think>` 标签）在多模态下可能因 image token 对 hidden states 的影响而产生不完整的推理路径。视觉推理（如 "图中发生了什么"）是 VLM 的核心使用场景。 |
| **价值描述** | 视觉推理是 VLM 的高价值场景；技术交互风险低（parser 仅处理文本分割）；验证点明确（标签对完整性可机器检查）；GLM-4.6V 和 Qwen3-VL 支持 reasoning parser，覆盖面好。P2 定位合理。 |

#### P2-010: 全模型 DP + 图像 → 多副本推理一致

| 项目 | 内容 |
|------|------|
| **场景** | 使用 `--dp-size=2` 全模型数据并行，发送图像请求 |
| **输入** | 50 并发图片请求 |
| **期望结果** | 各 DP 副本上的输出一致，负载均衡有效 |
| **验证点** | 1. 全部成功 2. 副本间输出语义一致 3. 吞吐 ≈ 2x baseline |
| **关联特性** | data_parallelism |
| **所需能力** | `vlm` |
| **用例价值** | **5/10** — DP 是全模型复制（非分片），每个副本独立运行完整推理。多模态下主要风险是各 DP rank 的 ViT 编码器产生不同 embedding（由于 float non-determinism），但 replica 间的请求是独立的，无直接交互。 |
| **价值描述** | 高吞吐部署场景的基础验证；DP 副本间无共享状态，交互风险本质低（每个副本独立处理分配到它的请求）；"副本间输出语义一致" 需明确是指 "同一请求分配到不同副本时输出一致" 还是指 "各副本处理的请求输出质量无退化"。P2 定位合理。 |

#### P2-011: 量化 + 分块预填充 + 图像 → 量化精度跨块保持

| 项目 | 内容 |
|------|------|
| **场景** | 使用量化模型，发送长文本 + 图片请求（触发分块预填充），验证量化精度在跨块边界不退化 |
| **输入** | 3K token 前缀 + 图片 + "总结" / `--quantization modelslim` + chunked prefill |
| **期望结果** | 输出语义与 FP16 baseline 一致，图像内容不因量化精度损失而丢失 |
| **验证点** | 1. 输出与图片相关 2. 语义等价于 FP16 + non-chunked 3. 不崩溃 |
| **关联特性** | quantization, chunked_prefill |
| **所需能力** | `vlm` |
| **来源** | ascend_npu_optimization.mdx: 兼容矩阵 Quantization 🟠 Chunked Prefill |
| **用例价值** | **7/10** — 三重交互（量化 × 分块预填充 × 多模态），兼容矩阵标记 🟠。量化误差 + chunk 边界的信息丢失可能叠加——量化模型的 hidden states 精度已降低，跨 chunk 时进一步累积误差，图像 token 位于 chunk 边界时精度损失最大。 |
| **价值描述** | 🟠 兼容矩阵条目直接触发——已知存在部分兼容性问题；三重交互叠加使误差模式复杂（非单一根因）；验证条件严格（需对比 FP16+non-chunked baseline），执行成本高但发现问题的价值也高。P2 定位偏保守——建议提升为 P1（兼容矩阵 🟠 通常意味着已知风险）。 |

#### P2-012: 投机解码 + NPU Graph + 图像 → 图兼容草稿验证

| 项目 | 内容 |
|------|------|
| **场景** | 启用投机解码且不关闭 NPU Graph 时发送图像请求，验证图回放与草稿验证不冲突 |
| **输入** | 图片 + "描述图片" / `--speculative-algorithm NEXTN` + 不传 `--disable-cuda-graph` |
| **期望结果** | 输出正确，不因 NPU Graph 与投机解码的 🟠 兼容性而崩溃 |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. 不崩溃（若不兼容则至少优雅报错） |
| **关联特性** | speculative_decoding, graph_compilation |
| **所需能力** | `vlm-mtp` |
| **来源** | ascend_npu_optimization.mdx: 兼容矩阵 Speculative Decoding 🟠 NPU Graph |
| **用例价值** | **6/10** — 投机解码的 EAGLE draft graph runner 和 NPU Graph runner 是两条独立的图编译路径，🟠 兼容性标记表明已知冲突。不崩溃是底线，但 "输出正确" 验证了图回放下的草稿 token 验证正确性。 |
| **价值描述** | 🟠 兼容矩阵直接触发——投机解码 + NPU Graph 是已知的部分兼容组合；依赖 `vlm-mtp`（tbc）——与 P1-001/P1-010 共享风险。验证门槛低（"至少不崩溃"），P2 定位合理但应在 vlm-mtp 确认可用后再排期。 |

#### P2-013: DP-attention + NPU Graph + 图像 → DP 下的图回放正确

| 项目 | 内容 |
|------|------|
| **场景** | 启用 DP-attention 且不关闭 NPU Graph 时发送图像请求 |
| **输入** | 图片 + "描述图片" / `--enable-dp-attention` + 不传 `--disable-cuda-graph` |
| **期望结果** | DP 分片注意力在 NPU Graph 回放模式下正确工作 |
| **验证点** | 1. 返回码 200 2. 输出与图片相关 3. TPOT 与基线一致 4. 不崩溃 |
| **关联特性** | dp_attention, graph_compilation |
| **所需能力** | `vlm` |
| **来源** | ascend_npu_optimization.mdx: 兼容矩阵 DP 🟠 NPU Graph |
| **用例价值** | **6/10** — DP-attention 与 NPU Graph 的 🟠 兼容性。DP 分片注意力的 batch 内通信 pattern 与 NPU Graph 的静态图假设可能冲突（图捕获时的通信拓扑 vs 回放时的实际 DP 分片）。TPOT 验证提供了比 "不崩溃" 更强的正确性信号。 |
| **价值描述** | 🟠 兼容矩阵触发；不依赖 vlm-mtp（相比 P2-012 执行条件更宽松）；TPOT 验证是强信号（DP 分片错误会导致 decode 阶段延迟异常）。P2 定位合理。 |

#### P2-014: Multistream MoE + 多模态 MoE 模型 → 双流不破坏图像路由

| 项目 | 内容 |
|------|------|
| **场景** | 在 NPU 上启用双流 MoE 执行，发送图像请求，验证共享专家和路由专家并行执行时，图像 token 路由不受影响 |
| **输入** | 图片 + "描述图片" / `SGLANG_NPU_USE_MULTI_STREAM=1` |
| **期望结果** | 输出正确，专家路由在双流模式下与单流一致 |
| **验证点** | 1. 返回码 200 2. 输出语义一致 3. 不崩溃 |
| **关联特性** | moe, hardware_backend |
| **所需能力** | `vlm-moe` |
| **来源** | ascend_npu_optimization.mdx: Multistream MoE — compatibility matrix shows 🟠 with Chunked Prefill, NPU Graph |
| **用例价值** | **5/10** — NPU 独有特性（`SGLANG_NPU_USE_MULTI_STREAM`）。双流执行下共享专家和路由专家并行，stream 同步不当可能导致图像 token 的路由结果在错误的 stream 上被消费。 |
| **价值描述** | NPU 独有路径（GPU 无对应功能）；双流同步 bug 是经典的 CUDA/NPU stream 编程陷阱；env var 触发方式使 CI 集成简单；兼容矩阵 🟠 Chunked Prefill 和 NPU Graph，但本用例未同时测试这些组合——建议增加组合测试（multistream + chunked_prefill + 图像）。P2 定位合理。 |

---

## 3. 测试模型矩阵

> 每个模型具备的能力标签在前一章节的"能力标签说明"中定义。选用模型时，确保所选模型的标签覆盖用例的"所需能力"。

| 模型系列 | 能力标签 | P0 | P1 | P2 | NPU | 官方 | 特点 |
|-----------|----------|-----|-----|-----|-----|------|------|
| Qwen3-VL | vlm, vlm-moe, vlm-lora, vlm-reasoning, vlm-mtp(tbc), vlm-pp | ✓ | ✓ | ✓ | ✓ | ✅ | DeepStack ViT + MoE，当前主力，最多能力标签 |
| Qwen2.5-VL | vlm | ✓ | ✓ | ✓ | ✓ | ✅ | 标准 ViT, 非 MoE 基线, 最广泛使用 |
| Qwen3.5-MoE | **vlm, vlm-gdn, vlm-moe** | ✓ | ✓ | ✓ | ✓ | ❌ | GDN+MoE+ViT 唯一路径，同时覆盖 GDN 和 MoE |
| GLM-4.6V | vlm, vlm-reasoning | ✓ | ✓ | ✓ | ✓ | ❌ | NPU 专用 patch processor（glm46v），唯一 NPU 补丁路径 |

**官方支持依据**: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx` Multimodal Language Models 章节
**移除说明**: Qwen3.5（非 MoE GDN 路径已被 Qwen3.5-MoE 覆盖）；GLM-4.5V（标准处理器路径已被 Qwen2.5-VL 覆盖，GLM 路径已被 GLM-4.6V 覆盖）；GLM-4V / Qwen2-VL / Dots-VLM-ViT / InternVL3.5 / IDEFICS2 / MiniCPMV-ViT / PaddleOCR-VL（代码路径冗余或未列入官方 NPU 支持列表）

**P0 执行**: Qwen3-VL + Qwen2.5-VL + Qwen3.5-MoE + GLM-4.6V（覆盖 DeepStack/MoE + 标准非 MoE + GDN + NPU patch 四条独立代码路径）
**P1 执行**: P0 全部 4 个模型
**P2 回归**: P0 全部 4 个模型

---

## 4. 用例 × 交互覆盖矩阵

| 用例 | attn | graph | sched | chunk | TP | DP | PD | spec | LoRA | offl | cache | struct | tool | det | 其他 |
|------|------|-------|-------|-------|----|----|----|------|------|------|-------|--------|------|-----|------|
| P0-001 | ✓ | | ✓ | | | | | | | | | | | | |
| P0-002 | | | | | | | | | | | ✓ | | | | |
| P0-003 | ✓ | | ✓ | | | | | | | | | | | | |
| P0-004 | ✓ | | ✓ | | | | | | | | | | | | |
| P0-005 | ✓ | ✓ | | | | | | | | | | | | | |
| P0-006 | | | ✓ | ✓ | | | | | | | | | | |
| P0-007 | | ✓ | | | | | | | | | | | | | |
| P0-008 | | | | | | | | | ✓ | | | | | | |
| P0-009 | ✓(gdn) | | | | | | | | | | | | | | |
| P0-010 | ✓(gdn) | | | | | | | | | | | | | | | gdn+moe |
| P1-001 | | | | | | | | ✓ | | | | | | | |
| P1-002 | | | | | | | ✓ | | | | | | | | |
| P1-003 | | | | | ✓ | | | | | | | | | | |
| P1-004 | | | | | ✓ | ✓ | | | | | | | | | |
| P1-005 | | | | | | | | | | | | | | ✓ | |
| P1-006 | | | | | | | | | | | | ✓ | | | |
| P1-007 | | | | | | | | | | | | | ✓ | | |
| P1-008 | | | | ✓ | | | | | | ✓ | | | | | |
| P1-009 | | | | | | ✓ | | | | | | | | | dp-lm |
| P1-010 | | | | | | | | ✓ | | | | | | | overlap |
| P2-001 | | | | | | | | | | | | | | | moe,eplb |
| P2-002 | | | | | | | | | | | | | | | ctx_par |
| P2-003 | | | | | | | | | | | hicache | | | | |
| P2-004 | | | | | | | | | | | | | | | quant |
| P2-005 | | | | | | | | | | | kv-quant | | | | |
| P2-006 | | | | | | | | | | | | | | | pp |
| P2-007 | | | | | | | | | | | | | | | prefetch |
| P2-008 | | | | | | | | | | | | | | | eplb |
| P2-009 | | | | | | | | | | | | | | | reason |
| P2-010 | | | | | | ✓ | | | | | | | | | dp |
| P2-011 | | | | ✓ | | | | | | | | | | | quant+chunk |
| P2-012 | | ✓ | | | | | | ✓ | | | | | | | spec+graph |
| P2-013 | | ✓ | | | | ✓ | | | | | | | | | dp+graph |
| P2-014 | | | | | | | | | | | | | | | moe+multistream |

---

## 5. 附录：实现参考（代码路径）

> 以下为用例实现时建议的代码参考路径，帮助定位相关逻辑。

| 交互 | 关键代码位置 | 类型 |
|------|-------------|------|
| ViT 编码器入口 | `python/sglang/srt/models/qwen3_vl.py` | 主模型文件 |
| 多模态处理器基类 | `python/sglang/srt/multimodal/processors/base_processor.py` | 处理器框架 |
| 多模态嵌入例程 | `python/sglang/srt/managers/mm_utils.py:879,1113` | 嵌入+offload |
| Radix 缓存哈希 | `python/sglang/srt/managers/schedule_batch.py:140` | 前缀缓存 |
| NPU ViT 图运行器 | `python/sglang/srt/hardware_backend/npu/graph_runner/vit_npu_graph_runner.py` | NPU 图编译 |
| NPU Qwen VL 处理器 | `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` | NPU 处理器补丁 |
| NPU GLM-4.6V 处理器 | `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py` | NPU 处理器补丁 |
| LoRA 适配器门控 | `python/sglang/srt/models/qwen3_vl.py:1236` | LoRA |
| 确定性推理 VLM 缓存 | `python/sglang/srt/server_args.py:4189` | 确定性推理 |
| 多模态特征释放 | `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:826` | 可观测性 |
| EAGLE3 层配置 | `python/sglang/srt/models/qwen3_vl.py:1403` | 投机解码 |
| DFLASH 层配置 | `python/sglang/srt/models/qwen3_vl.py:1305` | 投机解码 |

---

## 6. 正确性审查报告（2026-06-11）

> **审阅方法**: 逐条对文档内容与 features.json（39 features）、graphify 知识图谱进行交叉验证；逐用例评估 NPU 参与度标记、能力标签完整性、期望结果可验证性。

### 6.1 发现的问题

#### P1: 交互拓扑遗漏（§1.2 修复）

原始文档声称 "22 个特性与 multimodal 交互"，但以下特性出现在了 §4 覆盖矩阵和测试用例中却未列入交互列表：

| 遗漏特性 | 相关用例 | features.json npu | 说明 |
|----------|----------|-------------------|------|
| `dp_for_multi_modal_encoder` | 无（缺失！）| platform_agnostic | **最严重遗漏**——专门针对 VLM ViT 编码器的数据并行特性，与 multimodal 直接相关，应排入交互前 10 |
| `dp_attention` | P1-004, P1-009, P2-013 | weak | 出现在覆盖矩阵但不在交互列表中 |
| `lora` | P0-008 | medium | P0 用例已覆盖但交互列表遗漏 |
| `structured_outputs` | P1-006 | platform_agnostic | P1 用例已覆盖但交互列表遗漏 |
| `tool_parser` | P1-007 | platform_agnostic | P1 用例已覆盖但交互列表遗漏 |

> **已修复**: §1.2 交互拓扑表已扩展为 27 个特性（排名 1~15 显示 + 16~27 折叠）。**建议**: 为 `dp_for_multi_modal_encoder` 增加专项测试用例。

#### P2: NPU 不支持特性的测试用例未标注平台限制

| 用例 | 特性 | features.json npu | 当前状态 | 风险 |
|------|------|-------------------|----------|------|
| **P1-005** | deterministic_inference | **not_supported** | 未标注 NPU 不可用 | **高** — CI 在 NPU 上执行必然失败或静默跳过 |
| **P2-005** | quantized_kv_cache | **not_supported** | 已正确标注 | ✓ 无需修复 |

> **已修复**: P1-005 价值描述中标注了 NPU 阻塞，建议降为 GPU-only P2 或标注 `pytest.mark.skip_npu`。

#### P3: `vlm-mtp` 能力标签风险

`vlm-mtp` 标记为 `tbc`（待确认），且仅 Qwen3-VL 可能支持（Qwen3.5-MoE 不支持）。

| 受影响用例 | 优先级 | 风险 |
|-----------|--------|------|
| P1-001（投机解码+图像）| P1 | 若无 vlm-mtp 模型，用例无法执行 |
| P1-010（Overlap+投机+图像）| P1 | 同上，且依赖更多 env var |
| P2-012（投机+NPU Graph+图像）| P2 | 同上 |

> **影响**: 3 个用例（2 P1 + 1 P2）的执行依赖于 tbc 能力。**建议**: 在测试排期前确认 Qwen3-VL MTP 支持状态；若不支持，则这些用例需标记为 blocked 或寻找替代模型。

#### P4: 模型覆盖与用例所需能力的匹配

§3 模型矩阵声明 "P1 执行: P0 全部 4 个模型"，但：

- P1-001、P1-010 需要 `vlm-mtp`——Qwen2.5-VL、Qwen3.5-MoE、GLM-4.6V 均无此能力
- P2-006 需要 `vlm-pp`——仅 Qwen3-VL 标记支持
- P2-014 需要 `vlm-moe` + NPU 双流——仅 Qwen3-VL（Qwen3.5-MoE 有 moe 但无 multistream 确认）

> **建议**: §3 增加 "各用例 × 模型可用性矩阵"，标注每个用例在哪个模型上可执行。

#### P5: 验证条件不够精确的用例

| 用例 | 问题 | 建议 |
|------|------|------|
| P0-002 | "TTFT₂ < TTFT₁ × 0.6" — 若 ViT 编码时间占 TTFT 的 80%，缓存仅节省 20%（LLM prefill），0.6× 不可达 | 分 ViT 耗时占比设定阈值，或改为 "TTFT₂ < TTFT₁" |
| P0-004 | "输出同时涉及两张图片" — 主观判定 | 改为结构化验证：要求模型列出 "图片1: ... 图片2: ..." |
| P1-005 | "两次输出逐 token 一致" — NPU 不支持 deterministic_inference | 标记为 GPU-only |
| P2-001 | "专家利用率标准差在合理范围内" — 未量化 | 定义 σ < μ × 30%，或对比纯文本分布做 KS-test |
| P2-003 | "缓存层正确工作" — 过于模糊 | 改为验证请求 2（相同前缀）TTFT 降低 |

#### P6: 缺失的交互维度

以下特性与 multimodal 在 features.json 中存在交互但无对应测试用例：

| 缺失特性 | npu | 影响 | 建议 |
|----------|-----|------|------|
| `dp_for_multi_modal_encoder` | platform_agnostic | ViT DP 编码（降低 TTFT）——最直接的 multimodal 相关特性 | **增加 P1 级测试用例** |
| `multimodal_gen` | strong | 多模态生成（diffusion）与 multimodal 推理可能共享 ViT 编码器代码 | 评估共享代码路径后决定 |

### 6.2 评分摘要

| 优先级 | 平均分 | 最高 | 最低 |
|--------|--------|------|------|
| **P0 (10 用例)** | **8.7** | 10 (P0-001, P0-009) | 7 (P0-008) |
| **P1 (10 用例)** | **6.6** | 8 (P1-002, P1-003, P1-008) | 5 (P1-005) |
| **P2 (14 用例)** | **5.4** | 7 (P2-011) | 4 (P2-003, P2-007, P2-008) |

> P1-005 (deterministic_inference) 评分 5 主要因 NPU 不可用；若仅在 GPU 上执行则评分可上调至 7。

### 6.3 合理性总结

**设计优点**:
1. 用例从用户视角描述（非实现细节），可读性好
2. 能力标签（vlm/vlm-moe/vlm-gdn/...）替代具体模型名，可扩展
3. P0→P1→P2 分层策略清晰，P0 覆盖了最核心的交互
4. §5 实现参考提供了代码路径回溯能力
5. 模型矩阵精简合理（9→4 模型），减少了冗余

**设计改进建议**:
1. 补充 `dp_for_multi_modal_encoder` 测试用例（P1 优先级）
2. 建立 "用例 × 模型可用性矩阵"，标注阻塞项（vlm-mtp tbc、NPU not_supported）
3. 将 P1-005 降级为 GPU-only P2（NPU not_supported）
4. P0-008 (LoRA) 建议降级为 P1——LoRA 失败仅影响微调用户，非核心多模态功能
5. 量化验证条件（P0-002 的 0.6× 阈值、P2-001 的标准差范围）
6. 对 🟠 兼容矩阵触发的用例（P1-009, P1-010, P2-011, P2-012, P2-013, P2-014）增加交叉组合测试（如 multistream + chunked + 图像）

---

## 7. 审查记录

| 日期 | 内容 | 发现 | 状态 |
|------|------|------|------|
| 2026-06-10 | 初始分析 + 3 agent 对抗验证 | 17 个缺失交互, 5 个代码引用错误, 3 个边数遗漏 | 已修复 |
| 2026-06-10 | E2E 重构 | 将 39 个实现级测试点重组为 25 个业务级 E2E 用例 + 实现附录 | 当前版本 |
| 2026-06-11 | 人工正确性审查 + 价值打分 | 5 个交互拓扑遗漏 (P1), 1 个 NPU 不支持未标注 (P1-005, P2), vlm-mtp tbc 风险 (P3), 模型覆盖不完全 (P4), 5 个验证条件不精确 (P5), 2 个缺失交互维度 (P6) | P1+P2 已修复, P3~P6 待讨论 |
