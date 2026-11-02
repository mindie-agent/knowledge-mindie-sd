---
schema: mindie-graded-unit/1
id: unit-mindie-sd-cache-agent-ditcache-attentioncache-calibration-workflow
title: "MindIE-SD CacheAgent DiTBlockCache 与 AttentionCache 跨步复用标定工作流"
tier: L3
category: workflow
evidence_strength: software_contract
domain_path: mindie-sd/dit-compute/sparse-and-cache
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-dit-compute-sparse-cache-and-quant-selection-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "The 'block_end' must <= 'blocks_count'"
    - "The 'step_end' must >= 'step_start'"
    - "Method .* is not supported"
  keywords:
    - CacheAgent
    - CacheConfig
    - DiTBlockCache
    - AttentionCache
    - dit_block_cache
    - attention_cache
    - step_interval
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/cache_agent/cache_agent.py
  source_digest: bb5cdca8508cd7e80f2f52f46d95fa6f1ad64683c11c8e2a1d656d8d2e711ccd
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/cache_agent/cache_agent.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/cache_agent/dit_block_cache.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/cache_agent/attention_cache.py
  supersedes: []
revision: 1
---

## 工作流目标与适用条件

本工作流规定在 MindIE-SD 中使用 `CacheAgent`（支持 `"dit_block_cache"` 整块残差缓存与 `"attention_cache"` 注意力子层缓存）实现去噪时间步与 Transformer 层区间跨步复用的配置校验、计数器推进与精度-吞吐标定流程。

## 端到端执行步骤

1. **阶段一：构造 `CacheConfig` 并通过 `CacheAgent._check_config` 静态门禁**：
   - 明确指定 `method` 为 `"dit_block_cache"` 或 `"attention_cache"`，设置模型总块数 `blocks_count > 0` 与总去噪步数 `steps_count > 0`。
   - 配置生效步区间 `[step_start, step_end)`、复用周期 `step_interval > 1` 以及生效层区间 `[block_start, block_end)`：
     - 必须满足 `0 <= step_start <= step_end`、`0 <= block_start <= block_end`；在 `"dit_block_cache"` 模式下额外强制 `block_end <= blocks_count`。
     - 若 `step_start >= steps_count`、`step_end == step_start`、`block_start >= blocks_count`、`block_end == block_start` 或 `step_interval == 1`，`CacheAgent.apply` 会安全退化为直通调用 `function(*args, **kwargs)`。
2. **阶段二：接入前向调用链与步/层计数器生命周期管理**：
   - **DiTBlockCache 接入**：将 Transformer Block 列表的循环步或单个 Block 的残差分支封装进 `cache_agent.apply(block_fn, ...)`。当当前步 `cur_step` 处于 `[step_start, step_end)` 且 `(cur_step - step_start) % step_interval == 0` 时，完整执行所有 Block 并记录 `[block_start, block_end)` 区间首尾的残差增量 `delta = hidden_states_out - hidden_states_in`；在随后的 `step_interval - 1` 个复用步中，直接跳过 `[block_start, block_end)` 内的 Block 计算并加上缓存的 `delta`。
   - **AttentionCache 接入**：在每个 Block 内部将 Self-Attention / Cross-Attention / FFN 子函数包装进 `cache_agent.apply`，在复用步直接返回上一基准步缓存的子层输出张量。
   - **CFG 双分支与多请求状态重置**：当开启非并行 CFG（单卡串行跑 conditional 与 unconditional 两次前向，或 batch=2 拼接）时，必须确保 `CacheConfig` 的缓存槽位按分支隔离或 `steps_count` 准确对齐实际前向调用次数；每次推理请求（Pipeline call）开始前必须重置 `cur_step` / `cur_block` 计数与缓存引用。
3. **阶段三：敏感头尾步保护与分档标定**：
   - **保护高曲率头尾步**：去噪前 15%~20% 步（构图期）与最后 10% 步（高频细节收敛期）特征变化剧烈，必须通过 `step_start` 与 `step_end` 排除在缓存区间之外。
   - **保护首尾 Transformer Block**：首尾各 2~4 个 Block 承担输入嵌入对齐与输出投影准备，在 `dit_block_cache` 中应设置 `block_start >= 2`、`block_end <= blocks_count - 2`。
   - **递增标定**：从保守档（`step_interval=2`、中间 50% Block）开始，对比与无缓存基线的 PSNR（目标 `>= 28 dB`）与 LPIPS（目标 `<= 0.08`），达标后再尝试 `step_interval=3` 或扩大 `[block_start, block_end)`。

## 验收判据与常见故障排查

- **生效判据**：实测 E2E 去噪步均耗时在命中步下降比例与 `(block_end - block_start) / blocks_count` 相符，且未触发直通退化分支。
- **故障排查**：若生成图像出现重影或色彩断层，优先检查是否在 `cur_step == 0` 附近过早开启了缓存，或 CFG 正负分支共享了同一个 `CacheAgent` 实例导致 unconditional 残差覆盖了 conditional 残差。
