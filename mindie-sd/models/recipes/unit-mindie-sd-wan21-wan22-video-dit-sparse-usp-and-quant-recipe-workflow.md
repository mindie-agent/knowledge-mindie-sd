---
schema: mindie-graded-unit/1
id: unit-mindie-sd-wan21-wan22-video-dit-sparse-usp-and-quant-recipe-workflow
title: "MindIE-SD Wan2.1/Wan2.2 视频生成模型（稀疏注意力 × USP × 量化 × 3D VAE）联合调优工作流"
tier: L3
category: workflow
evidence_strength: reproduced_empirical
domain_path: mindie-sd/models/recipes
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-usp-ulysses-ring-and-cfg-sequence-parallel-axioms
  - unit-mindie-sd-rainfusion-rf-v2-rf-v3-layout-and-block-mask-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "MIE-USP-003"
    - "12 % 8 != 0"
  keywords:
    - Wan2.1
    - Wan2.2
    - WanAttentionBlock
    - rotary_position_embedding
    - rf_v2
    - rf_v3
    - dit_block_cache
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/wan/parameter_config.md
  source_digest: fe8a90af8299476ec36ea7a1a8f9d518a389e3b9e2a810f4041e067ab72506b4
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/wan/parameter_config.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/dummy_run/model/wan_model.py
  supersedes: []
revision: 1
---

## 工作流目标与适用条件

本工作流规定在昇腾 800I A2/A3 及 A5 上针对 Wan2.1 / Wan2.2（1.3B / 14B 文生视频 T2V 与图生视频 I2V，含双噪声专家架构）组合应用 RoPE + RMSNorm 算子融合、CFG × Ulysses 序列并行、RainFusion V2/V3 3D 块稀疏、DiTBlockCache 以及 W8A8/MXFP8 量化的端到端调优配方。

## 端到端执行步骤

1. **阶段一：无损算子融合与 3D RoPE 替换（S2 无损收益层）**：
   - 将 WanAttentionBlock 中的 Q/K RMSNorm 替换为 `mindiesd.ops.rms_norm`，并将 3D 视频旋转位置编码替换为融合算子 `mindiesd.rotary_position_embedding`（传入预计算的 cos/sin 表与 `rotated_mode`），消除逐元素 Split/Concat/Mul 胶水算子。
   - 对 Wan2.1-14B 单卡或少卡高分辨率（720P, 81 帧）场景，若 HBM 容量受限，优先开启 `enable_offload(transformer, transformer.blocks, min_reserved_blocks_count=2)` 解锁运行。
2. **阶段二：8 卡 CFG × Ulysses 正交并行配置（S3 多卡扩展层）**：
   - 在 8 卡（如 A2/A3 同节点 8 卡）开启 Classifier-Free Guidance 时，推荐采用 `cfg_size = 2, ulysses_size = 4`（或纯 `ulysses_size = 8` 当头数 `num_heads = 40` 可被 8 整除时）：
     -注意 Wan2.1-14B 的 `num_heads = 40`：可被 `2, 4, 8` 整除，天然支持 4 卡或 8 卡纯 Ulysses；
     - Wan2.1-1.3B 的 `num_heads = 12`：**不能被 8 整除**（`12 % 8 != 0`），因此在 8 卡下必须采用 `cfg_size = 2 × ulysses_size = 4`（`12 % 4 == 0`）或 `ulysses_size = 4 × ring_size = 2`，严禁对 1.3B 模型直接设置 `ulysses_size = 8`。
3. **阶段三：叠加 RainFusion V2/V3 3D 块稀疏与 `DiTBlockCache`（S4 有损计算减免层）**：
   - **块稀疏配置**：对 Self-Attention 启用 `sparse_attention`（A2/A3 选 `sparse_type="rf_v2"` 或 `"rf_v3"`，A5 自动路由至 `"rf_v3"`），传入潜空间帧高宽 `latent_shape_q = (F_lat, H_lat, W_lat)`，设置 `sparsity = 0.5 ~ 0.65`、`protect_first_frame = True`、前 20% 去噪步（如 50 步中的前 10 步）保持稠密全注意力以稳定视频全局构图与动态轨迹。
   - **跨步缓存配置**：启用 `CacheAgent(CacheConfig(method="dit_block_cache", step_start=10, step_end=45, step_interval=2, block_start=2, block_end=len(blocks)-2))`，避开前 10 步与最后 5 步。
   - **线性层量化**：对 `WanAttentionBlock` 内的 QKV 投影与 FFN 启用 `W8A8_DYNAMIC`（A2/A3）或 `W8A8_MXFP8`（A5），对 Wan2.2 双专家（高噪/低噪专家）分别绑定各自的量化描述符；保持时间步调制层 `scale_shift_table` / `time_embedding` 为 BF16。
4. **阶段四：WanVAE 3D 因果解码优化**：
   - 对 81 帧长视频潜变量解码启用潜帧边界特征缓存（Feature Cache）切分，避免 8 卡重复全量解码或单次大张量触发 HBM OOM。

## 验收判据与常见故障排查

- **验收判据**：720P 81 帧 50 步推理端到端耗时在 8 卡下达到线性加速叠加上稀疏/缓存/量化的乘性收益，且 VBench 视频质量评分与语义一致性处于验收阈值内。
- **故障排查**：若 Wan2.1-1.3B 在 8 卡时报 `USPShapeError [MIE-USP-003]`，检查是否误将 `ulysses_size` 设为了 `8`（`12 % 8 != 0`）。
