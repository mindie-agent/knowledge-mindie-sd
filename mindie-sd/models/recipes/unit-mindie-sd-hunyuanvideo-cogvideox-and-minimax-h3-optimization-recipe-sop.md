---
schema: mindie-graded-unit/1
id: unit-mindie-sd-hunyuanvideo-cogvideox-and-minimax-h3-optimization-recipe-sop
title: "MindIE-SD HunyuanVideo / HunyuanImage-3.0 / MiniMax-H3 MoE 与多模态扩散调优规程"
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: mindie-sd/models/recipes
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-usp-ulysses-ring-and-cfg-sequence-parallel-axioms
  - unit-mindie-sd-cfg-parallel-group-init-and-allgather-kv-hybrid-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "EP All-to-All"
    - "Exposed Communication"
  keywords:
    - HunyuanVideo
    - HunyuanImage-3.0
    - MiniMax-H3
    - DyEPLB
    - AllGather-KV
    - distributed-layerwise-offload
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/minimax-h3/infer.md
  source_digest: b83863ed3af84feea282f54ffc4f9bbf4254140618b7a976212b73b84ada2bca
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/minimax-h3/infer.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/DyEPLB.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/dummy_run/model/hunyuan_image3_model.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当优化采用 MoE（混合专家）架构的扩散模型（如 HunyuanImage-3.0）、音视频联合或多阶段扩散模型（如 MiniMax-H3）以及大参数量 3D 视频生成模型（如 HunyuanVideo、CogVideoX）在昇腾多卡集群上的负载均衡、序列并行形态与量化叠加策略时，必须遵循本规程。

## 核心操作规程

1. **MoE 扩散模型动态专家负载均衡（`DyEPLB` — HunyuanImage-3.0）**：
   - 在 HunyuanImage-3.0 等包含稀疏 MoE FFN 层的扩散模型中，不同去噪时间步与不同空间区域的 Token 路由分布存在显著热点倾斜，导致专家并行（EP）下最慢卡拖慢全局 All-to-All 同步。
   - 启用动态专家并行负载均衡（`DyEPLB`）：
     - 在去噪步迭代过程中实时统计各专家激活负载热度；
     - 利用去噪相邻时间步路由分布高度平滑的特性，根据前序步统计结果动态重映射或复制高热度冗余专家（Expert Replication & Placement），消除跨卡木桶效应。
2. **GQA 与跨域拓扑下的复合序列并行（MiniMax-H3 & HunyuanVideo）**：
   - **GQA 头数约束与 `AllGather-KV × Ulysses` 翻转**：当模型采用小 `H_kv` 的 GQA 注意力，或 8 卡部署跨越了两个 4 卡 UB/HCCS 互连域（域间为低带宽 SYS/PCIe）时，域内 4 卡采用 `Ulysses`（高带宽执行 `all_to_all`），跨域 2 组采用 `AllGather-KV`（跨域仅同步体积较小的 K/V），并注入专用通信流掩盖（comm-stream masking）。
   - **并行 × 稀疏叠加（Seam 契约）**：在完成序列汇聚后，对视觉序列开启 `rf_v3`（含 EagleQBSA 或 FP8 旋转量化块稀疏），同时保证 `S_PAD` 对齐到 `128` 倍数以绕过 `HcclAlltoAllV` 不等长分片瓶颈。
3. **显存解锁与多模态解码协同**：
   - 当超大模型（如 HunyuanImage-3.0 或 MiniMax-H3）单卡 BF16 显存不足以开启更优的序列并行度时，先启用分布式层间卸载（DLO，优先选用 AllGather 路径而非慢速的 no-AllGather 路径）或 `W8A8_MXFP8` / `W4A4_MXFP4` 权重量化降低静态显存，释放出 HBM 余量后再复查并升级序列并行档位。
   - 对 MiniMax-H3 / HunyuanVideo 的尾部解码阶段：若采用 `tae-2d` 逐帧预览解码器则开启多卡帧轴等分并行；若采用 `taehv` 或 3D 因果 VAE 则开启 2~4 段潜帧边界状态携带切分（`apply_sliced`）。

## 验证与反模式警报

- **验证判据**：开启 `DyEPLB` 后 MoE 层跨卡 `EP All-to-All` 等待气泡显著收敛；开启复合序列并行 + 通信流掩盖后，Profiler 中未掩盖通信（`Exposed Communication`）占比降至极低水平。
- **反模式警报**：
  - 严禁在更换卡组（如从跨域 4 卡切换到同域 4 卡）后直接沿用旧卡组的基线耗时做加速比对比，换卡组必须在同一新卡组同窗重测基线与优化档。
