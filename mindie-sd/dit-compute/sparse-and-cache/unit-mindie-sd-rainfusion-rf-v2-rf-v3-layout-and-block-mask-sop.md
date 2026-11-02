---
schema: mindie-graded-unit/1
id: unit-mindie-sd-rainfusion-rf-v2-rf-v3-layout-and-block-mask-sop
title: "MindIE-SD RainFusion V2/V3 3D 时空窗重排、分块池化掩码与首帧保护规程"
tier: L2
category: sop
evidence_strength: software_contract
domain_path: mindie-sd/dit-compute/sparse-and-cache
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-custom-csrc-ascendc-triton-catlass-operator-dispatch-axioms
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "sparse_flash_attn_rf_v2 (rain_fusion_attention) is not supported on A5 devices"
    - "ParametersInvalid"
  keywords:
    - rain_fusion_attention
    - bsa_sparse_attention_v3
    - rf_v2
    - rf_v3
    - get_blockwise_mask
    - rearrange_with_remaining
    - protect_first_frame
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py
  source_digest: f01ab7d03120ebfc9b453165df4672d840bd7fc9e51824e622ea0fe97684c811
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当视频生成扩散模型（如 Wan2.1/Wan2.2、HunyuanVideo）的长序列 3D 潜空间注意力采用 RainFusion V2（`sparse_type="rf_v2"`）或 RainFusion V3（`sparse_type="rf_v3"`）块稀疏加速，需要配置 `latent_shape_q`/`latent_shape_k`、3D 时空窗重排（`do_tensor_rearrange_pooling` / `do_multi_span_tensor_rearrange_pooling`）、均值池化掩码生成（`get_blockwise_mask`）以及首帧与文本 Token 全注意力保护策略时，必须遵循本规程。

## 核心操作规程

1. **硬件平台与公共入口路由约束**：
   - 在 A5 硬件上，旧算子 `mindiesd.rain_fusion_attention`（`rf_v2`）不可直接调用（直接调用 `sparse_flash_attn_rf_v2` 将抛出 `ParametersInvalid`）。
   - 业务代码应始终通过统一入口 `mindiesd.layers.flash_attn.sparse_attention` 传入 `sparse_type="rf_v2"` 或 `"rf_v3"`；在 A5 设备上，公共入口会自动将 `rf_v2` 路由至 `sparse_flash_attn_rf_v3.bsa_sparse_attention_v3`。
2. **3D 时空窗重排与逆重排配对（`rearrange_with_remaining`）**：
   - 3D 视频潜空间张量 `(T, H, W)` 在一维序列中相邻位置的局部性较差。必须通过 `do_tensor_rearrange_pooling`（单跨度）或 `do_multi_span_tensor_rearrange_pooling`（多跨度）按 3D 窗口将时空邻近 Token 聚集到同一个 `pool_size=128` 块内。
   - 当 `(T, H, W)` 无法被窗口尺寸整除时，`rearrange_with_remaining` 会对主块与尾部余数块分别重排并拼接；注意力计算完成后，必须调用 `do_tensor_inv_rearrange` 严格恢复原始 `(T, H, W)` 序列顺序，严禁漏掉逆重排导致画面网格撕裂。
3. **均值池化粗粒度打分与块掩码生成（`avgpool` + `get_blockwise_mask`）**：
   - 使用 `avgpool(input_tensor, pool_size=128, input_layout)` 对重排后的 `Q/K` 沿序列维做块均值池化（同时处理 `num_full_blocks` 与尾部 `tail_size`）。
   - 在块级序列维度上计算粗粒度 Softmax 得分矩阵 `score_matrix`，按 `sparsity` 比例通过 Top-K 选出高权重块。
   - **首帧与文本保护（`protect_first_frame=True` & `txt_len > 0`）**：视频首帧（前 `hq * wq` 个视觉 Token 对应的块）以及多模态文本序列（`txt_len` 对应的块）对全局语义和时序一致性起锚定作用，必须在 `get_blockwise_mask` 中强制置为 `True`（全可见），不参与稀疏丢弃。
4. **掩码索引转换与内核调用**：
   - 在 `rf_v2` 路径中，通过 `get_mask_index(mask)` 将布尔掩码转换为稠密前置的列索引矩阵 `select_idx` 及有效块计数 `select_num_idx`，传入 `ops.rain_fusion_attention`。
   - 在 `rf_v3` 路径中，直接使用布尔或打包掩码配合量化/非量化 BSA 内核（`bsa_sparse_attention_v3`）执行块稀疏计算。

## 验证与反模式警报

- **验证判据**：开启 `rf_v2` / `rf_v3`（如 `sparsity=0.5~0.75`，`first_layers_fp` 与 `first_steps_fp` 设为约 10%~25% 全量步/层）后，单步注意力耗时显著下降，且首帧结构与文本语义对齐度（VBench / CLIP / LPIPS）无明显退化。
- **反模式警报**：
  - 严禁传入错误的 `latent_shape_q=(tq, hq, wq)` 导致 `tq * hq * wq + txt_len != seqlen`，否则 3D 窗口重排与首帧边界计算会错位。
  - 严禁在文生视频任务中关闭 `protect_first_frame` 或将 `txt_len` 误设为 `0`（当存在拼接文本 Token 时），否则会导致生成的视频丢失提示词语义或首尾帧闪烁崩塌。
