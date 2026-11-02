---
schema: mindie-graded-unit/1
id: unit-mindie-sd-flux1-qwen-image-and-sd3-mmdit-optimization-recipe-sop
title: "MindIE-SD Flux.1 / Qwen-Image / SD3 MMDiT 双流图像扩散模型联合调优规程"
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: mindie-sd/models/recipes
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-usp-ulysses-ring-and-cfg-sequence-parallel-axioms
  - unit-mindie-sd-dit-compute-sparse-cache-and-quant-selection-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "JointTransformerBlock"
    - "SingleTransformerBlock"
  keywords:
    - FLUX.1-dev
    - Qwen-Image
    - MMDiT
    - AdaLayerNormZero
    - cache-dit
    - DiTBlockCache
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/cache-dit/FLUX.1-dev.md
  source_digest: 9907e9100e7db451c1a9059dcf1e7957350dfcf2db4352c5665a6717eea19352
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/cache-dit/FLUX.1-dev.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/qwen-image-2.1/infer.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:examples/dummy_run/model/flux_model.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当优化基于 MMDiT（Multimodal Diffusion Transformer，包含双流 `JointTransformerBlock` 与单流 `SingleTransformerBlock`）架构的高分辨率文生图与图像编辑模型（如 FLUX.1-dev、Qwen-Image / Qwen-Image-Edit、Stable Diffusion 3.5）时，必须遵循本规程。

## 核心操作规程

1. **MMDiT 双流与单流块差异化缓存策略（`CacheAgent` / `cache-dit`）**：
   - FLUX.1-dev 与 Qwen-Image 前半部分为图像流（`hidden_states`）与文本流（`encoder_hidden_states`）联合注意力的双流块（如 FLUX 19 层双流块），后半部分为拼接后的单流块（如 FLUX 38 层单流块）。
   - 使用 `DiTBlockCache` 时，由于双流块同时更新视觉与文本两个残差流，而单流块仅更新合并流，**必须对双流 ModuleList 与单流 ModuleList 分别实例化独立的 `CacheAgent`**，严禁将具有不同返回值签名的双流块与单流块混入同一个 `DiTBlockCache` 实例。
2. **序列并行（USP）中的文本流与图像流拼接对齐**：
   - 在 MMDiT 联合注意力中，Query/Key/Value 由定长文本序列 $S_{\text{txt}}$（如 T5/Qwen2.5-VL 编码输出）与图像 Patch 序列 $S_{\text{img}}$ 拼接而成。
   - 启用 Ulysses 序列并行时：
     - 方案 A（仅切分图像序列 $S_{\text{img}}$，文本序列 $S_{\text{txt}}$ 全卡复制并在进入注意力前按头切分拼接）：避免对短文本序列做多余的序列填充；
     - 方案 B（拼接后统一做 Ulysses All-to-All）：必须确保总序列长度 $S_{\text{txt}} + S_{\text{img}}$ 满足 `U` 整除（否则按 `S_PAD` 规则对齐并在后处理时精确剥离文本与 Padding 区间）。
   - 注意力头数核验：FLUX.1-dev 为 `24` 头（支持 `U = 2, 4, 8`），Qwen-Image 需核对 `H_q` 与 `H_kv` 是否满足 `U` 整除。
3. **AdaLN 调制融合与分级量化配置**：
   - **算子融合**：MMDiT 每层包含密集的 `AdaLayerNormZero` / `AdaLayerNormContinuous`（由 timestep embedding 经 SiLU + Linear 预测 6 组或 2 组 `shift, scale, gate` 参数），优先开启 `mindie_sd_backend` 编译图融合或使用融合 `layernorm + scale + shift` 模式消减 Vector 访存开销。
   - **线性层与注意力量化**：
     - 对双流块的 `to_q/to_k/to_v/to_out`、`add_q_proj/add_k_proj/add_v_proj` 及单流块的 `proj_mlp` / `proj_out` 启用 `W8A8_DYNAMIC`（A2/A3）或 `W8A8_MXFP8` / `W4A4_MXFP4_DUALSCALE`（A5）。
     - **保护敏感层**：必须将 `time_text_embed`、`context_embedder`、`x_embedder` 以及各层 AdaLN 的 `norm1.linear` / `norm1_context.linear` 排除在 4-bit 量化之外（保持 BF16 或高精度 W8A8），防止调制缩放系数 `gate_msa / scale_mlp` 量化失真导致整幅图像出现高频噪点网格。

## 验证与反模式警报

- **验证判据**：1024×1024 或 2048×2048 分辨率下，开启 USP + `CacheAgent` + `W8A8/MXFP8` 后单图生成延迟显著下降，且文字渲染准确率（特に Qwen-Image 与 FLUX 的中英文排版）与色彩饱和度无退化（PSNR `>= 28 dB`）。
- **反模式警报**：
  - 严禁对 AdaLN 调制投影层（`adaLN_modulation`）强开粗糙的 `W4A4` 量化，该层参数量极小但直接乘性控制整层激活幅度，量化收益极小而精度破坏极大。
