---
schema: mindie-graded-unit/1
id: unit-mindie-sd-w8a8-mxfp8-w4a4-mxfp4-calibration-and-accuracy-recovery-workflow
title: "MindIE-SD W8A8/MXFP8/W4A4/MXFP4 线性层与注意力分级量化及精度回退工作流"
tier: L3
category: workflow
evidence_strength: software_contract
domain_path: mindie-sd/quantization/contracts
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-quantization-device-byte-contract-and-alignment-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "W4A4 MXFP4 weight must be torch.float8_e4m3fn or torch.uint8"
    - "npu_dynamic_mx_quant must support axis, scale_alg and dst_type_max"
  keywords:
    - W8A8MXFP8QuantLinear
    - W4A4MXFP4QuantLinear
    - W4A4MXFP4DualQuantLinear
    - W4A4MXFP4FourOverSixQuantLinear
    - FP8RotateQuantFA
    - MXFP4QuantFA
    - quant_four_over_six_a5
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/layer.py
  source_digest: 315f3ac4215b401cbf248b8089f36aa411dba6497e3c67d88f3898bca401e0aa
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/layer.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/quantization.md
  supersedes: []
revision: 1
---

## 工作流目标与适用条件

本工作流规定在 MindIE-SD 中对 DiT 线性层与 FlashAttention 执行 `W8A8`、`W8A8_MXFP8`、`W4A4_DYNAMIC`、`W4A4_MXFP4`（含 `DUALSCALE` 与 A5 `FOUROVERSIX` 4/6 自适应量化）以及 `FP8RotateQuantFA` / `MXFP4QuantFA` 的分级选档、算子契约核验与逐层精度回退流程。

## 端到端执行步骤

1. **阶段一：按硬件代际与算子形态初选量化档位**：
   - **A2 / A3 平台**：线性层优先选用 `W8A8_DYNAMIC`（Per-Token 动态激活量化）或 `W8A8_TIMESTEP`；显存受限时选用 `W8A16` / `W4A16`（`WeightQuantLinear`）。
   - **A5 平台（支持 Microscaling MXFP8 / MXFP4）**：
     - 稳健高吞吐档：线性层选用 `W8A8_MXFP8`（`W8A8MXFP8QuantLinear` / `W8A8MXFP8OnlineQuantLinear`，每 32 元素共享 `E8M0` 指数缩放），注意力选用 `FP8RotateQuantFA` 或 `MXFP8RotateQuantFA`（可选 `FP8FAMode.HIGH_PRECISION` 或高吞吐 `FP8FAMode.C8V16_TILING512`）。
     - 极限低比特档：线性层选用 `W4A4_MXFP4_DYNAMIC`、`W4A4_MXFP4_DUALSCALE`（双级缩放兼顾通道离群值与微块精度）或 `W4A4_MXFP4_FOUROVERSIX_DYNAMIC`（调用 `quant_four_over_six_a5` 算子对尾轴 32 对齐的 BF16 激活做 4/6 自适应 FP4 打包量化）。
2. **阶段二：校验 Microscaling 与 NZ 格式转换契约**：
   - 在 `W4A4_MXFP4` 中，权重必须为 `torch.float8_e4m3fn`（经 `torch_npu.npu_dtype_cast` 转为 `float4_e2m1fn_x2`）或已打包的 `torch.uint8`；开启 `use_nz=True` 时通过 `torch_npu.npu_format_cast(..., 29)` 转为分形 `FRACTAL_NZ` 布局以提升 Cube 搬运效率。
   - 在 `_four_over_six_quant` 中，输入激活必须为 `torch.bfloat16` 且尾维被 `32` 整除，产出尾维减半的 `uint8` 打包 FP4 载荷与形状为 `[..., dim/64, 2]` 的 `uint8 E8M0` 缩放因子。
3. **阶段三：敏感层隔离（Layer Exclusion）与精度回退闭环**：
   - **默认免量化白名单**：时间步嵌入投影（`time_embed` / `adaLN_modulation`）、输入 Patch 嵌入（`patch_embed`）、最终输出投影层（`proj_out` / `norm_out`）以及 MoE 路由门控层（`gate`）严禁纳入 W4A4 或粗粒度静态 W8A8，应在量化描述文件或 `OnlineQuantConfig.exclude_layers` 中保持 `FLOAT`（BF16/FP16）。
   - **二分法定位失真层**：若整网开启 `W4A4_MXFP4` 后精度门禁（如 LPIPS 或 VBench）不达标：
     1. 先将 `FFN.down_proj`（`fc2`）及首尾各 2 个 Transformer Block 从 `W4A4_MXFP4` 回退至 `W8A8_MXFP8`；
     2. 若注意力层开启了 `MXFP4QuantFA`，优先切换为 `MXFP4FourOverSixQuantFA` 或回退至 `FP8RotateQuantFA(fp8_fa_mode="high_precision")`；
     3. 每回退一组层，在固定随机种子下复测单图/单视频 PSNR 与单步耗时，保留满足质量门且收益最高的混合配置。

## 验收判据与常见故障排查

- **验收判据**：量化后模型显存占用与 `QuantLinear` 预期字节数一致，Cube MatMul 耗时相对 BF16 基线显著下降，且生成质量通过三级精度门禁。
- **故障排查**：若抛出 `npu_dynamic_mx_quant must support axis, scale_alg and dst_type_max`，说明当前 CANN/torch_npu 版本不满足 MXFP4 C7 缩放算法（`scale_alg=2, dst_type_max=7.25`）要求，需升级算子包或回退至标准 `W8A8_MXFP8`。
