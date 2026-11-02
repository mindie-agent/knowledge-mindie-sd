---
schema: mindie-graded-unit/1
id: unit-mindie-sd-quantize-model-descriptor-and-quantlinear-replacement-sop
title: "MindIE-SD quantize() 模型描述符解析、Safetensors 权重挂载与 QuantLinear 原地替换规程"
tier: L2
category: sop
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
    - "MAX_WEIGHT_SIZE"
    - "parse_quant_description"
  keywords:
    - quantize
    - OnlineQuantConfig
    - WeightQuantLinear
    - smooth_quantize_w8a8
    - replace_rank_suffix
    - build_online_fa_rot_weights
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/quantize.py
  source_digest: 14810594e51c5caae3c278cce17b5218d67f26a605161c5aada4ffb59006efb8
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/quantize.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/mode.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当调用 `mindiesd.quantization.quantize(model, quant_des_path, ...)`（离线描述符 + Safetensors 量化权重模式）或传入 `OnlineQuantConfig`（无需外部校准权重的在线动态量化模式）将 `nn.Linear` 与注意力模块原地替换为对应的 `QuantLinear` / `QuantFA` 子类时，必须遵循本规程。

## 核心操作规程

1. **离线量化描述符解析与多卡 Rank 后缀适配（`parse_quant_description` & `replace_rank_suffix`）**：
   - 离线量化入口接收 JSON 描述符路径 `quant_des_path`。若模型已启用张量并行（TP）且按卡切分导出描述符，使用 `replace_rank_suffix` 自动匹配当前 Rank 对应的描述符与 `.safetensors` 权重文件。
   - 描述符文件大小受 `MAX_WEIGHT_SIZE` 与 `file_utils.standardize_path` 安全校验保护；解析后按层名键模式（`{layer_name}.weight`、`{layer_name}.linear.weight`、`{layer_name}.fa_q.scale`、`{layer_name}.quant_type`）匹配每层的 `QuantAlgorithm`。
2. **原地释放浮点权重与构造 `QuantLinear`（防显存双倍峰值）**：
   - 在 `w8a16_quantize` 与 `smooth_quantize_w8a8` 替换过程中，通过 `extract_constructor_args(layer, quant_cls)` 提取 `in_features`、`out_features`、`bias` 等构造参数。
   - **关键显存纪律**：在实例化 `quant_cls(**init_params)` 之前，必须先执行 `del layer.weight` 和 `del layer.bias` 释放原始浮点 Parameter 引用，防止替换期间原始 BF16 权重与新量化权重同时驻留导致 Host/NPU 内存溢出。
   - 当存在 SmoothQuant 通道抑制系数 `{name}.div.mul_scale` 时，自动提取 `mul_scale` 并将权重前缀重定向至 `{name}.linear`。
3. **在线动态量化模式（`OnlineQuantConfig`）与 FA 旋转矩阵注入**：
   - 当未提供离线校准权重但传入 `OnlineQuantConfig` 时，`quantize()` 遍历模型子树，通过 `match_layer_config` 跳过被排除的敏感层，将其余 `nn.Linear` 替换为对应的 `*OnlineQuantLinear`（如 `W8A8OnlineQuantLinear`、`W8A8MXFP8OnlineQuantLinear`、`W4A4MXFP4OnlineQuantLinear`），在首次前向或加载时直接对模型自身浮点权重做在线量化转换。
   - 对匹配到的注意力层（`match_fa_layer`），通过 `build_online_fa_rot_weights` 构建正交旋转矩阵并绑定 `FP8RotateQuantFA` / `MXFP8RotateQuantFA` / `MXFP4QuantFA`。

## 验证与反模式警报

- **验证判据**：调用 `quantize()` 后遍历 `model.named_modules()`，目标线性层类型应全部变为预期的 `*QuantLinear`，且未出现在描述符/配置中的保护层保持 `nn.Linear`。
- **反模式警报**：
  - 严禁先把整个超大 BF16 模型 `.to("npu")` 塞满 HBM 后再尝试不删原权重的非原地量化加载；应遵循“CPU 上结构构建与 `quantize()` 替换 -> 挂载量化权重 -> `.to('npu')` 或启用 `enable_offload`”的顺序。
