---
schema: mindie-graded-unit/1
id: unit-mindie-sd-timestep-aware-dynamic-quantization-calibration-sop
title: "MindIE-SD W8A8_TIMESTEP 时间步感知量化与 TimestepManager 动态切换规程"
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
    - "TimestepManager.set_timestep_idx"
    - "t_idx >= max_timestep"
  keywords:
    - W8A8_TIMESTEP
    - W8A8TimeStepQuantLinear
    - TimestepManager
    - TimestepPolicyConfig
    - set_timestep_idx
    - get_timestep_idx
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/config.py
  source_digest: b17edb4a43c959fe07d42e6288546f1c6a7100e978f08c14954e232afcf4d3eb
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/config.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/quantization/utils.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当扩散模型去噪过程中激活分布随时间步 `t` 剧烈漂移，单一静态缩放因子会导致构图期或细节收敛期失真，需要使用 `QuantAlgorithm.W8A8_TIMESTEP`（`W8A8TimeStepQuantLinear`）或 `TimestepPolicyConfig`（在线量化时间步策略）并通过 `TimestepManager` 动态同步当前去噪步索引时，必须遵循本规程。

## 核心操作规程

1. **理解时间步量化契约（`W8A8TimeStepQuantLinear`）**：
   - 在 `W8A8_TIMESTEP` 模式下，权重量化参数或激活平滑/缩放张量（`input_scale`、`input_offset`、`deq_scale`、`quant_bias`）包含时间步维度（或分段桶索引 `max_timestep`）。
   - 前向推理时，`W8A8TimeStepQuantLinear.forward` 通过 `TimestepManager.get_timestep_idx()` 获取当前时间步索引 `t_idx`，并从多步缩放表中切取对应的 `[t_idx]` 缩放因子喂入 `torch_npu.npu_quant_matmul`。
2. **在去噪主循环中严密同步 `TimestepManager`**：
   - 在每个去噪步进入 Transformer 前向传播**之前**，必须在 Pipeline 循环内显式调用 `TimestepManager.set_timestep_idx(step_idx)`（或确认 `quantize()` 注入的顶层前向 `wrapper` 已自动推进时间步索引）。
   - **边界校验**：`t_idx` 必须为非负整数且严格满足 `0 <= t_idx < t_scale_len`；若实际推理步数 `num_inference_steps` 与离线校准导出时的步数不一致，必须按比例映射或查表对齐到校准桶索引，严禁直接传入原始连续浮点噪声日程值（如 `t = 999.0`）充当数组下标。
3. **配置在线量化时间步混合策略（`TimestepPolicyConfig`）**：
   - 当使用 `OnlineQuantConfig`（如 `W8A8_MXFP8` 或 `W4A4_MXFP4` 在线量化）时，可通过 `TimestepPolicyConfig` 指定在不同去噪步区间切换量化精度（例如在敏感的前/后若干步回退至高精或 `W8A8_MXFP8`，在中间步启用激进的 `W4A4_MXFP4`）。

## 验证与反模式警报

- **验证判据**：在去噪循环第 `0` 步与中间步打印 `TimestepManager.get_timestep_idx()`，确认其随步数单调递增并在新请求开始时归零；相比非时间步感知的静态 W8A8，图像色彩偏移与边缘伪影显著改善（PSNR 提升 `>= 1.5 dB`）。
- **反模式警报**：
  - 严禁在开启 `W8A8_TIMESTEP` 时遗漏 `TimestepManager.set_timestep_idx` 更新（会导致 `t_idx` 保持 `None` 或停留在常数步，抛出 `ParametersInvalid` 或全步误用同一套缩放因子）。
  - 严禁在单卡串行执行 CFG 正负两次前向时让内部包装器把单个去噪步误计为两次时间步推进，导致后半程 `t_idx >= max_timestep` 越界。
