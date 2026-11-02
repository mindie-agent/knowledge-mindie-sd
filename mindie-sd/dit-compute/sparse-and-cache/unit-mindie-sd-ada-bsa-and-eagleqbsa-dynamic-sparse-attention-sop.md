---
schema: mindie-graded-unit/1
id: unit-mindie-sd-ada-bsa-and-eagleqbsa-dynamic-sparse-attention-sop
title: "MindIE-SD AdaBSA 自适应 CDF 块稀疏与 EagleQBSA/FP8 旋转量化块稀疏规程"
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
    - "ada_block_sparse_attention / get_estimate_mask (ada_bsa) are not supported on A5 devices"
    - "eagle qbsa per-block quant only supports 4D qkv"
  keywords:
    - ada_block_sparse_attention
    - sparse_block_estimate
    - get_estimate_mask
    - _eagle_qbsa_quant_qkv
    - _fp8_quant_qkv
    - cdf_threshold
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/sparse_flash_attn_ada_bsa.py
  source_digest: d58788e3ae696722f3ec5e9448046811d85dcc7442342cf2f6f701b5467d8765
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/sparse_flash_attn_ada_bsa.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当 DiT 模型需要在注意力层根据累积分布函数（CDF）阈值动态决定每个 Query 块保留的 Key/Value 块数量（`ada_bsa`），或在 `rf_v3` 块稀疏注意力中进一步叠加低比特量化（EagleQBSA 块级 INT8 Q/K + 通道级 FP8 V，或 Hadamard 旋转 + 块级 FP8 Q/K/V）以突破注意力 Cube 吞吐上限时，必须遵循本规程。

## 核心操作规程

1. **AdaBSA 自适应 CDF 块稀疏（`sparse_flash_attn_ada_bsa.py`）**：
   - **第一阶段（掩码估计）**：调用 `get_estimate_mask` -> 底层算子 `ops.sparse_block_estimate`，以 `stride=8`、`sparse_size=128`、`scale_value=scale / stride` 对 Q/K 进行下采样估计，结合 `cdf_threshold`（累积概率阈值）与 `row_sparse=1.0 - sparsity` 动态生成块掩码 `smask` 与每行有效块计数表 `sct`（`sparse_count_table`），默认开启 `keep_sink=True` 与 `keep_recent=True` 保留首块与邻近块。
   - **第二阶段（稀疏计算）**：将 `(smask, sct)` 传入 `ada_block_sparse_attention` -> 底层算子 `ops.ada_block_sparse_attention`。
   - **A5 硬件守卫**：`ada_bsa` 目前不支持 A5 设备（调用 `_raise_if_a5()` 会显式抛出 `ParametersInvalid`），在 A5 上必须改用 `sparse_type="rf_v3"` 或稠密注意力路径。
2. **EagleQBSA 混合精度量化块稀疏（`_eagle_qbsa_quant_qkv`）**：
   - 在 `rf_v3` 中启用 EagleQBSA 时，输入 Q/K/V 统一转置为连续的 `BNSD` 布局。
   - 对 `Q` 与 `K` 调用 `_perblock_quant(tensor, block_size=64, dst_type=torch.int8)`：当序列长 `S` 不是 `64` 的整数倍时，先沿序列维末尾零填充对齐，再按 `[B, N, ceil(S/64), 64*D]` 调用 `torch_npu.npu_dynamic_quant` 量化为 `INT8` 并切片回原始长度 `S`，产出 `fp32` 块级缩放因子 `[B, N, ceil(S/64)]`。
   - 对 `V` 先沿最后两维转置 `v.transpose(-1, -2)`（使序列维归一、按通道维 `D` 对齐），调用 `torch_npu.npu_dynamic_quant(..., dst_type=torch_npu.float8_e4m3fn)` 执行 Per-Channel `FP8_E4M3FN` 动态量化，再转置回 `BNSD`。
3. **Hadamard 旋转 + FP8 块稀疏注意力（`_fp8_quant_qkv`）**：
   - 通过 `_get_rot_matrices(device, dtype, head_dim)` 生成并按 `(str(device), dtype, head_dim)` 线程安全缓存缩放 Sylvester Hadamard 正交矩阵（阶数取 `1 << (head_dim - 1).bit_length()` 并切片至 `[head_dim, head_dim]`）。
   - 对 `Q` 和 `K` 施以相同的正交旋转以抹平通道间离群值，随后按 `block_size_q=128`、`block_size_kv=256` 对 Q/K/V 进行块级 FP8 动态量化。

## 验证与反模式警报

- **验证判据**：`get_estimate_mask` 输出的 `sct` 均值应随 `cdf_threshold` 减小或 `sparsity` 增大而单调下降；EagleQBSA 与 FP8 旋转路径在非对齐序列长 `S` 下无越界报错且尾块激活无 NaN/Inf。
- **反模式警报**：
  - 严禁在每次前向传播中重新构造 Hadamard 矩阵并做 Host-to-Device 拷贝，必须复用 `_ROT_MATRIXS` 缓存。
  - 严禁在未做 Hadamard 旋转或 Smooth 均值中心化的情况下直接对含剧烈离群值的 DiT Q/K 做粗粒度低比特量化，否则会导致 Softmax 分布坍缩。
