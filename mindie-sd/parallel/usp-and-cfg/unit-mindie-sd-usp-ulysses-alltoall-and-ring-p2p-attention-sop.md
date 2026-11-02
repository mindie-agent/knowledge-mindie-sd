---
schema: mindie-graded-unit/1
id: unit-mindie-sd-usp-ulysses-alltoall-and-ring-p2p-attention-sop
title: "MindIE-SD USP Ulysses All-to-All 与 Ring P2P 混合序列并行注意力规程"
tier: L2
category: sop
evidence_strength: software_contract
domain_path: mindie-sd/parallel/usp-and-cfg
inherits_from:
  - unit-mindie-sd-usp-ulysses-ring-and-cfg-sequence-parallel-axioms
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "MIE-USP-001"
    - "MIE-USP-002"
    - "MIE-USP-003"
    - "MIE-USP-004"
    - "the number of q heads must be divisible by the number of kv heads"
  keywords:
    - usp
    - ulysses_group
    - ring_group
    - comm_dtype
    - comm_quant_scope
    - USPShapeError
    - USPTopologyError
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/usp.py
  source_digest: b2a9986c49333a8c479fdcdb45613fc3a9d80f398142c91d573ad1e9e408748a
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/usp.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/usp.md
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当长序列视频或高分辨率图像 DiT 模型使用 `mindiesd.layers.usp` 执行 Ulysses（All-to-All 切头聚合序列）、Ring（P2P 环形传递 KV 并用 LSE 在线归约）或 2D 混合序列并行注意力，并配置通信张量量化（`fp8_e4m3` / `mxfp8`）或双缓冲工作区时，必须遵循本规程。

## 核心操作规程

1. **进程组与头数/形状前置校验（`MIE-USP-002` / `MIE-USP-003` 防护）**：
   - 输入 `q, k, v` 必须为同设备、同 dtype 的 4D 张量（`layout="BSND"` 或 `"BNSD"`），且 `B` 相同、`S_k == S_v`、`D_q == D_k == D_v`。
   - **GQA 与 Ulysses 整除律**：必须满足 `H_q % H_kv == 0`；当启用 `ulysses_group`（`U = world_size(ulysses_group) > 1`）时，`H_q` 与 `H_kv` 均必须能被 `U` 整除，否则抛出 `USPShapeError [MIE-USP-003]`。
   - 若同时传入 `ulysses_group` 与 `ring_group`，两者必须构成正交 2D 网格（当前 rank 必须属于两个通信组且组大小乘积等于 USP 总并行度），否则抛出 `USPTopologyError [MIE-USP-002]`。
2. **Ulysses All-to-All 与通信量化配置**：
   - 在 Ulysses 阶段，通过 `all_to_all_single` 将 `[B, S_local, H, D]` 重排为 `[B, U * S_local, H / U, D]`，计算完成后再反向 `all_to_all_single` 恢复 `[B, S_local, H, D]`。
   - **低比特通信压缩（`comm_dtype` / `comm_tensors` / `comm_quant_scope`）**：
     - 支持 `comm_dtype in ("none", "fp8_e4m3", "mxfp8")`；`comm_tensors` 可选 `{"q", "k", "v", "out"}` 的子集；`comm_quant_scope` 支持 `"exposed"`（仅对无法被计算掩盖的暴露通信张量如 Ulysses Q/K/V/Out 执行量化）或 `"all"`（对所有通信张量含 Ring KV 传递均执行量化）。
     - 开启 `fp8_e4m3` 或 `mxfp8` 通信量化时，发送端一并传输量化载荷与缩放因子（`_QuantizedFAInput`），接收端按 `backend`（`"auto"` / `"npu_fa"` / `"quant_fa"`）决定反量化回 BF16/FP16 还是直接喂入 `quant_fa` 内核。
3. **Ring Attention P2P 环形传递与 LSE 在线合并**：
   - 在 `ring_group`（大小 `R > 1`）内，每个步长 `r in range(R)` 一边用当前持有的 KV 分片调用 FlashAttention（获取局部输出 `O_r` 与对数指数和 `LSE_r`），一边在独立 P2P 通信上异步向 `(rank + 1) % R` 发送当前 KV、从 `(rank - 1 + R) % R` 接收下一轮 KV。
   - 使用标准数值稳定公式 $O_{\text{new}} = \frac{e^{\text{LSE}_{\text{acc}} - \text{LSE}_{\text{max}}} O_{\text{acc}} + e^{\text{LSE}_r - \text{LSE}_{\text{max}}} O_r}{e^{\text{LSE}_{\text{acc}} - \text{LSE}_{\text{max}}} + e^{\text{LSE}_r - \text{LSE}_{\text{max}}}}$ 逐轮更新累加器。

## 验证与反模式警报

- **验证判据**：单卡全序列输出与多卡 `U x R` USP 输出在 `comm_dtype="none"` 下的余弦相似度应 `>= 0.999`；开启 `comm_dtype="fp8_e4m3"` 时 All-to-All 通信字节数减半且无 `USPError` 抛出。
- **反模式警报**：
  - 严禁在 `H_kv < U` 的强 GQA 模型（如 `H_kv=2, U=8`）上直接强开纯 Ulysses，必须改用 `U <= H_kv` 配合 Ring 或切换为 `AllGather-KV x Ulysses` 复合形态。
  - 严禁传入形状或 dtype 不匹配的预分配 `out` / `workspace` 缓冲区（会触发 `USPWorkspaceError [MIE-USP-004]`）。
