---
schema: mindie-graded-unit/1
id: unit-mindie-sd-sequence-padding-alignment-and-multi-card-scaling-workflow
title: "MindIE-SD 序列并行填充对齐、AlltoAllV 缺陷绕过与并行×稀疏叠加工作流"
tier: L3
category: workflow
evidence_strength: reproduced_empirical
domain_path: mindie-sd/parallel/usp-and-cfg
inherits_from:
  - unit-mindie-sd-usp-ulysses-ring-and-cfg-sequence-parallel-axioms
  - unit-mindie-sd-usp-ulysses-alltoall-and-ring-p2p-attention-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "HcclAlltoAllV SIGSEGV"
    - "EE1003 coreDim"
    - "hcom_alltoallv"
  keywords:
    - S_PAD
    - HcclAlltoAllV
    - hcom_alltoall
    - Gather-then-Sparse
    - actual_seq_lengths
    - step_trace
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
  source_digest: a48ef062ddc8a6354655aafe27336e4ee7ccf2e45cc7e0c494e4f6e9fb73256c
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/usp.py
  supersedes: []
revision: 1
---

## 工作流目标与适用条件

本工作流规定在多卡序列并行（USP / CP）中处理非整除序列长度的 `S_PAD` 对齐、规避不等长 `HcclAlltoAllV` 崩溃、打通“序列并行 × 块稀疏注意力”拼接契约（Seam Contract）以及执行 2→4→8 卡线性度归因的完整流程。

## 端到端执行步骤

1. **阶段一：序列填充对齐与 `HcclAlltoAllV` 等分绕过**：
   - 当视觉或文本序列长度 $S$ 不能被序列并行度 $SP$（以及稀疏块大小 `128`）整除时，若直接按不等长切片调用 `HcclAlltoAllV`，在部分 CANN 版本上会触发底层 `SIGSEGV` 或性能退化。
   - **Pad + 等分 AlltoAll 规程**：
     1. 由全局最大分片长度推导全 rank 一致的对齐长度 `S_PAD`（向上对齐到 `128` 的整数倍），将各块输入在尾部 Pad 到 `S_PAD`。
     2. 调用等分 `HcclAlltoAll(count = S_PAD * row_elems)` 完成跨卡交换。
     3. 接收端按各源 rank 真实有效长度 `out_sizes[j]` 切片还原，或通过 `actual_seq_lengths` 传给 FlashAttention 屏蔽尾部 Padding Token。
     4. 核验 Profiler 中算子名由 `hcom_alltoallv` 变为 `hcom_alltoall`，且输出与单卡参考逐位/高余弦对齐。
2. **阶段二：打通“序列并行 × 块稀疏”叠加契约（Seam Contract）**：
   - **先汇聚后稀疏（Gather-then-Sparse）**：Ulysses 在进入注意力内核前已完成序列维全量聚合（每个 rank 持有 `H / U` 个头的完整序列 $S$）。因此，RainFusion V2/V3 或 AdaBSA 的 3D 窗口重排、池化打分与块掩码生成必须在**Ulysses All-to-All 汇聚为完整序列之后、按本 rank 局部头数 `H_local = H / U` 独立执行**。
   - **Per-Head 掩码偏移修正**：若稀疏掩码在进入注意力前已预先生成或跨层缓存，必须按 `rank_in_ulysses * H_local : (rank_in_ulysses + 1) * H_local` 切取当前 rank 对应的头切片，严禁所有 rank 重复使用前 `H_local` 个头的掩码。
3. **阶段三：多卡扩展线性度核算与阶段 $\Delta$ 归因**：
   - 先跑 4 步 Smoke 测试确认单步耗时无整组异常抬高（排除前序进程 `SIGKILL` 导致的 HCCL/NPU 残留降频态）。
   - 采集 2 卡、4 卡、8 卡的单步 `step_trace`，将耗时分解为：`Computing`（应接近按 $1/SP$ 线性缩减）、`Exposed Communication`（按 communicator 区分 `sp_group` 与 `cfg_group`）以及非并行固定开销（如未切分的文本编码或冗余 VAE 解码）。

## 验收判据与常见故障排查

- **生效判据**：并行 × 稀疏叠加后，单卡注意力内核同时体现序列切头（`H_local = H / U`）与块稀疏加速（核耗时在单纯 USP 基础上按稀疏比进一步下降）。
- **故障排查**：若开启稀疏后多卡耗时无变化，检查是否因 `S_PAD` 改变了序列长度导致 `tq * hq * wq + txt_len != S_padded` 而回退到了稠密分支。
