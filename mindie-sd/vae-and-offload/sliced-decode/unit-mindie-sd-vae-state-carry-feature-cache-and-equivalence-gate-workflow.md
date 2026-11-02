---
schema: mindie-graded-unit/1
id: unit-mindie-sd-vae-state-carry-feature-cache-and-equivalence-gate-workflow
title: "MindIE-SD VAE/TAE 状态携带切分四道等价验证闸与集体通信防死锁工作流"
tier: L3
category: workflow
evidence_strength: reproduced_empirical
domain_path: mindie-sd/vae-and-offload/sliced-decode
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-3d-causal-vae-wanvae-hunyuan-taehv-temporal-tiling-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "all_gather_into_tensor"
    - "repeat_interleave"
    - "rank < pieces"
  keywords:
    - vae-opt
    - shard_pieces_from_env
    - all_gather_into_tensor
    - repeat_interleave
    - bitwise
    - apply_sliced
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/vae-opt/SKILL.md
  source_digest: 351aff4228d45196180fe9a7fd227070f7cff06c7df846332af3d0cc8c43dcf6
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/vae-opt/SKILL.md
  supersedes: []
revision: 1
---

## 工作流目标与适用条件

本工作流规定在实现或修改 MindIE-SD VAE/TAE 分片/分段解码时，核算跨卡交换预算、编写与 Rank 无关的集体通信分支判定、排查张量原地填充陷阱以及执行“四道等价验证闸”的完整工程闭环。

## 端到端执行步骤

1. **阶段一：前置收益与交换预算核算（第 0 步 & 第 5 步）**：
   - 先测量 VAE/TAE 解码在端到端（E2E）中的真实耗时占比。若解码占比 `< 5%`，优先优化 DiT 去噪与 Host 侧交付开销（如 mp4 编码与落盘）。
   - 计算跨卡交换字节数：$\text{Bytes}_{\text{comm}} = T \times H \times W \times C \times \text{sizeof(dtype)} \times \frac{\text{pieces} - 1}{\text{pieces}}$。仅当通信耗时严格小于切分省下的计算耗时时，才允许引入多卡切分 + `all_gather`。
2. **阶段二：编写与 Rank 无关的通信分支判定并规避静默陷阱**：
   - **铁律（防 HCCL 死锁）**：决定是否进入集合通信（状态交换、`all_gather_into_tensor`）的布尔条件 `ok` **只能由全局一致的环境变量与输入张量 `shape` 推导**，严禁出现 `if rank < pieces:` 之类的不对称集合通信入口。
   - **三大已知静默陷阱排查**：
     1. `dist.all_gather_into_tensor(out, local_slice)` 的目标 `out` 必须是原地持久分配的张量，严禁传入由 `.contiguous()` 临时生成的栈上副本（会导致集合通信写进临时副本后即被丢弃，主张量全零或残留旧值）；
     2. 避免使用 `torch.repeat_interleave` 整张量复制扩张时间维（会额外分配整份巨型显存），应改用 `expand` + `reshape` 视图操作；
     3. 在 NPU 上当单次 `Upsample` / `Conv` 帧数超过算子硬件分块上限导致尾段帧数值塌陷时，用 2 段或 4 段状态携带切分（`apply_sliced`）将单次调用的帧数降到安全阈值内。
3. **阶段三：依次通过四道等价验证闸（Gate ①~④）**：
   - **闸 ①（CPU Bitwise 真值对拍）**：固定同一份输入 latent 与权重，在 CPU 上对比整段解码 `apply_full` 与 `apply_sliced(..., slices=2/4)`，必须满足 `max|diff| == 0`（逐位完全相等）。
   - **闸 ②（NPU 分段与 CPU 逐桶数值对齐）**：在 NPU 上运行分段解码，逐帧核对均值、方差与分位数，确认与 CPU 真值对齐且尾帧无数值塌陷。
   - **闸 ③（端到端产物 MD5 / 像素级一致性）**：对于无损分片（如 `tae-2d` 帧轴切分或 `taehv` 状态携带切分），验证多请求生成的视频帧指纹与基线一致。
   - **闸 ④（生效状态量与显存/耗时实测）**：确认分片状态计数器确实命中切分分支（非静默回退整段），且 NPU 峰值显存与耗时达到预期收益。

## 验收判据与常见故障排查

- **验收判据**：四道验证闸全部通过，多卡压力测试无 HCCL 挂起超时。
- **故障排查**：若 CPU 上 `max|diff| > 0`，立即检查是否有 `GroupNorm` 跨切分轴统计了均值方差，或某个 `MemBlock` 漏存/错存了输入端末帧状态。
