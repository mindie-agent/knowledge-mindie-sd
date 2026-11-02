---
schema: mindie-graded-unit/1
id: unit-mindie-sd-cfg-parallel-group-init-and-allgather-kv-hybrid-sop
title: "MindIE-SD CFG 并行通信组正交划分、AllGather-KV×Ulysses 复合选型与通信流掩盖规程"
tier: L2
category: sop
evidence_strength: reproduced_empirical
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
    - "profile 只有 allGather 无 allToAll"
    - "Communication(Not Overlapped)"
  keywords:
    - cfg_parallel
    - AllGather-KV
    - context_parallel_config
    - comm-stream masking
    - all_to_all_single
    - npu-smi info -t topo
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
  source_digest: a48ef062ddc8a6354655aafe27336e4ee7ccf2e45cc7e0c494e4f6e9fb73256c
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/parallelism.md
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当扩散模型启用 Classifier-Free Guidance（`guidance_scale > 1.0`）需要在多卡间划分 `cfg_group` 与序列并行组（`sp_group`），或在跨域（SYS/PCIe）拓扑与小 `H_kv` GQA 约束下抉择纯 Ulysses 与复合 `AllGather-KV × Ulysses`、并注入专用通信流掩盖（comm-stream masking）时，必须遵循本规程。

## 核心操作规程

1. **卡组拓扑合法性核验与正交进程组划分**：
   - 运行多卡任务前必须通过 `npu-smi info -t topo` 核对互连分域：2 卡必须位于同一 UB/HCCS 互连对，4 卡必须构成完整同域组，严禁随意跨互连域拼凑卡组。
   - **CFG × SP 正交划分**：总卡数 `W = 2 * SP` 时，将条件分支（conditional）与无条件分支（unconditional）分配到两个独立的 `sp_group`（每个 `sp_group` 大小为 `SP`，尽量落在同一 UB/HCCS 高带宽域内以加速高频的每层序列并行通信）；同时每对对应 rank 组成大小为 2 的 `cfg_group`，仅在每步 DiT 前向结束时执行一次 `all_gather` 汇总双分支噪声预测并计算 `noise_pred = uncond + guidance_scale * (cond - uncond)`。
2. **纯 Ulysses vs `AllGather-KV × Ulysses` 复合序列并行选型**：
   - **同域高带宽 + 充足 KV 头（`H_kv >= SP`）**：优先选用纯 Ulysses（4 次 `all_to_all_single`：Q/K/V 切头聚序列 + Out 逆变换）。
   - **跨域低带宽（如跨 PCIe/SYS）或 GQA 头数受限（`H_kv < SP`）**：将序列并行拆分为域内 `Ulysses`（利用高带宽 HCCS/UB 跑 `all_to_all`）与跨域 `AllGather-KV`（跨域仅聚合体积较小的 GQA K/V 张量，避免跨域 `all_to_all` 打满慢速链路）。
   - **处理器配置注入**：在框架集成时，必须给每个 `attn.processor._parallel_config` 显式注入 `ParallelConfig(context_parallel_config=cp_cfg)`，不能仅对输入序列做切分而遗漏处理器配置（否则 Profile 中只有 `allGather` 而无 `allToAll` 切头路径）。
3. **独立通信流掩盖（Comm-Stream Masking）与上限核算**：
   - 将集合通信（`all_to_all_single` / `all_gather_tensor`）投递到专用 HCCL 通信流（compute stream 记录 `ready` event -> comm stream `wait_event` 并执行集合通信 -> 记录 `done` event -> compute stream 在真正消费结果前 `wait_event`），使通信与不依赖该结果的投影或另一分支计算重叠。
   - 掩盖率真实上限由可重叠计算耗时 $c$ 与暴露通信耗时 $f$ 的比值 $\min(1 - 1/n, c/f)$ 决定。

## 验证与反模式警报

- **验证判据**：通过 Ascend Profiler 的 `step_trace` 检查 `Communication(Not Overlapped)` 占比及 HCCL 算子名称（确认出现 `hcom_alltoall` 且按 communicator 归属到预期的 `sp_group` / `cfg_group`）。
- **反模式警报**：
  - 严禁把高频的 `sp_group` 跨 SYS/PCIe 布置而把低频的 `cfg_group` 放在同域内，这会导致每层 4 次的 All-to-All 被慢速跨域链路卡死。
  - 严禁仅凭日志打印“USP enabled”就认定序列并行生效，必须核验单卡注意力的输入形状与 HCCL 算子轨迹。
