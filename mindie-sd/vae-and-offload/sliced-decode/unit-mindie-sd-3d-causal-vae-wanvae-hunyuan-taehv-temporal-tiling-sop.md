---
schema: mindie-graded-unit/1
id: unit-mindie-sd-3d-causal-vae-wanvae-hunyuan-taehv-temporal-tiling-sop
title: "MindIE-SD 3D 因果 VAE（WanVAE / HunyuanVAE / TAEHV）跨轴耦合审计与潜帧边界切分规程"
tier: L2
category: sop
evidence_strength: reproduced_empirical
domain_path: mindie-sd/vae-and-offload/sliced-decode
inherits_from:
  - unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
  - unit-mindie-sd-vae-tae-sliced-state-decode-and-offload-sop
conditions:
  framework: mindie-sd
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
triggers:
  error_signatures:
    - "chunk_latents=5 / trim=3"
    - "帧轴切分不可能精确"
  keywords:
    - vae-opt
    - taehv
    - tae-2d
    - MemBlock
    - apply_sliced
    - TGrow
    - time_upscale
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/vae-opt/SKILL.md
  source_digest: 351aff4228d45196180fe9a7fd227070f7cff06c7df846332af3d0cc8c43dcf6
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/vae-opt/SKILL.md
  supersedes: []
revision: 1
---

## 适用场景与触发条件

当长视频扩散模型（如 Wan2.1/Wan2.2、HunyuanVideo、LTX-Video 等采用 3D 因果卷积 VAE 或带 `MemBlock` 时间递归的 `taehv` 解码器）在解码高分辨率长视频潜变量时遭遇单卡 HBM 峰值溢出、CANN 算子超长帧数尾段数值塌陷或 8 卡冗余解码耗时过高，需要决策切分轴、潜帧边界对齐与状态传递方案时，必须遵循本规程。

## 核心操作规程

1. **第一步：逐算子跨轴耦合审计（禁止凭经验盲切帧轴）**：
   - 遍历解码器内所有算子（Conv2d/Conv3d、GroupNorm/RMSNorm、Upsample、Attention、MemBlock），按“输出元素是否依赖切片边界外的输入”分类：
     - **纯逐帧 2D 解码器（如 `tae-2d`）**：`Clamp`、`3x3 Conv2d`、2D `Upsample`、逐帧 `quantile` 归一化在时间轴 $T$ 上完全独立 -> **允许沿帧轴 $T$ 多卡无 Halo 等分并行**，解码后执行一次 `all_gather` 拼接。
     - **含时间维因果递归或 3D 因果卷积的解码器（如 `taehv-temporal`、WanVAE、Hunyuan 3D Causal VAE）**：存在 `MemBlock(past)` 或因果 `Conv3d`（第 $t$ 帧依赖前序帧激活） -> **帧轴 $T$ 绝不可能无状态独立并行切分**；盲目按帧切给多卡会导致每段开头被零填充（`zeros pad`）重置，产生周期性帧闪烁与数值失真。
2. **第二步：潜帧边界对齐（Latent Frame Boundary Alignment）**：
   - 时间上采样层（如 `TGrow` 或因果 3D 上采样，`time_upscale = 4`）将潜空间帧索引 $j$ 映射为输出像素帧 $4j$（或首帧 $1 + 4(j-1)$）。
   - 任何沿时间轴的分段切分点**必须严格落在最粗一级的潜帧整数边界上**，保证每个切片在网络内所有时间上采样层都对应整数帧边界。
3. **第三步：按耦合形态选择合法切分模式**：
   - **模式 A（时间维顺序状态携带切分，省显存 + 绕过超长序列算子缺陷）**：对 `taehv-temporal` / WanVAE 沿潜帧边界切为 $K$ 段（如 2 段或 4 段），单卡按时间顺序依次执行第 $0 \dots K-1$ 段，每段将每一层 `MemBlock` **输入处**的末帧激活（或因果 `Conv3d` 尾部 `kernel_t - 1` 帧特征缓存）作为 `state` 显式传给下一段（首段传 `None` 由内部做零填充）。
   - **模式 B（空间维 $H/W$ 带 Halo 分块，支持多卡分摊）**：当帧轴强耦合但空间维仅含有限感受野卷积时，沿 $H$ 或 $W$ 轴切分并携带宽度 `halo >= 感受野半径` 的重叠区；执行前必须先核算 `all_gather` 输出视频张量的通信字节数是否小于省下的计算耗时。

## 验证与反模式警报

- **验证判据**：开启潜帧边界状态携带切分（2 段或 4 段）后，在 CPU 上与整段一次性解码输出满足 `max|diff| == 0`（逐位等价），在 NPU 上峰值显存减半且消除长帧尾段坏帧。
- **反模式警报**：
  - 严禁把开源代码中用于裁剪边界伪影的有损启发式（如 `chunk_latents=5, trim=3`）当成帧轴数学独立的证据。
  - 严禁把 `MemBlock` 的**输出**末帧错当成 `state` 传给下一段——`MemBlock` 递推公式引用的是上一帧在该块**输入处**的激活 `x[:, :-1]`，传错位置会破坏逐位等价性。
