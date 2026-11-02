---
schema: mindie-graded-unit/1
id: unit-mindie-sd-usp-ulysses-ring-and-cfg-sequence-parallel-axioms
title: MindIE-SD sequence parallelism (Ulysses AllToAll, Ring P2P attention, CFG parallel, and hybrid USP) axioms
tier: L1
category: axiom
evidence_strength: software_contract
domain_path: mindie-sd/concepts/architecture
inherits_from: []
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
    - "q/kv heads must be divisible by Ulysses world size"
    - "FA output sequence"
  keywords:
    - usp_attention
    - ulysses_group
    - kv_gather_group
    - USPTopologyError
    - USPShapeError
    - cfg_size
    - ulysses_size
    - ring_size
    - comm_quant_scope
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/usp.py
  source_digest: b2a9986c49333a8c479fdcdb45613fc3a9d80f398142c91d573ad1e9e408748a
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/usp.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/usp.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/parallelism.md
  supersedes: []
revision: 1
---
## Core Invariants

1. **Unified Sequence-Parallel Attention (`mindiesd.layers.usp.usp_attention`)**:
   - `usp_attention` executes Ulysses (`ulysses_group`), KV-AllGather (`kv_gather_group`), optional communication quantization (`comm_dtype="none" | "fp8_e4m3" | "mxfp8"`), and local FlashAttention (`backend="auto" | "npu_fa" | "quant_fa"`) in a single stateless call without mutating or introspecting caller-owned process-group state.
   - **Strict Head & Sequence Divisibility (`USPTopologyError` / `MIE-USP-002`)**: Both `q_heads` and `kv_heads` must be evenly divisible by `world_size(ulysses_group)`. After forward Ulysses `all_to_all_single`, the global sequence length must be divisible by `world_size(ulysses_group)` during `_ulysses_reverse`.
   - **Replicated `joint_k` / `joint_v` Contract**: When MMDiT joint tokens (`joint_k`, `joint_v`) are passed, both must be provided together (`USPShapeError` / `MIE-USP-003` otherwise); they are treated as replicated across ranks and head-sliced via `_slice_joint` along `ulysses_group`, never AllGathered along the sequence axis.

2. **Hybrid 2D Sequence Parallelism (`Ulysses x KV-AllGather` vs `Ring` P2P Attention)**:
   - **`Ulysses x KV-AllGather`**: When card count exceeds `kv_heads` (GQA models) or crosses HCCS/UB island boundaries, `ulysses_group` exchanges sequence shards for head shards within an island while `kv_gather_group` gathers K/V sequence shards across islands before a single local FA call. When `backend="quant_fa"` and `comm_dtype="fp8_e4m3"` with `key.shape[1] % kv_block_size == 0`, `_all_gather_quant_fa` gathers FP8 payloads and scales directly without dequantizing and re-quantizing.
   - **`Ring` Sequence Parallelism (RSP)**: Circulates K/V chunks across ranks via P2P `send_recv` while accumulating partial attention outputs via online softmax log-sum-exp (`LSE`) rescaling. Unlike `KV-AllGather`, Ring P2P avoids materializing the full global K/V sequence on each rank, but cannot be combined with full-sequence block-sparse attention (`RainFusion` / `Ada-BSA`) because block selection requires global K/V visibility.

3. **Dual-Stream Head-Chunk Overlap & Communication Quantization Scope (`overlap=True`)**:
   - When `overlap=True` and `head_chunk_size` is configured (defaulting to `world_size(ulysses_group)`), `_run_head_chunks` pipelines forward collectives on a dedicated `comm_stream` with FA execution on `current_stream` using per-chunk `torch.npu.Event` synchronization and `tensor.record_stream(comm_stream)`.
   - `comm_quant_scope="exposed"` quantizes only the exposed pipeline boundary chunks (chunk `0` on forward `q/k/v` and chunk `chunk_count - 1` on reverse `out`), leaving overlapped interior chunks in native BF16/FP16; `comm_quant_scope="all"` quantizes all chunks when communication time exceeds FA compute time ($c > f$).

4. **Classifier-Free Guidance Parallelism (`CFG Parallel` vs `CFG Fusion`)**:
   - **CFG Parallel (`cfg_size = 2`)**: Splits positive (`cond`) and negative (`uncond`) guidance branches across two process groups (`ulysses_size * cfg_size == nproc_per_node`), exchanging only the final denoised latent tensor per step via `all_gather` before computing `uncond + guidance_scale * (cond - uncond)`.
   - Only enable `cfg_size = 2` when `guidance_scale > 1.0` and `negative_prompt` is active; enabling `cfg_size = 2` when `guidance_scale <= 1.0` wastes half the NPU fleet on redundant computation without automatic fallback.
