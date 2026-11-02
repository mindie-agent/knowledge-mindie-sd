---
category: sop
conditions:
  framework: mindie-sd
  hardware:
  - Ascend 910B (Atlas A2)
  - Ascend 910C (Atlas A3)
  - Ascend 950PR/DT
domain_path: mindie-sd/operators/custom-ops
evidence_strength: software_contract
id: unit-mindie-sd-custom-sparse-attention-catlass-ffn-and-triton-ops-sop
inherits_from:
- unit-mindie-sd-dit-acceleration-sparse-cache-and-quantization-axioms
- unit-mindie-sd-custom-csrc-ascendc-triton-catlass-operator-dispatch-axioms
provenance:
  evidence_refs:
  - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/operator-dev/SKILL.md
  - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/sparse.md
  - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/quantization.md
  source_digest: 747d1489067e7d888c9c344a8535d6518e846d82a2bbbcd5264585158e1de186
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/operator-dev/SKILL.md
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 3
schema: mindie-graded-unit/1
tier: L2
title: MindIE-SD custom C++/AscendC/TIK/Triton/Catlass operators (RainFusion, EagleQBSA,
  LaPreprocess, mm_swiglu_mxquant) SOP
triggers:
  error_signatures:
  - inferShape function does not exist
  - AI Core 507015
  - aclnnBlockSparseAttention
  keywords:
  - RainFusionAttention
  - AdaBlockSparseAttention
  - EagleQuantBlockSparseAttention
  - EagleQBSA
  - EagleFfn
  - mm_swiglu_mxquant
  - LaPreprocess
  - ascend_laser_attention
  - NormRopeConcat
  - quant_four_over_six_a5
---
## Standard Operating Procedure

1. **Enforce Custom OPP Runtime Registration Ordering (`ASCEND_CUSTOM_OPP_PATH` & `FLA_NPU_DISABLE_PTH=1`)**:
   - `MindIE-SD` packages compiled vendor operators into `CANN-custom_ops--linux.<arch>.run` across three CANN OPP vendor directories (`vendors/customize`, `vendors/aie_ascendc`, and `vendors/customize_transformer`, plus `mindiesd/plugin/libPTAExtensionOPS.so`) and prepends `ASCEND_CUSTOM_OPP_PATH` inside `mindiesd/env.py` on package import.
   - Always execute `import mindiesd` **before** initializing the NPU runtime or constructing any `torch.npu` tensors. Importing `mindiesd` after NPU context initialization causes CANN `aclnn` lookup to miss custom operator tiling/shape libraries and fail with `EZ1014: aclnnXxx ... inferShape function does not exist`.
   - In containers where `flash-linear-attention` (`fla`) installs `fla_npu_opp_env.pth` in `site-packages`, Python startup overwrites `ASCEND_CUSTOM_OPP_PATH` with `fla/ops/ascend/vendors/customize`, breaking both `opc` custom operator compilation (`build.sh`) and runtime vendor resolution unless `export FLA_NPU_DISABLE_PTH=1` is set and all three vendor paths (`customize:aie_ascendc:customize_transformer`) plus their `op_api/lib` directories are exported to `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`.
   - When building `csrc/ops/eagle_ffn` from source on CANN `9.2.0` (`bisheng`), remove `-Werror` alongside `-fpermissive` in `csrc/ops/eagle_ffn/op_host/CMakeLists.txt` (`target_compile_options(cust_optiling PRIVATE -fpermissive)`), as `bisheng` treats `-fpermissive` as `-Werror,-fpermissive` when inherited `-Werror` remains active.

2. **Dispatch Custom Block-Sparse & Quantized Attention Kernels**:
   - **`RainFusionAttention` (`rf_v2`) & `SparseBlockEstimate` (`aclnnSparseBlockEstimate`)**: Combines 3D `[t, h, w]` window token rearrangement, representative-token pooling (`SparseBlockEstimate` with `row_sparse`), first-frame sink (`first_frame_weight`), and block-sparse attention inside `csrc/ops/` and `csrc/plugin/`. On Atlas A5 (`DAV_3510` / `arch35`), `InitSparseParamsArch35` rejects non-empty `actual_seq_lengths` / `actual_seq_lengths_kv`; always pass `actual_seq_lengths=None, actual_seq_lengths_kv=None` on A5.
   - **`aclnnBlockSparseAttention` (`rf_v3`) & `EagleQuantBlockSparseAttention` (`EagleQBSA`)**:
     - `rf_v3` automatically probes V3 $\to$ V2 $\to$ V1 `aclnnBlockSparseAttention` entrypoints.
     - Under `precision="mix"`, `EagleQBSA` executes block-sparse attention with per-block INT8 Q/K and per-channel FP8 V inside a single fused kernel (avoiding the per-layer full-tensor rearrange/pooling overhead of external Python mask paths).
     - Under `precision="fp8"` (`RotateAttention`) and `precision="mxfp4"` (`MXAttention` on Ascend 950), apply online Hadamard rotation before FP8 or FP4 (E2M1 + E8M0 microscaling) attention execution.
   - **`LaPreprocess` (`ascend_laser_attention`)**: Pads Q, K, V sequence lengths to 256/128-token multiples and casts FP32 inputs to BF16/FP16 on device before invoking Atlas A2 Laser Attention.

3. **Integrate Catlass Fused Quantized FFN Kernels (`mm_swiglu_mxquant` / `EagleFfn`)**:
   - `mm_swiglu_mxquant` fuses `QuantMatMul (MXFP8) + SwiGLU + Output MXQuant` into a single Catlass kernel for SwiGLU/MoE DiT blocks (such as `MiniMax-H3`).
   - For `eagle_ffn_linear` (`aclnnEagleFfnV2`), `gelu` and `silu` activations expect `weight1` of shape `[H, K]` and `weight2` of shape `[N2, H]`, whereas `swiglu` splits the first projection along the output channel dimension into gate and up halves and therefore requires `weight1` of shape `[2H, K]` (with optional `bias1` of shape `[2H]`) and `weight2` of shape `[N2, H]`.
   - **Adaptive Weight Layout Caching (`AdaptLayoutCached`)**: Third-party serving engines (`vllm-omni`) store quantized weights in GEMM-ready `(K, N)` layout with `(c, N, 2)` scales and gate-first row ordering. Use C++ `AdaptLayoutCached` to detect `w.size(0) == k` and cache the transposed `w^T` once per weight tensor without runtime row-swapping.
   - **Mandatory Stream Synchronization Before Raw `aclrtLaunch`**: When `AdaptLayoutCached` or upstream preparation performs asynchronous transpose/copy operations, call `aclrtSynchronizeStream` (or bind to the exact caller ACL stream) before raw `aclrtLaunch` of the Catlass kernel; otherwise cross-stream races trigger `AI Core 507015` exceptions.

4. **Apply Ascend 950 (A5) `quant_four_over_six_a5` & Triton-Ascend Vertical Fusions**:
   - On Ascend 950, `W4A4_MXFP4_FOUROVERSIX_DYNAMIC` dispatches `quant_four_over_six_a5` (`aclnnQuantFourOverSixA5V2`) to evaluate adaptive 4-vs-6 block scaling thresholds for MXFP4 dynamic quantization. The host tiling check (`QuantFourOverSixA5CheckParams`) strictly requires `dst_type_max=4.0` (the FP4 E2M1 positive maximum); passing `6.0` fails tiling with `aclnnQuantFourOverSixA5V2GetWorkspaceSize failed (161002)`.
   - For Triton-Ascend vertical fusions (`NormRopeConcat`, `AdaLayerNorm`, `mmgelu`), keep tile sizes within UB capacity, use integer bitwise RNE for any intermediate BF16 rounding, and verify zero spillover in compiled `.ttir`/`.ttadapter` IR.
