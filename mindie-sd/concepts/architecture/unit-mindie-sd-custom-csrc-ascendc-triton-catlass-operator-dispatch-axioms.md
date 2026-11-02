---
schema: mindie-graded-unit/1
id: unit-mindie-sd-custom-csrc-ascendc-triton-catlass-operator-dispatch-axioms
title: MindIE-SD custom C++/AscendC/Triton/Catlass operator registration, fake-tensor meta ops, and MindieSDBackend compilation axioms
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
    - "inferShape function does not exist"
    - "sparse_flash_attn_rf_v2 (rain_fusion_attention) is not supported on A5 devices"
    - "ada_block_sparse_attention / get_estimate_mask (ada_bsa) are not supported on A5 devices"
    - "eagle_ffn_linear: weight1 shape"
  keywords:
    - torch.ops.mindiesd
    - register_mindie_fake_op
    - MindieSDBackend
    - ACLGraphBackend
    - eagle_ffn_linear
    - quant_four_over_six_a5
    - rainfusionattention
    - block_sparse_attention
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/_custom_ops.py
  source_digest: 58988ad797b6ca5c85203e488918f0c5b45b91547b57ad7dc88669b9e4809714
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/_custom_ops.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/compilation/mindie_sd_backend.py
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/compilation.md
  supersedes: []
revision: 1
---
## Core Invariants

1. **Dual-Layer C++ Plugin & Vendor OPP Registration (`csrc/plugin` + `mindiesd/ops/vendors`)**:
   - Custom operators in `csrc/ops/` (AscendC/TIK/Catlass kernels) are compiled into vendor OPP packages and loaded via `ASCEND_CUSTOM_OPP_PATH` in `mindiesd/env.py`, while `csrc/plugin/*.cpp` binds them into `torch.ops.mindiesd.*` (`la`, `la_preprocess`, `rainfusionattention`, `sparse_block_estimate`, `ada_block_sparse_attention`, `block_sparse_attention`, `eagle_block_sparse_attention`, `eagle_quant_block_sparse_attention`, `adaln`, `adaln_v2`, `layernorm`, `eagle_ffn_linear`, `quant_four_over_six_a5`, `rope`, `mul_add`).
   - Every `torch.ops.mindiesd` operator in `mindiesd/layers/_custom_ops.py` **must** register a matching `@register_ops.register_mindie_fake_op("<op_name>")` meta implementation so `torch.compile` (`Dynamo` / `AOTAutograd`) can trace output shapes, dtypes, and layouts without executing NPU kernels during graph capture.
   - In `block_sparse_attention_fake`, when `quant_mode >= 2` (packed `MXFP4` `float4_e2m1fn_x2` where the last dimension packs 2 elements per `uint8` byte), the fake op must double `out_shape[-1] *= 2` and return `torch.bfloat16` so downstream traced nodes receive the unpacked head dimension $D$.

2. **SoC-Specific Operator Routing & Hardware Guards (`is_a5_device`)**:
   - `rain_fusion_attention` (`rf_v2`) and `ada_block_sparse_attention` / `get_estimate_mask` (`ada_bsa`) are **Atlas A2/A3 only**; invoking their low-level entrypoints on Ascend 950 (`is_a5_device() == True`) raises `ParametersInvalid`. The public `mindiesd.layers.flash_attn.sparse_attention` dispatcher automatically routes `rf_v2` requests on A5 to `rf_v3` (`bsa_sparse_attention_v3`).
   - Conversely, `eagle_ffn_linear` ($y = \text{act}(x W_1^T + b_1) W_2^T + b_2$ with automatic `[N, K]` vs `[K, N]` weight layout detection) and `quant_four_over_six_a5` (32-aligned tail-axis BF16 $\to$ MXFP4 adaptive 4/6 block scaling) target Ascend 950 (A5).

3. **Two-Phase FX Graph Rewriting in `MindieSDBackend` (`mindiesd/compilation/mindie_sd_backend.py`)**:
   - `MindieSDBackend` splits FX passes around Inductor `freeze` (`CompilationConfig.enable_freezing`):
     1. **Before Freezing (`graph_rewrite_before_freezing`)**: Runs `ReduandantNodeEliminationPass` followed by `apply_pattern_match_passes` (`patterns` registered once via `activate_pattern_once()`), because Inductor `freeze` performs common subexpression elimination (CSE) that would otherwise mutate subgraph topologies and break pattern matching rules (such as `RMSNorm`, `RoPE`, `AdaLayerNorm`, and `DualLinear` fusion patterns).
     2. **After Freezing (`graph_rewrite_after_freezing`)**: Runs `ReduandantNodeEliminationPass` and `decompose_auto_functionalized` to clean up functionalized in-place custom ops.
   - When `CompilationConfig.aclgraph_with_compile` is enabled and `npu_graph_available` is true, `MindieSDBackend` chains FX pattern fusion into `create_aclgraph_backend()` for static NPU stream capture and replay.
