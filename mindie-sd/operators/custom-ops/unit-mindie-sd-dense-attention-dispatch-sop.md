---
schema: mindie-graded-unit/1
id: unit-mindie-sd-dense-attention-dispatch-sop
title: MindIE-SD dense attention dispatch from vLLM-Omni and other Ascend callers
tier: L2
category: sop
evidence_strength: software_contract
domain_path: mindie-sd/operators/custom-ops
inherits_from: []
conditions:
  repository: Ascend/MindIE-SD
  revision: 9fe772dfb90696e21b5e5d4e92ca07dffab84cf4
  attention_implementation: mindiesd.attention_forward
triggers:
  error_signatures:
    - FlashAttentionScore host call path unknown
    - Unsupported FA type
  keywords:
    - MindIE-SD
    - mindiesd.attention_forward
    - vLLM-Omni
    - MiniMax-H3
    - dense attention
    - FLASH_ATTN
    - opt_mode
    - FlashAttentionScore
provenance:
  source_type: source_code
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/attention_forward.py
  source_digest: fba14cac85c59cc212be646a358a012afc26045719576beca3fd041d5fec56f2
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:mindiesd/layers/flash_attn/attention_forward.py
    - vllm-project/vllm-omni@9dd5c74af1e32095129085055988517f1369f210:vllm_omni/diffusion/attention/backends/flash_attn.py
  supersedes: []
revision: 1
---
## Applicability

Use this SOP after confirming that the profiled attention call reaches the
public `mindiesd.attention_forward` API at the declared library revision. In
Consult, report the serving framework separately. For this library-scoped
query, `repository` and `revision` identify the library source.

An installed package, a `FLASH_ATTN` backend label, or a
`FlashAttentionScore` kernel name leaves the actual call path unconfirmed.
Confirm it in the deployed caller's selected branch and loaded library or
with a host/API trace before supplying `attention_implementation`. The
vLLM-Omni caller also has branches that bypass this API.

## Dispatch checks

1. Record the caller's actual `fused`, `opt_mode`, data layout, input shape,
   dtype, and mask path. With `fused=False`, the dispatcher executes
   `attention_math`. With fused attention, `runtime` selects with runtime
   search, `static` uses a lookup, and `manual` uses the chosen operator and
   layout. An enabled environment override can change the manual operator
   requested by the caller.
2. For the declared revision, A5 remaps manual `prompt_flash_attn` and
   `ascend_laser_attention` requests to `fused_attn_score`. Inspect the final
   operator and layout in the trace when comparing candidates.
3. Measure first-call dispatch separately from repeated calls with the same
   geometry. Compare the host dispatch interval and the device kernel time
   separately. A change in kernel selection needs matched input shapes and
   an output comparison.

Service-level sequence parallelism, offload, sparse attention and VAE
configuration need their own integration evidence.
