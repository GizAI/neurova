# Compiler lowering boundary

The compiler does not declare MLA, GDN, attention, MoE, Conv3D, or VAE to be a
single native instruction merely because a name appears in a dictionary.
High-level model nodes are expanded into validated primitives.

Examples:

- Gated DeltaNet: RMSNorm -> GEMM -> vector gate -> state update -> GEMM.
- GQA/MLA attention: RMSNorm -> projections -> RoPE -> score GEMM -> Softmax ->
  value GEMM -> output GEMM.
- MoE: router GEMM -> Top-K -> indexed gather -> expert GEMMs -> combine.
- H3 video transformer: Patchify3D -> AdaLN -> GEMMs -> RoPE/Softmax -> FFN.

A profile passes only when every emitted primitive is in the supported set.
That pass is operator legalization, not full-checkpoint quality certification.
