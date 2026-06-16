from __future__ import annotations

import torch

from ..ops import cuda_ops
from ..loader import FP16Tensor, QuantizedStore
from .qwen36_impl.model import (
    Qwen36MLP,
    Qwen36Model,
    WeightResolver,
    apply_rope_single_token,
    attention_decode_any,
    embed_lookup,
    linear_any,
    qwen_rmsnorm,
    qwen_rmsnorm_lastdim,
)
from ..speculation import DraftProposal, DraftRequest


class QwenNativeMTP1:
    """Qwen3.6 native MTP1 candidate generator."""

    def __init__(self, model: Qwen36Model, store: QuantizedStore):
        self.model = model
        self.cfg = model.cfg
        self.device = model.device
        wr = WeightResolver(store)
        self.pre_fc_norm_embedding = wr.fp16("mtp.pre_fc_norm_embedding.weight").to(self.device, dtype=torch.float16).contiguous()
        self.pre_fc_norm_hidden = wr.fp16("mtp.pre_fc_norm_hidden.weight").to(self.device, dtype=torch.float16).contiguous()
        self.fc = wr.get("mtp.fc.weight")
        if not isinstance(self.fc, FP16Tensor):
            raise TypeError("mtp.fc.weight must be fp16_raw for native MTP")
        self.input_norm = wr.fp16("mtp.layers.0.input_layernorm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.post_norm = wr.fp16("mtp.layers.0.post_attention_layernorm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.qkv_proj = wr.any_linear("mtp.layers.0.self_attn.qkv_proj.weight")
        self.o_proj = wr.any_linear("mtp.layers.0.self_attn.o_proj.weight")
        self.q_norm = wr.fp16("mtp.layers.0.self_attn.q_norm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.k_norm = wr.fp16("mtp.layers.0.self_attn.k_norm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.mlp = Qwen36MLP(self.cfg, wr, prefix="mtp.layers.0")
        self.norm = wr.fp16("mtp.norm.weight").to(self.device, dtype=torch.float16).contiguous()
        kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
        self.qkv_q_rows = self.cfg.num_attention_heads * self.cfg.attention_head_dim * 2
        self.qkv_split = (self.qkv_q_rows, kv_rows, kv_rows)

    def _single_token_decoder_layer(
        self,
        x: torch.Tensor,
        *,
        pos: int,
        k_cache: list[torch.Tensor] | None = None,
        v_cache: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = x
        h = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        qkv_all = linear_any(self.qkv_proj, h)
        q_all, k_all, v_all = torch.split(qkv_all, self.qkv_split, dim=0)
        q_heads = q_all.view(self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
        q, gate = torch.chunk(q_heads, 2, dim=-1)
        k = k_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        v = v_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        q = qwen_rmsnorm_lastdim(q.contiguous(), self.q_norm, self.cfg.rms_norm_eps)
        k = qwen_rmsnorm_lastdim(k.contiguous(), self.k_norm, self.cfg.rms_norm_eps)
        q, k = apply_rope_single_token(q, k, pos=pos, rope_dim=self.cfg.rope_dim, rope_theta=self.cfg.rope_theta)

        if k_cache is None or v_cache is None:
            ratio = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
            att = v.repeat_interleave(ratio, dim=0)
        else:
            k_cache.append(k.contiguous())
            v_cache.append(v.contiguous())
            k_live = torch.stack(k_cache, dim=1).contiguous()
            v_live = torch.stack(v_cache, dim=1).contiguous()
            att = attention_decode_any(
                q.contiguous(),
                k_live,
                v_live,
                len(k_cache),
                self.cfg.attention_head_dim ** -0.5,
            )
        att_flat = (att.reshape(-1) * torch.sigmoid(gate.reshape(-1).to(att.dtype))).contiguous()
        h = residual + linear_any(self.o_proj, att_flat)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(h)

    def _next_token_hidden(
        self,
        hidden_state: torch.Tensor,
        input_token: torch.Tensor,
        *,
        pos: int,
        k_cache: list[torch.Tensor] | None = None,
        v_cache: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        emb = embed_lookup(self.model.embed, input_token).to(self.device, dtype=torch.float16).reshape(-1).contiguous()
        h_norm = qwen_rmsnorm(
            hidden_state.to(self.device, dtype=torch.float16).contiguous(),
            self.pre_fc_norm_hidden,
            self.cfg.rms_norm_eps,
        )
        e_norm = qwen_rmsnorm(emb, self.pre_fc_norm_embedding, self.cfg.rms_norm_eps)
        x = torch.cat([e_norm, h_norm], dim=0)
        x = linear_any(self.fc, x).to(self.device, dtype=torch.float16).contiguous()
        x = self._single_token_decoder_layer(x, pos=pos, k_cache=k_cache, v_cache=v_cache)
        return qwen_rmsnorm(x.contiguous(), self.norm, self.cfg.rms_norm_eps)

    def _second_token_hidden(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        return self._next_token_hidden(raw_hidden, first_token, pos=pos)

    @torch.no_grad()
    def logits_for_second_token(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        x = self._second_token_hidden(raw_hidden, first_token, pos=pos)
        # LowBitMarlinTensor reuses per-batch output buffers. Return an owned
        # tensor so candidate logits cannot alias later target/model projections.
        return linear_any(self.model.lm_head, x).contiguous().clone()

    @torch.no_grad()
    def argmax_for_second_token(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        x = self._second_token_hidden(raw_hidden, first_token, pos=pos)
        # Candidate generation only needs argmax. Avoid cloning the full
        # 248K-vocab logits buffer on the speculative hot path.
        logits = linear_any(self.model.lm_head, x).contiguous()
        if logits.device.type == "cuda":
            return cuda_ops().argmax(logits).reshape(()).to(device=logits.device, dtype=torch.long)
        return torch.argmax(logits, dim=-1).reshape(()).to(dtype=torch.long)

    @torch.no_grad()
    def argmax_sequence(
        self,
        raw_hidden: torch.Tensor,
        first_token: torch.Tensor,
        *,
        pos: int,
        max_draft: int,
    ) -> torch.Tensor:
        """Generate NEXTN-style draft tokens with the native Qwen MTP layer.

        external serving engine feeds the sampled first token and current target hidden into the
        Qwen3Next MTP module, then repeatedly feeds each draft token with the
        previous MTP hidden. Qwen3.6-27B ships one MTP layer, so the same layer is
        reused for each speculative step.
        """
        if max_draft <= 0:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        tokens = torch.empty((max_draft,), device=self.device, dtype=torch.long)
        hidden = raw_hidden
        token = first_token.reshape(()).to(device=self.device, dtype=torch.long)
        k_cache: list[torch.Tensor] = []
        v_cache: list[torch.Tensor] = []
        for step in range(max_draft):
            hidden = self._next_token_hidden(
                hidden,
                token,
                pos=pos + step,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            logits = linear_any(self.model.lm_head, hidden).contiguous()
            if logits.device.type == "cuda":
                token = cuda_ops().argmax(logits).reshape(()).to(device=logits.device, dtype=torch.long)
            else:
                token = torch.argmax(logits, dim=-1).reshape(()).to(dtype=torch.long)
            tokens[step] = token
        return tokens


class QwenNativeMTP1Proposer:
    """Native Qwen MTP1 proposer behind the shared speculation contract."""

    method = "native_mtp1"

    def __init__(self, mtp: QwenNativeMTP1):
        self.mtp = mtp

    @torch.no_grad()
    def propose_tensor(self, request: DraftRequest) -> torch.Tensor:
        if request.max_draft <= 0:
            raise ValueError("native_mtp1 tensor proposal requires max_draft > 0")
        return self.propose_tensors(request)[0]

    @torch.no_grad()
    def propose_tensors(self, request: DraftRequest) -> torch.Tensor:
        if request.max_draft <= 0:
            return torch.empty((0,), device=self.mtp.device, dtype=torch.long)
        signals = request.signals or {}
        raw_hidden = signals.get("raw_hidden")
        first_token = signals.get("first_token")
        pos = signals.get("pos")
        if raw_hidden is None or first_token is None or pos is None:
            raise ValueError("native_mtp1 proposer requires raw_hidden, first_token, and pos signals")
        return self.mtp.argmax_sequence(
            raw_hidden,
            first_token,
            pos=int(pos),
            max_draft=int(request.max_draft),
        )

    @torch.no_grad()
    def propose(self, request: DraftRequest) -> DraftProposal:
        if request.max_draft <= 0:
            return DraftProposal(method=self.method, tokens=[])
        tokens = self.propose_tensors(request)
        return DraftProposal(method=self.method, tokens=[int(t) for t in tokens.detach().cpu().tolist()])


def native_mtp1_proposer_for_model(model: object) -> QwenNativeMTP1Proposer | None:
    if not isinstance(model, Qwen36Model):
        return None
    try:
        return QwenNativeMTP1Proposer(QwenNativeMTP1(model, model.store))
    except (KeyError, TypeError):
        return None
