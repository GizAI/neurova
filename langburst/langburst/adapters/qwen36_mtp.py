from __future__ import annotations

import os
from typing import Sequence

import torch

from ..ops import cuda_ops
from ..profiling import decode_profile_scope
from ..loader import FP16Tensor, QuantizedStore
from .qwen36_impl.model import (
    Qwen36MLP,
    Qwen36Model,
    WeightResolver,
    apply_rope_decode_batch,
    apply_rope_single_token,
    attention_decode_gated_any,
    embed_lookup,
    embed_lookup_batch,
    linear_any,
    linear_argmax_any,
    marlin_internal_argmax_enabled,
    qwen_rmsnorm,
    qwen_rmsnorm_lastdim,
    qwen_rmsnorm_rope_lastdim,
    qwen_rmsnorm_pair_cat,
    sigmoid_mul_repeat_kv,
)
from ..speculation import DraftProposal, DraftRequest


def _env_enabled(name: str, default: str = "0") -> bool:
    return (os.environ.get(name, default).strip().lower() in {"1", "true", "on", "yes"})


class _MTPLocalKVCache:
    """Time-major local MTP KV cache; avoids per-step torch.stack/contiguous."""

    def __init__(self, *, max_tokens: int, kv_heads: int, head_dim: int, device: torch.device | str, dtype: torch.dtype):
        self.k = torch.empty((int(max_tokens), int(kv_heads), int(head_dim)), device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[self.length].copy_(k)
        self.v[self.length].copy_(v)
        self.length += 1

    def attend(self, q: torch.Tensor, gate: torch.Tensor, scale: float) -> torch.Tensor:
        op = getattr(cuda_ops(), "attention_decode_fp16_gated_tkh", None)
        if q.device.type == "cuda" and callable(op) and _env_enabled("LANGBURST_MTP_LOCAL_TKH_ATTENTION", "1"):
            return op(q.contiguous(), self.k, self.v, gate.contiguous(), int(self.length), float(scale))
        k_live = self.k[: self.length].permute(1, 0, 2).contiguous()
        v_live = self.v[: self.length].permute(1, 0, 2).contiguous()
        return attention_decode_gated_any(q.contiguous(), k_live, v_live, gate.contiguous(), int(self.length), float(scale))


class _MTPBatchedLocalKVCache:
    """Batched time-major MTP cache used by K>1 native proposer batching."""

    def __init__(self, *, batch: int, max_tokens: int, kv_heads: int, head_dim: int, device: torch.device | str, dtype: torch.dtype):
        self.k = torch.empty((int(batch), int(max_tokens), int(kv_heads), int(head_dim)), device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[:, self.length].copy_(k)
        self.v[:, self.length].copy_(v)
        self.length += 1

    def attend(self, q: torch.Tensor, gate: torch.Tensor, scale: float) -> torch.Tensor:
        op = getattr(cuda_ops(), "attention_decode_fp16_gated_tkh", None)
        rows: list[torch.Tensor] = []
        for row in range(int(q.size(0))):
            q_row = q[row].contiguous()
            gate_row = gate[row].contiguous()
            if q_row.device.type == "cuda" and callable(op) and _env_enabled("LANGBURST_MTP_LOCAL_TKH_ATTENTION", "1"):
                rows.append(op(q_row, self.k[row], self.v[row], gate_row, int(self.length), float(scale)))
            else:
                k_live = self.k[row, : self.length].permute(1, 0, 2).contiguous()
                v_live = self.v[row, : self.length].permute(1, 0, 2).contiguous()
                rows.append(attention_decode_gated_any(q_row, k_live, v_live, gate_row, int(self.length), float(scale)))
        return torch.stack(rows, dim=0).contiguous()


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
        local_cache: _MTPLocalKVCache | None = None,
    ) -> torch.Tensor:
        residual = x
        h = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        qkv_all = linear_any(self.qkv_proj, h, profile="mtp_qkv")
        q_all, k_all, v_all = torch.split(qkv_all, self.qkv_split, dim=0)
        q_heads = q_all.view(self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
        q, gate = torch.chunk(q_heads, 2, dim=-1)
        k = k_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        v = v_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        ratio = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        first_mtp_token = local_cache is not None and int(local_cache.length) == 0
        if local_cache is not None:
            if first_mtp_token:
                k = qwen_rmsnorm_rope_lastdim(
                    k,
                    self.k_norm,
                    pos=pos,
                    rope_dim=self.cfg.rope_dim,
                    rope_theta=self.cfg.rope_theta,
                    eps=self.cfg.rms_norm_eps,
                )
            else:
                q = qwen_rmsnorm_rope_lastdim(
                    q,
                    self.q_norm,
                    pos=pos,
                    rope_dim=self.cfg.rope_dim,
                    rope_theta=self.cfg.rope_theta,
                    eps=self.cfg.rms_norm_eps,
                )
                k = qwen_rmsnorm_rope_lastdim(
                    k,
                    self.k_norm,
                    pos=pos,
                    rope_dim=self.cfg.rope_dim,
                    rope_theta=self.cfg.rope_theta,
                    eps=self.cfg.rms_norm_eps,
                )
            local_cache.append(k.contiguous(), v.contiguous())

        if local_cache is None or first_mtp_token:
            att_flat = sigmoid_mul_repeat_kv(
                v.reshape(1, v.size(0), v.size(1)),
                gate.reshape(1, gate.size(0), gate.size(1)),
                ratio,
            ).reshape(-1)
        else:
            with decode_profile_scope("mtp_local_attention"):
                att_flat = local_cache.attend(q.contiguous(), gate.contiguous(), self.cfg.attention_head_dim ** -0.5)
        h = residual + linear_any(self.o_proj, att_flat, profile="mtp_o")
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(h)

    def _single_token_decoder_layer_legacy(
        self,
        x: torch.Tensor,
        *,
        pos: int,
        k_cache: list[torch.Tensor] | None = None,
        v_cache: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = x
        h = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        qkv_all = linear_any(self.qkv_proj, h, profile="mtp_qkv")
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
            att = attention_decode_gated_any(
                q.contiguous(),
                k_live,
                v_live,
                gate.contiguous(),
                len(k_cache),
                self.cfg.attention_head_dim ** -0.5,
            )
            h = residual + linear_any(self.o_proj, att, profile="mtp_o")
            residual = h
            h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
            return residual + self.mlp(h)
        att_flat = (att.reshape(-1) * torch.sigmoid(gate.reshape(-1).to(att.dtype))).contiguous()
        h = residual + linear_any(self.o_proj, att_flat, profile="mtp_o")
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(h)

    def _next_token_hidden(
        self,
        hidden_state: torch.Tensor,
        input_token: torch.Tensor,
        *,
        pos: int,
        local_cache: _MTPLocalKVCache | None = None,
    ) -> torch.Tensor:
        emb = embed_lookup(self.model.embed, input_token).to(self.device, dtype=torch.float16).reshape(-1).contiguous()
        hidden = hidden_state.to(self.device, dtype=torch.float16).contiguous()
        x = qwen_rmsnorm_pair_cat(
            emb,
            self.pre_fc_norm_embedding,
            hidden,
            self.pre_fc_norm_hidden,
            self.cfg.rms_norm_eps,
        )
        x = linear_any(self.fc, x, profile="mtp_fc").to(self.device, dtype=torch.float16).contiguous()
        x = self._single_token_decoder_layer(x, pos=pos, local_cache=local_cache)
        return qwen_rmsnorm(x.contiguous(), self.norm, self.cfg.rms_norm_eps)

    def _next_token_hidden_legacy(
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
        x = linear_any(self.fc, x, profile="mtp_fc").to(self.device, dtype=torch.float16).contiguous()
        x = self._single_token_decoder_layer_legacy(x, pos=pos, k_cache=k_cache, v_cache=v_cache)
        return qwen_rmsnorm(x.contiguous(), self.norm, self.cfg.rms_norm_eps)

    def _next_token_hidden_batch(
        self,
        hidden_state: torch.Tensor,
        input_tokens: torch.Tensor,
        *,
        positions: torch.Tensor,
        local_cache: _MTPBatchedLocalKVCache | None = None,
    ) -> torch.Tensor:
        hidden = hidden_state.to(device=self.device, dtype=torch.float16).contiguous()
        if hidden.ndim != 2:
            raise ValueError("batched MTP hidden_state must be [batch, hidden]")
        tokens = input_tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        if int(tokens.numel()) != int(hidden.size(0)):
            raise ValueError("batched MTP tokens must match hidden batch")
        pos = positions.to(device=self.device, dtype=torch.long).reshape(-1)
        if int(pos.numel()) != int(hidden.size(0)):
            raise ValueError("batched MTP positions must match hidden batch")
        emb = embed_lookup_batch(self.model.embed, tokens, self.device).to(dtype=torch.float16)
        x = qwen_rmsnorm_pair_cat(
            emb,
            self.pre_fc_norm_embedding,
            hidden,
            self.pre_fc_norm_hidden,
            self.cfg.rms_norm_eps,
        )
        x = linear_any(self.fc, x, profile="mtp_fc").to(dtype=torch.float16).contiguous()

        residual = x
        h = qwen_rmsnorm(x, self.input_norm, self.cfg.rms_norm_eps)
        qkv_all = linear_any(self.qkv_proj, h, profile="mtp_qkv")
        q_all, k_all, v_all = torch.split(qkv_all, self.qkv_split, dim=-1)
        q_heads = q_all.view(hidden.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
        q, gate = torch.chunk(q_heads, 2, dim=-1)
        k = k_all.view(hidden.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        v = v_all.view(hidden.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        ratio = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        first_mtp_token = local_cache is not None and int(local_cache.length) == 0
        if local_cache is not None:
            q = qwen_rmsnorm_lastdim(q.contiguous(), self.q_norm, self.cfg.rms_norm_eps)
            k = qwen_rmsnorm_lastdim(k.contiguous(), self.k_norm, self.cfg.rms_norm_eps)
            q, k = apply_rope_decode_batch(
                q,
                k,
                positions=pos,
                rope_dim=self.cfg.rope_dim,
                rope_theta=self.cfg.rope_theta,
            )
            local_cache.append(k.contiguous(), v.contiguous())
        if local_cache is None or first_mtp_token:
            att_flat = sigmoid_mul_repeat_kv(v, gate, ratio)
        else:
            with decode_profile_scope("mtp_local_attention"):
                att_flat = local_cache.attend(q.contiguous(), gate.contiguous(), self.cfg.attention_head_dim ** -0.5)
        h = residual + linear_any(self.o_proj, att_flat, profile="mtp_o")
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        x = residual + self.mlp(h)
        return qwen_rmsnorm(x.contiguous(), self.norm, self.cfg.rms_norm_eps)

    def _second_token_hidden(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        return self._next_token_hidden(raw_hidden, first_token, pos=pos)

    @torch.no_grad()
    def logits_for_second_token(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        x = self._second_token_hidden(raw_hidden, first_token, pos=pos)
        # LowBitMarlinTensor reuses per-batch output buffers. Return an owned
        # tensor so candidate logits cannot alias later target/model projections.
        return linear_any(self.model.lm_head, x, profile="mtp_lm_head_full").contiguous().clone()

    @torch.no_grad()
    def argmax_for_second_token(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        x = self._second_token_hidden(raw_hidden, first_token, pos=pos)
        # Candidate generation only needs argmax. Avoid cloning the full
        # 248K-vocab logits buffer on the speculative hot path.
        return linear_argmax_any(self.model.lm_head, x, profile="mtp_lm_head_argmax").reshape(())

    @torch.no_grad()
    def argmax_first_batch(
        self,
        raw_hidden: torch.Tensor,
        first_tokens: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = raw_hidden.to(device=self.device, dtype=torch.float16).contiguous()
        if hidden.ndim != 2:
            raise ValueError("raw_hidden must be [batch, hidden]")
        tokens = first_tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        if tokens.numel() != hidden.size(0):
            raise ValueError("first_tokens batch size must match raw_hidden")
        if positions is None:
            positions = torch.zeros((hidden.size(0),), device=self.device, dtype=torch.long)
        x = self._next_token_hidden_batch(hidden, tokens, positions=positions, local_cache=None)
        return linear_argmax_any(self.model.lm_head, x, profile="mtp_lm_head_argmax").to(device=x.device, dtype=torch.long)

    @torch.no_grad()
    def argmax_sequence(
        self,
        raw_hidden: torch.Tensor,
        first_token: torch.Tensor,
        *,
        pos: int,
        max_draft: int,
    ) -> torch.Tensor:
        """Generate NEXTN-style draft tokens with the native Qwen MTP layer."""
        if max_draft <= 0:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        if _env_enabled("LANGBURST_MTP_LEGACY_LIST_CACHE", "1"):
            tokens = torch.empty((max_draft,), device=self.device, dtype=torch.long)
            hidden = raw_hidden
            token = first_token.reshape(()).to(device=self.device, dtype=torch.long)
            k_cache: list[torch.Tensor] = []
            v_cache: list[torch.Tensor] = []
            for step in range(max_draft):
                hidden = self._next_token_hidden_legacy(
                    hidden,
                    token,
                    pos=pos + step,
                    k_cache=k_cache,
                    v_cache=v_cache,
                )
                token = linear_argmax_any(self.model.lm_head, hidden, profile="mtp_lm_head_argmax").reshape(())
                tokens[step] = token
            return tokens
        tokens = torch.empty((max_draft,), device=self.device, dtype=torch.long)
        hidden = raw_hidden
        token = first_token.reshape(()).to(device=self.device, dtype=torch.long)
        local_cache = _MTPLocalKVCache(
            max_tokens=max_draft,
            kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.attention_head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        for step in range(max_draft):
            hidden = self._next_token_hidden(
                hidden,
                token,
                pos=pos + step,
                local_cache=local_cache,
            )
            if marlin_internal_argmax_enabled():
                token = linear_argmax_any(self.model.lm_head, hidden, out=tokens.narrow(0, step, 1), profile="mtp_lm_head_argmax").reshape(())
            else:
                token = linear_argmax_any(self.model.lm_head, hidden, profile="mtp_lm_head_argmax").reshape(())
                tokens[step] = token
        return tokens

    @torch.no_grad()
    def argmax_sequence_batch(
        self,
        raw_hidden: torch.Tensor,
        first_tokens: torch.Tensor,
        *,
        positions: torch.Tensor,
        max_draft: int,
    ) -> torch.Tensor:
        if max_draft <= 0:
            return torch.empty((int(raw_hidden.size(0)), 0), device=self.device, dtype=torch.long)
        hidden = raw_hidden.to(device=self.device, dtype=torch.float16).contiguous()
        batch = int(hidden.size(0))
        token = first_tokens.to(device=self.device, dtype=torch.long).reshape(batch)
        base_pos = positions.to(device=self.device, dtype=torch.long).reshape(batch)
        tokens_t = torch.empty((int(max_draft), batch), device=self.device, dtype=torch.long)
        local_cache = _MTPBatchedLocalKVCache(
            batch=batch,
            max_tokens=max_draft,
            kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.attention_head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        for step in range(int(max_draft)):
            hidden = self._next_token_hidden_batch(
                hidden,
                token,
                positions=base_pos + int(step),
                local_cache=local_cache,
            )
            if marlin_internal_argmax_enabled():
                token = linear_argmax_any(self.model.lm_head, hidden, out=tokens_t[step], profile="mtp_lm_head_argmax").reshape(batch)
            else:
                token = linear_argmax_any(self.model.lm_head, hidden, profile="mtp_lm_head_argmax").reshape(batch)
                tokens_t[step].copy_(token)
        return tokens_t.t().contiguous()


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
        first_token_t = torch.as_tensor(first_token, device=self.mtp.device, dtype=torch.long)
        return self.mtp.argmax_sequence(
            raw_hidden,
            first_token_t,
            pos=int(pos),
            max_draft=int(request.max_draft),
        )

    @torch.no_grad()
    def propose_tensors_batch(self, requests: Sequence[DraftRequest]) -> torch.Tensor:
        if not requests:
            return torch.empty((0, 0), device=self.mtp.device, dtype=torch.long)
        max_draft = int(requests[0].max_draft)
        if any(int(req.max_draft) != max_draft for req in requests):
            raise ValueError("batched native_mtp1 requires a uniform max_draft")
        if max_draft <= 0:
            return torch.empty((len(requests), 0), device=self.mtp.device, dtype=torch.long)
        raw_hiddens: list[torch.Tensor] = []
        first_token_values: list[int] = []
        positions: list[int] = []
        for req in requests:
            signals = req.signals or {}
            raw_hidden = signals.get("raw_hidden")
            first_token = signals.get("first_token")
            pos = signals.get("pos")
            if raw_hidden is None or first_token is None or pos is None:
                raise ValueError("native_mtp1 batch proposer requires raw_hidden, first_token, and pos signals")
            raw_hiddens.append(raw_hidden.reshape(-1).to(device=self.mtp.device, dtype=torch.float16))
            if torch.is_tensor(first_token):
                first_token_values.append(int(first_token.reshape(()).detach().cpu().item()))
            else:
                first_token_values.append(int(first_token))
            positions.append(int(pos))
        hidden = raw_hiddens[0].reshape(1, -1).contiguous() if len(raw_hiddens) == 1 else torch.stack(raw_hiddens, dim=0).contiguous()
        token_tensor = torch.tensor(first_token_values, device=self.mtp.device, dtype=torch.long)
        position_tensor = torch.tensor(positions, device=self.mtp.device, dtype=torch.long)
        if max_draft == 1:
            tokens = self.mtp.argmax_first_batch(hidden, token_tensor, positions=position_tensor)
            return tokens.reshape(len(requests), 1).contiguous()
        return self.mtp.argmax_sequence_batch(
            hidden,
            token_tensor,
            positions=position_tensor,
            max_draft=max_draft,
        )[:, :max_draft].contiguous()

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
