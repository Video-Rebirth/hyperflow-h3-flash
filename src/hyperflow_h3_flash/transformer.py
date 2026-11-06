# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# `HyperFlowH3FlashTransformer3DModel.forward` is derived from `MiniMaxH3Transformer3DModel.forward` in diffusers
# (`models/transformers/transformer_minimax_h3.py`), Copyright 2025 The MiniMax Team and The HuggingFace Team,
# Apache License 2.0. See https://github.com/Video-Rebirth/hyperflow-h3-flash/blob/main/THIRD_PARTY_NOTICES.md.
"""The HyperFlow H3 Flash transformer: MiniMax-H3, compressed and conditioned on two times.

Four changes to `MiniMaxH3Transformer3DModel`, all driven by `config.json`:

* **Two-time conditioning.** An `endpoint_time_embedder` embeds the step's endpoint ``r`` next to the current time
  ``t``; every AdaLN reads ``temb(t, r) = emb_t(t) + hyperflow_gate * (emb_r(r) - emb_t(t))``.
* **Dropped blocks** (`dropped_blocks`): those blocks are identities and have no weights.
* **Low-rank linears** (`lowrank_ranks`): ``W ~= up @ down``, stored as `<name>.down.weight` (rank x in) and
  `<name>.up.weight` (out x rank). Ranks are zero-padded to multiples of 64, in the checkpoint or at load
  (`pad_lowrank_ranks`), so both GEMMs are aligned; `set_lowrank_rank_multiple` re-pads them at run time.
* **INT8 linears** (`quantization`): W8A8 with a Hadamard-rotated input, see `int8.py`.

Embeddings, AdaLN, the token refiner and the output heads are the base model's, in bf16 / fp32. The block
feed-forwards run in token chunks (`ChunkedFeedForward`, `set_ff_chunk`), which bounds their activations;
`set_block_chunk` slices every token-wise op of a block.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3TransformerOutput,
)
from diffusers.utils import apply_lora_scale
from torch import Tensor, nn

from .int8 import Int8RotLinear

#: Tokens per feed-forward chunk; a 5.2 s clip has about 38k, a 14.4 s clip about 105k.
FF_CHUNK_TOKENS = 16384
#: Low-rank ranks are padded to multiples of this at load, so both GEMMs of a low-rank linear are aligned.
RANK_MULTIPLE = 64


class LowRankLinear(nn.Module):
    """``y = up(down(x))``, the factorized replacement of a bias-free `nn.Linear`."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))

    @torch.no_grad()
    def pad_rank(self, multiple: int = RANK_MULTIPLE) -> bool:
        """Zero-pad the rank up to a multiple of ``multiple``: new rows of ``down`` and columns of ``up`` are zero, so
        ``up @ down`` is unchanged. Returns whether the rank changed."""
        rank = -(-self.rank // multiple) * multiple
        if rank == self.rank:
            return False
        extra = rank - self.rank
        down, up = self.down.weight, self.up.weight
        self.down.weight = nn.Parameter(torch.cat([down, down.new_zeros(extra, down.shape[1])]), requires_grad=False)
        self.up.weight = nn.Parameter(torch.cat([up, up.new_zeros(up.shape[0], extra)], dim=1), requires_grad=False)
        self.down.out_features = self.up.in_features = self.rank = rank
        return True


class ChunkedFeedForward(nn.Module):
    """A block's feed-forward run over at most ``chunk_tokens`` tokens at a time; it keeps the block's ``net``.

    The feed-forward is token-wise, so the chunks compute the same function; only the rounding of bf16 GEMMs can
    change with their row count. It bounds the SwiGLU input, ``2 * ffn_dim`` wide per token (6 GiB for a 14.4 s
    clip), to one chunk.
    """

    def __init__(self, net: nn.ModuleList, out_features: int, chunk_tokens: int) -> None:
        super().__init__()
        self.net, self.out_features, self.chunk_tokens = net, out_features, chunk_tokens

    def _forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[-2] <= self.chunk_tokens:
            return self._forward(hidden_states)
        out = hidden_states.new_empty(*hidden_states.shape[:-1], self.out_features)
        for start in range(0, hidden_states.shape[-2], self.chunk_tokens):
            stop = start + self.chunk_tokens
            out[..., start:stop, :] = self._forward(hidden_states[..., start:stop, :])
        return out


class DroppedBlock(nn.Module):
    """A removed transformer block: the hidden states go straight to the next block.

    The signature is the block's own: context parallel hooks address `adaln_indices` by name.
    """

    def forward(self, hidden_states: Tensor, temb=None, adaln_indices=None, rotary_emb=None, attention_mask=None):
        return hidden_states


def _replace_linear(model: nn.Module, name: str, make: type[nn.Module], *args: Any) -> None:
    owner_name, _, attr = name.rpartition(".")
    owner = model.get_submodule(owner_name)
    dense = owner[int(attr)] if isinstance(owner, nn.ModuleList) else getattr(owner, attr)
    if type(dense) is not nn.Linear or dense.bias is not None:
        raise TypeError(f"{name} is not a bias-free nn.Linear")
    replacement = make(dense.in_features, dense.out_features, *args)
    if isinstance(owner, nn.ModuleList):
        owner[int(attr)] = replacement
    else:
        setattr(owner, attr, replacement)


class HyperFlowH3FlashTransformer3DModel(MiniMaxH3Transformer3DModel):
    """`MiniMaxH3Transformer3DModel` conditioned on ``(t, r)``, with dropped blocks, low-rank and INT8 linears."""

    # Matched against whole dotted name parts at load time, so `time_embedder` does not cover the endpoint embedder.
    _keep_in_fp32_modules: ClassVar[list[str]] = [
        *MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        "endpoint_time_embedder",
        "wscale",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 16,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        hyperflow_gate: float = 0.25,
        dropped_blocks: list[int] | None = None,
        lowrank_ranks: dict[str, int] | None = None,
        quantization: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        self.endpoint_time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
        )

        dropped = set(dropped_blocks or [])
        for index, block in enumerate(self.transformer_blocks):
            if index in dropped:
                self.transformer_blocks[index] = DroppedBlock()
            else:
                block.ff = ChunkedFeedForward(block.ff.net, block.ff.net[-1].out_features, FF_CHUNK_TOKENS)
        for name, rank in (lowrank_ranks or {}).items():
            if int(name.split(".")[1]) not in dropped:
                _replace_linear(self, name, LowRankLinear, int(rank))
        if quantization:
            if quantization.get("method") != "int8_convrot":
                raise ValueError(f"unsupported quantization method {quantization.get('method')!r}")
            for name in quantization["linears"]:
                _replace_linear(self, name, Int8RotLinear, int(quantization["group"]))
        for block in self.transformer_blocks:
            attn = getattr(block, "attn", None)
            for name in ("to_q", "to_k"):
                if isinstance(layer := getattr(attn, name, None), Int8RotLinear):
                    layer.keep_input = True

    def pad_lowrank_ranks(self, multiple: int = RANK_MULTIPLE) -> int:
        """Zero-pad every low-rank linear to a rank multiple of ``multiple`` (see `LowRankLinear.pad_rank`) for a
        checkpoint stored at its unpadded ranks. Returns how many linears changed."""
        ranks = dict(self.config.lowrank_ranks or {})
        padded = 0
        for name, module in self.named_modules():
            if isinstance(module, LowRankLinear) and module.pad_rank(multiple):
                ranks[name] = module.rank
                padded += 1
        if padded:
            self.register_to_config(lowrank_ranks=ranks)
        return padded

    def time_embedding(self, timestep: Tensor, endpoint_timestep: Tensor) -> Tensor:
        """``temb`` of every distinct ``(t, r)`` pair, at the time embedders' precision (fp32)."""
        if endpoint_timestep.shape != timestep.shape:
            raise ValueError(f"endpoint_timestep {list(endpoint_timestep.shape)} != timestep {list(timestep.shape)}")
        dtype = get_parameter_dtype(self.time_embedder)
        current = self.time_embedder(self.time_proj(timestep).to(dtype))
        endpoint = self.endpoint_time_embedder(self.time_proj(endpoint_timestep).to(dtype))
        return current + self.config.hyperflow_gate * (endpoint - current)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: Tensor,
        audio_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        timestep_indices: Tensor,
        token_tags: Tensor,
        position_ids: Tensor,
        video_indices: Tensor,
        audio_indices: Tensor,
        text_indices: Tensor,
        endpoint_timestep: Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[Tensor, Tensor]:
        """The base forward plus ``endpoint_timestep``: ``(timestep[i], endpoint_timestep[i])`` is the ``i``-th pair.

        ``timestep_indices`` maps every row of the packed sequence to its pair. Build the pairs with
        `hyperflow_h3_flash.schedule.build_row_time_pairs`, or run the model through `flash_blocks(...)`.
        """
        if endpoint_timestep is None:
            raise ValueError(
                "HyperFlow H3 Flash needs `endpoint_timestep`, the endpoint `r` of every `(t, r)` pair: it is a "
                "few-step model and has no single-time mode. Run it through `hyperflow_h3_flash.flash_blocks(...)`."
            )
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors.")

        rotary_emb = self.rope(position_ids)

        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_embedding(timestep, endpoint_timestep)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)

        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def _used_rank(layer: LowRankLinear) -> int:
    """The rank without the zero padding (trailing zero rows of ``down`` and columns of ``up``)."""
    used = (layer.down.weight.detach().abs().amax(1) != 0) | (layer.up.weight.detach().abs().amax(0) != 0)
    nonzero = used.nonzero()
    return int(nonzero[-1]) + 1 if len(nonzero) else 1


def set_lowrank_rank_multiple(model: nn.Module, multiple: int) -> int:
    """Re-pad every low-rank linear to a rank multiple of ``multiple`` (0: no padding); returns the layers changed.

    Zero rows and columns add exact zeros, so any multiple computes the same function; the GEMM shapes change, and with
    them the bf16 rounding (8 is 2 to 7 % faster than 64 on H200). Run time only: the config still lists the stored
    ranks, so do not `save_pretrained` the re-padded model.
    """
    changed = 0
    for layer in model.modules():
        if not isinstance(layer, LowRankLinear):
            continue
        rank = _used_rank(layer)
        target = -(-rank // multiple) * multiple if multiple > 0 else rank
        if target == layer.rank:
            continue
        down, up = layer.down.weight.detach(), layer.up.weight.detach()
        if target < layer.rank:
            down, up = down[:target], up[:, :target]
        else:
            down = torch.cat([down, down.new_zeros(target - layer.rank, down.shape[1])])
            up = torch.cat([up, up.new_zeros(up.shape[0], target - layer.rank)], dim=1)
        layer.down.weight = nn.Parameter(down.contiguous(), requires_grad=False)
        layer.up.weight = nn.Parameter(up.contiguous(), requires_grad=False)
        layer.rank = layer.down.out_features = layer.up.in_features = target
        changed += 1
    return changed


def set_ff_chunk(model: nn.Module, tokens: int) -> None:
    """Run every feed-forward over at most ``tokens`` tokens at a time (0: whole sequence)."""
    for module in model.modules():
        if isinstance(module, ChunkedFeedForward):
            module.chunk_tokens = tokens if tokens > 0 else sys.maxsize


def chunk_block(block: nn.Module, chunk: int) -> None:
    """Run every token-wise op of a transformer block over ``chunk``-token slices; only the attention sees all of q /
    k / v.

    The unchunked block holds about a dozen ``(seq, hidden)`` tensors at once (each modulation picked per token, the
    modulated norm, the residual sums), 1.2 GB apiece at 108k tokens. Per-row results are the same up to the bf16 GEMM
    rounding for the smaller row count. Apply before context parallelism and offloading, which hook the block.
    """
    import diffusers.models.transformers.transformer_minimax_h3 as h3

    attn = block.attn

    def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
        batch, seq, _ = hidden_states.shape
        cos, sin = rotary_emb
        qkv = None
        for start in range(0, seq, chunk):
            rows = slice(start, min(start + chunk, seq))
            idx = adaln_indices[rows]
            h = block.norm1(hidden_states[:, rows])
            h = h * (1.0 + scale_msa.index_select(0, idx)) + shift_msa.index_select(0, idx)
            parts = []
            for proj, norm in ((attn.to_q, attn.norm_q), (attn.to_k, attn.norm_k), (attn.to_v, None)):
                x = proj(h).unflatten(-1, (attn.heads, -1))
                if norm is not None:
                    x = h3._apply_rotary_emb(norm(x), cos[rows], sin[rows])
                parts.append(x)
            if qkv is None:
                qkv = [x.new_empty((batch, seq, *x.shape[2:])) for x in parts]
            for full, x in zip(qkv, parts, strict=True):
                full[:, rows] = x
            del h, parts, x
        out = h3.dispatch_attention_fn(
            *qkv,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=attn.processor._attention_backend,
            parallel_config=attn.processor._parallel_config,
        )
        dtype = qkv[0].dtype
        del qkv
        result = torch.empty_like(hidden_states)
        for start in range(0, seq, chunk):
            rows = slice(start, min(start + chunk, seq))
            idx = adaln_indices[rows]
            a = attn.to_out[1](attn.to_out[0](out[:, rows].flatten(2, 3).to(dtype)))
            h = hidden_states[:, rows] + gate_msa.index_select(0, idx) * a
            n = block.norm2(h)
            n = n * (1.0 + scale_mlp.index_select(0, idx)) + shift_mlp.index_select(0, idx)
            result[:, rows] = h + gate_mlp.index_select(0, idx) * block.ff(n)
            del a, h, n
        return result

    block.forward = forward


def set_block_chunk(model: nn.Module, tokens: int) -> None:
    """`chunk_block` on every kept block (0: off). Call once, before context parallelism and offloading."""
    if tokens > 0:
        for block in model.transformer_blocks:
            if not isinstance(block, DroppedBlock):
                chunk_block(block, tokens)


__all__ = [
    "RANK_MULTIPLE",
    "ChunkedFeedForward",
    "DroppedBlock",
    "HyperFlowH3FlashTransformer3DModel",
    "LowRankLinear",
    "chunk_block",
    "set_block_chunk",
    "set_ff_chunk",
    "set_lowrank_rank_multiple",
]
