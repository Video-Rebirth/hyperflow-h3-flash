"""A tiny random HyperFlow H3 Flash transformer and a packed sequence for it. No weights are downloaded."""

from __future__ import annotations

import torch

TINY_CONFIG = dict(
    num_attention_heads=2,
    attention_head_dim=32,
    hidden_size=64,
    num_layers=3,
    num_refiner_layers=1,
    ffn_dim=128,
    in_channels=24,
    audio_in_channels=32,
    patch_size=(1, 2, 2),
    text_dim=48,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=16,
    rope_freq_dim=4,  # rotary dim = 3 axes * 2 * rope_freq_dim must fit in attention_head_dim
)

#: Block 1 is dropped, block 0 has low-rank attention, block 2 has INT8 attention and feed-forward.
COMPRESSION = dict(
    dropped_blocks=[1],
    lowrank_ranks={"transformer_blocks.0.attn.to_q": 8, "transformer_blocks.0.attn.to_out.0": 16},
    quantization={
        "method": "int8_convrot",
        "group": 16,
        "linears": ["transformer_blocks.2.attn.to_k", "transformer_blocks.2.ff.net.2"],
    },
)


def packed_inputs(config, *, num_text=3, num_condition_video=2, num_video=6, num_audio=4, seed=0) -> dict:
    """Transformer inputs for a packed sequence ``[text | audio | video (condition first)]``, all at one ``(t, r)``."""
    generator = torch.Generator().manual_seed(seed)
    video_patch_dim = config.in_channels * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    total_video = num_condition_video + num_video
    sequence_length = num_text + num_audio + total_video
    text_indices = torch.arange(0, num_text)
    audio_indices = torch.arange(num_text, num_text + num_audio)
    video_indices = torch.arange(num_text + num_audio, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices], token_tags[audio_indices], token_tags[video_indices] = 1, 2, 0
    return dict(
        hidden_states=torch.randn(1, total_video, video_patch_dim, generator=generator),
        audio_hidden_states=torch.randn(1, num_audio, config.audio_in_channels, generator=generator),
        encoder_hidden_states=torch.randn(1, num_text, config.text_dim, generator=generator),
        timestep=torch.tensor([0.3]),
        timestep_indices=torch.zeros(sequence_length, dtype=torch.long),
        token_tags=token_tags,
        position_ids=torch.rand(sequence_length, 3, dtype=torch.float64, generator=generator) * 8,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )
