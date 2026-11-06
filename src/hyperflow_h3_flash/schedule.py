# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# `shift_sigmas` is derived from `MiniMaxH3Scheduler.set_timesteps` in diffusers
# (`schedulers/scheduling_minimax_h3.py`), Copyright 2025 The MiniMax authors and The HuggingFace Team, and
# `build_row_time_pairs` from `MiniMaxH3SetTimestepsStep.build_row_timesteps` in diffusers
# (`modular_pipelines/minimax_h3/before_denoise.py`), Copyright 2026 The MiniMax and HuggingFace Teams; both Apache
# License 2.0. See THIRD_PARTY_NOTICES.md.
"""The 4 / 6 / 8-step sigma grids and the per-row ``(t, r)`` plan.

MiniMax-H3 runs one packed sequence (text, conditioning rows, audio, video) through one forward per step, on the
``t = 1 - sigma`` axis (``t = 1`` is clean). HyperFlow H3 Flash also gives every row the endpoint ``r`` its step aims
at, ``r_i = 1 - sigma_{i+1}``. Only the raw grids and the endpoint rule are HyperFlow's; the per-modality shifts (12
video, 3 audio) and which rows are pinned where are the official MiniMax-H3 recipe.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Raw (unshifted) sigma grids by step count: 4 favours speed, 6 (the default) balances speed and quality, 8 favours
#: quality. Each has ``steps + 1`` points and needs ``steps`` model evaluations.
SIGMA_GRIDS: dict[int, tuple[float, ...]] = {
    4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
    6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
    8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
}
DEFAULT_STEPS = 6

#: The scheduler shifts the model samples with; the official MiniMax-H3 schedulers ship the same values.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def sigma_grid(steps: int | None = None) -> tuple[float, ...]:
    """The raw grid for ``steps`` model evaluations (default 6); only the step counts in `SIGMA_GRIDS` exist."""
    steps = DEFAULT_STEPS if steps is None else int(steps)
    if steps not in SIGMA_GRIDS:
        raise ValueError(f"HyperFlow H3 Flash samples {sorted(SIGMA_GRIDS)} steps, got {steps}")
    return SIGMA_GRIDS[steps]


def shift_sigmas(sigmas: Sequence[float] | torch.Tensor, shift: float) -> torch.Tensor:
    """``shift * s / (1 + (shift - 1) * s)`` in float32, the formula `MiniMaxH3Scheduler` applies to its grid."""
    grid = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
    return shift * grid / (1.0 + (shift - 1.0) * grid)


def build_row_time_pairs(
    *,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_time: tuple[float, float],
    audio_time: tuple[float, float],
    condition_video_timestep: float,
    condition_audio_timestep: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give every row of the packed sequence a ``(t, r)`` pair and reduce them to the transformer's inputs.

    The official ``build_row_timesteps`` with the endpoint added; the pin rule is unchanged:

    * generated video rows step ``video_time``; text rows never reach an output head and follow video;
    * generated audio rows step ``audio_time`` on their own schedule;
    * keyframe and image / video reference rows are held at the noise-augmentation level, and an anchor goes
      nowhere: ``r == t``;
    * reference audio rows are never noised: ``t == r == 1``.

    Pairs are made unique as pairs, not by ``t``: at step 0 video and audio share ``t = 0`` but aim at different ``r``.

    Returns:
        ``(timestep, endpoint_timestep, timestep_indices)``: the distinct pairs, sorted, as two float32 tensors, and
        the index of every row into them.
    """
    video_indices = video_indices.cpu()
    audio_indices = audio_indices.cpu()
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)

    pairs = torch.tensor(video_time, dtype=torch.float32).repeat(sequence_length, 1)
    pairs[video_indices[:num_condition_video_rows]] = float(condition_video_timestep)
    pairs[audio_indices[num_condition_audio_rows:]] = torch.tensor(audio_time, dtype=torch.float32)
    pairs[audio_indices[:num_condition_audio_rows]] = float(condition_audio_timestep)

    unique, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


__all__ = [
    "AUDIO_SHIFT",
    "DEFAULT_STEPS",
    "SIGMA_GRIDS",
    "VIDEO_SHIFT",
    "build_row_time_pairs",
    "shift_sigmas",
    "sigma_grid",
]
