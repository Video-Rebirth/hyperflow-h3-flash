# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# The two blocks below subclass and rework `MiniMaxH3SetTimestepsStep` and `MiniMaxH3LoopDenoiser` from diffusers
# (`modular_pipelines/minimax_h3/`), Copyright 2026 The MiniMax and HuggingFace Teams, Apache License 2.0.
# See THIRD_PARTY_NOTICES.md.
"""The official MiniMax-H3 workflow with two blocks swapped for the few-step two-time sampler.

* ``denoise.set_timesteps``: the 4 / 6 / 8-step grid instead of ``linspace(1, 0, num_inference_steps)``, and one
  ``(timestep, endpoint_timestep, timestep_indices)`` triple per step instead of ``(timestep, timestep_indices)``.
* ``denoise.denoise.denoiser``: passes ``endpoint_timestep`` to the transformer.

Text encoding, VAE encoding, the packed layout, the Euler update and decoding are the official blocks, unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from diffusers.modular_pipelines import InputParam, OutputParam, PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser

from .schedule import (
    AUDIO_SHIFT,
    DEFAULT_STEPS,
    SIGMA_GRIDS,
    VIDEO_SHIFT,
    build_row_time_pairs,
    shift_sigmas,
    sigma_grid,
)

#: The official workflows. `t2va` and `fl2va` run one checkpoint partition, `ref2va` its own (`transformer_ref`).
WORKFLOWS = ("t2va", "fl2va", "ref2va")


class FlashSetTimestepsStep(MiniMaxH3SetTimestepsStep):
    """Official ``set_timesteps`` with a 4 / 6 / 8-step grid and a ``(t, r)`` pair per row."""

    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "HyperFlow H3 Flash: installs the 4, 6 (default) or 8-step sigma grid on the two schedulers (video shift "
            "12, audio shift 3) and stages one `(timestep, endpoint_timestep, timestep_indices)` triple per step."
        )

    @property
    def inputs(self) -> list[InputParam]:
        official = [param for param in super().inputs if param.name != "num_inference_steps"]
        steps = InputParam(
            name="num_inference_steps",
            type_hint=int,
            default=None,
            description=f"One of {sorted(SIGMA_GRIDS)}; default {DEFAULT_STEPS}.",
        )
        return [steps, *official]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps of the video schedule."),
            OutputParam("audio_timesteps", type_hint=torch.Tensor, description="Timesteps of the audio schedule."),
            OutputParam(
                "row_timestep_plan",
                type_hint=list,
                description="One `(timestep, endpoint_timestep, timestep_indices)` triple per step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        grid = sigma_grid(block_state.num_inference_steps)

        device = components._execution_device
        video_sigmas = shift_sigmas(grid, VIDEO_SHIFT)
        audio_sigmas = shift_sigmas(grid, AUDIO_SHIFT)
        components.scheduler.set_timesteps(sigmas=video_sigmas, device=device)
        components.audio_scheduler.set_timesteps(sigmas=audio_sigmas, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps

        # `t` is read back from the schedulers, so the plan matches the values the update step uses bit for bit.
        video_t, audio_t = block_state.timesteps.cpu(), block_state.audio_timesteps.cpu()
        video_r, audio_r = 1.0 - video_sigmas[1:], 1.0 - audio_sigmas[1:]
        plan = []
        for i in range(len(grid) - 1):
            triple = build_row_time_pairs(
                video_indices=block_state.video_indices,
                audio_indices=block_state.audio_indices,
                num_condition_video_rows=block_state.num_condition_video_rows,
                num_condition_audio_rows=block_state.num_condition_audio_rows,
                num_text_tokens=block_state.text_indices.numel(),
                video_time=(float(video_t[i]), float(video_r[i])),
                audio_time=(float(audio_t[i]), float(audio_r[i])),
                condition_video_timestep=max(float(video_t[i]), components.keyframe_noise_aug),
            )
            plan.append(tuple(tensor.to(device) for tensor in triple))
        block_state.row_timestep_plan = plan

        self.set_block_state(state, block_state)
        return components, state


class FlashLoopDenoiser(MiniMaxH3LoopDenoiser):
    """Official loop denoiser that also hands the step's ``endpoint_timestep`` to the transformer."""

    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return "HyperFlow H3 Flash: one forward per step, conditioned on the step's `(t, r)` pairs."

    @property
    def inputs(self) -> list[InputParam]:
        plan = InputParam(
            name="row_timestep_plan",
            type_hint=list,
            required=True,
            description="One `(timestep, endpoint_timestep, timestep_indices)` triple per step.",
        )
        return [plan if param.name == "row_timestep_plan" else param for param in super().inputs]

    @torch.no_grad()
    def __call__(self, components: Any, block_state: Any, i: int, t: torch.Tensor):
        transformer = getattr(components, self.transformer_name)
        timestep, endpoint_timestep, timestep_indices = block_state.row_timestep_plan[i]
        parameters = inspect.signature(transformer.forward).parameters
        layout = {k: v for k, v in block_state.denoiser_input_fields.items() if k in parameters}
        block_state.noise_pred, block_state.audio_noise_pred = transformer(
            hidden_states=block_state.latents[None],
            audio_hidden_states=block_state.audio_latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=timestep,
            endpoint_timestep=endpoint_timestep,
            timestep_indices=timestep_indices,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
            **layout,
        )
        return components, block_state


def _swap(container: Any, counts: dict[str, int]) -> None:
    for name, block in list((getattr(container, "sub_blocks", None) or {}).items()):
        if isinstance(block, MiniMaxH3SetTimestepsStep):
            container.sub_blocks[name] = block if isinstance(block, FlashSetTimestepsStep) else FlashSetTimestepsStep()
            counts["set_timesteps"] += 1
        elif isinstance(block, MiniMaxH3LoopDenoiser):
            if not isinstance(block, FlashLoopDenoiser):
                container.sub_blocks[name] = FlashLoopDenoiser(transformer_name=block.transformer_name)
            counts["denoiser"] += 1
        else:
            _swap(block, counts)


def flash_blocks(workflow: str = "t2va") -> SequentialPipelineBlocks:
    """The official MiniMax-H3 ``workflow`` (one of `WORKFLOWS`) with the two HyperFlow H3 Flash blocks.

    ``ref2va`` keeps its denoiser's ``transformer_ref`` component. Its reference rows follow the keyframe rule:
    image and video references are held at ``keyframe_noise_aug`` (``t == r``), reference audio at ``t == r == 1``.
    """
    if workflow not in WORKFLOWS:
        raise ValueError(f"unknown workflow {workflow!r}; HyperFlow H3 Flash supports {WORKFLOWS}")
    blocks = MiniMaxH3Blocks().get_workflow(workflow)
    counts = {"set_timesteps": 0, "denoiser": 0}
    _swap(blocks, counts)
    if counts != {"set_timesteps": 1, "denoiser": 1}:
        raise RuntimeError(f"expected one set_timesteps and one denoiser block in {workflow}, found {counts}")
    return blocks


__all__ = ["WORKFLOWS", "FlashLoopDenoiser", "FlashSetTimestepsStep", "flash_blocks"]
