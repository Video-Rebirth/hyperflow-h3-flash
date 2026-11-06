from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

from hyperflow_h3_flash.blocks import FlashLoopDenoiser, FlashSetTimestepsStep, flash_blocks
from hyperflow_h3_flash.schedule import AUDIO_SHIFT, DEFAULT_STEPS, SIGMA_GRIDS, VIDEO_SHIFT, shift_sigmas

SIGMAS = SIGMA_GRIDS[DEFAULT_STEPS]

KEYFRAME_NOISE_AUG = 0.999


def blocks_of(container, kind):
    found = []
    for block in (getattr(container, "sub_blocks", None) or {}).values():
        found += [block] if isinstance(block, kind) else blocks_of(block, kind)
    return found


@pytest.mark.parametrize(
    ("workflow", "transformer_name"),
    [("t2va", "transformer"), ("fl2va", "transformer"), ("ref2va", "transformer_ref")],
)
def test_flash_blocks_swap_exactly_two_blocks(workflow, transformer_name):
    blocks = flash_blocks(workflow)
    assert len(blocks_of(blocks, FlashSetTimestepsStep)) == 1
    (denoiser,) = blocks_of(blocks, FlashLoopDenoiser)
    assert denoiser.transformer_name == transformer_name


def test_unknown_workflow_is_rejected():
    with pytest.raises(ValueError, match="ref2va"):
        flash_blocks("i2v")


def run_set_timesteps(**inputs):
    components = SimpleNamespace(
        scheduler=MiniMaxH3Scheduler(shift=VIDEO_SHIFT),
        audio_scheduler=MiniMaxH3Scheduler(shift=AUDIO_SHIFT),
        keyframe_noise_aug=KEYFRAME_NOISE_AUG,
        _execution_device=torch.device("cpu"),
    )
    state = PipelineState()
    for key, value in inputs.items():
        state.set(key, value)
    FlashSetTimestepsStep()(components, state)
    return state.get("row_timestep_plan")


def test_plan_keeps_the_official_t_and_adds_the_next_grid_point_as_r():
    layout = dict(
        video_indices=torch.arange(7, 13),
        audio_indices=torch.arange(3, 7),
        text_indices=torch.arange(0, 3),
        num_condition_video_rows=2,
        num_condition_audio_rows=0,
    )
    plan = run_set_timesteps(**layout)
    assert len(plan) == DEFAULT_STEPS

    video_sigmas, audio_sigmas = shift_sigmas(SIGMAS, VIDEO_SHIFT), shift_sigmas(SIGMAS, AUDIO_SHIFT)
    generated_video, generated_audio = layout["video_indices"][2:], layout["audio_indices"]
    for i, (timestep, endpoint, indices) in enumerate(plan):
        official, official_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            layout["video_indices"],
            layout["audio_indices"],
            2,
            0,
            3,
            float(1 - video_sigmas[i]),
            float(1 - audio_sigmas[i]),
            max(float(1 - video_sigmas[i]), KEYFRAME_NOISE_AUG),
            1.0,
        )
        torch.testing.assert_close(timestep[indices], official[official_indices], rtol=0, atol=0)
        torch.testing.assert_close(endpoint[indices][generated_video], (1 - video_sigmas[i + 1]).expand(4))
        torch.testing.assert_close(endpoint[indices][generated_audio], (1 - audio_sigmas[i + 1]).expand(4))
        keyframes = indices[layout["video_indices"][:2]]
        torch.testing.assert_close(endpoint[keyframes], timestep[keyframes])


def test_reference_rows_are_held_clean():
    """``ref2va``: image / video reference rows sit at the noise-augmentation level, reference audio at ``t = 1``."""
    layout = dict(
        video_indices=torch.arange(8, 14),
        audio_indices=torch.arange(3, 8),
        text_indices=torch.arange(0, 3),
        num_condition_video_rows=2,
        num_condition_audio_rows=2,
    )
    for timestep, endpoint, indices in run_set_timesteps(**layout):
        references = indices[layout["video_indices"][:2]]
        reference_audio = indices[layout["audio_indices"][:2]]
        torch.testing.assert_close(timestep[references], torch.full((2,), KEYFRAME_NOISE_AUG))
        torch.testing.assert_close(endpoint[references], timestep[references])
        torch.testing.assert_close(timestep[reference_audio], torch.ones(2))
        torch.testing.assert_close(endpoint[reference_audio], torch.ones(2))


SMALL_LAYOUT = dict(
    video_indices=torch.arange(1, 3),
    audio_indices=torch.arange(3, 4),
    text_indices=torch.arange(0, 1),
    num_condition_video_rows=0,
    num_condition_audio_rows=0,
)


@pytest.mark.parametrize("steps", sorted(SIGMA_GRIDS))
def test_each_grid_runs_one_forward_per_interval(steps):
    plan = run_set_timesteps(num_inference_steps=steps, **SMALL_LAYOUT)
    assert len(plan) == steps
    video_sigmas = shift_sigmas(SIGMA_GRIDS[steps], VIDEO_SHIFT)
    for i, (_, endpoint, indices) in enumerate(plan):
        video_endpoints = endpoint[indices[SMALL_LAYOUT["video_indices"]]]
        torch.testing.assert_close(video_endpoints, (1 - video_sigmas[i + 1]).expand_as(video_endpoints))


def test_other_step_counts_are_rejected():
    with pytest.raises(ValueError, match=r"\[4, 6, 8\]"):
        run_set_timesteps(num_inference_steps=50, **SMALL_LAYOUT)
