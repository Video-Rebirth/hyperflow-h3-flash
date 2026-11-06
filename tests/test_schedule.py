import torch
from diffusers import MiniMaxH3Scheduler

from hyperflow_h3_flash.schedule import (
    AUDIO_SHIFT,
    DEFAULT_STEPS,
    SIGMA_GRIDS,
    VIDEO_SHIFT,
    build_row_time_pairs,
    shift_sigmas,
)

SIGMAS = SIGMA_GRIDS[DEFAULT_STEPS]


def test_shift():
    shifted = shift_sigmas(SIGMAS, VIDEO_SHIFT)
    assert shifted[0] == 1.0 and shifted[-1] == 0.0
    expected = VIDEO_SHIFT * SIGMAS[1] / (1 + (VIDEO_SHIFT - 1) * SIGMAS[1])
    torch.testing.assert_close(shifted[1], torch.tensor(expected, dtype=torch.float32))
    assert (shifted[1:] < shifted[:-1]).all()


def test_scheduler_keeps_the_shifted_grid():
    """The scheduler takes the grid as given; its timesteps are ``t = 1 - sigma`` on the same axis as ``r``."""
    sigmas = shift_sigmas(SIGMAS, VIDEO_SHIFT)
    scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    scheduler.set_timesteps(sigmas=sigmas)
    torch.testing.assert_close(scheduler.sigmas, sigmas)
    torch.testing.assert_close(scheduler.timesteps, 1 - sigmas[:-1])


def test_row_time_pairs():
    # [text x3 | audio: 1 reference + 3 generated | video: 2 keyframes + 4 generated]
    text, audio, video = torch.arange(0, 3), torch.arange(3, 7), torch.arange(7, 13)
    video_sigmas, audio_sigmas = shift_sigmas(SIGMAS, VIDEO_SHIFT), shift_sigmas(SIGMAS, AUDIO_SHIFT)
    video_time = (1 - video_sigmas[0].item(), 1 - video_sigmas[1].item())
    audio_time = (1 - audio_sigmas[0].item(), 1 - audio_sigmas[1].item())
    timestep, endpoint, indices = build_row_time_pairs(
        video_indices=video,
        audio_indices=audio,
        num_condition_video_rows=2,
        num_condition_audio_rows=1,
        num_text_tokens=3,
        video_time=video_time,
        audio_time=audio_time,
        condition_video_timestep=0.999,
    )
    rows = torch.stack([timestep[indices], endpoint[indices]], dim=1)
    torch.testing.assert_close(rows[text], torch.tensor([video_time] * 3))
    torch.testing.assert_close(rows[video[2:]], torch.tensor([video_time] * 4))
    torch.testing.assert_close(rows[audio[1:]], torch.tensor([audio_time] * 3))
    torch.testing.assert_close(rows[video[:2]], torch.full((2, 2), 0.999))
    torch.testing.assert_close(rows[audio[:1]], torch.ones(1, 2))
    # Step 0: video and audio share t = 0 but not r, so they are two pairs.
    assert video_time[0] == audio_time[0] and video_time[1] != audio_time[1]
    assert len(timestep) == 4
    assert torch.equal(torch.sort(timestep).values, timestep)
