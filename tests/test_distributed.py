import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.hooks import ModelHook, add_hook_to_module
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines import PipelineState

from hyperflow_h3_flash.distributed import (
    apply_state_changes,
    broadcast_from_rank0,
    enable_parallel_vae,
    encode_tile_plan,
    holds_text_encoder,
    resolve_num_gpus,
    state_changes,
    text_encoder_on_rank0,
    tile_plan,
    tile_source,
)

#: A 4x-downsampling `AutoencoderKLMiniMaxH3` (diffusers' own test config) with 16-pixel tiles, 8 pixels of overlap.
TINY_VAE = dict(
    latent_channels=4,
    block_out_channels=(8, 16),
    layers_per_block=1,
    spatial_downsample_factors=(2, 2),
    temporal_downsample_factors=(2, 2),
    norm_num_groups=8,
    decoder_num_layers=2,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=8,
    decoder_num_register_tokens=2,
    decoder_ffn_mult=2,
    latents_mean=(0.0,) * 4,
    latents_std=(1.0,) * 4,
)
#: 40x56 pixels is a 4x6 grid of tiles; 16x16 is one tile, fewer than the ranks.
VAE_FRAMES = ((40, 56), (16, 16))


class TiledVae:
    """The tile geometry of diffusers' `AutoencoderKLMiniMaxH3`, `_split_tiles` copied verbatim."""

    spatial_compression_ratio = 16
    tile_sample_min_height = tile_sample_min_width = 256
    tile_sample_min_overlap_height = tile_sample_min_overlap_width = 64

    def _split_tiles(self, length, tile_size, min_overlap):
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps


@pytest.fixture
def one_process_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def encoded_state() -> tuple[PipelineState, dict]:
    state = PipelineState()
    state.set("prompt", "a fox")
    state.set("num_frames", 124)
    before = dict(state.values)
    state.set("prompt_embeds", torch.randn(1, 7, 16), kwargs_type="denoiser_input_fields")
    state.set("prompt_embeds_mask", torch.ones(1, 7, dtype=torch.bool), kwargs_type="denoiser_input_fields")
    state.set("num_frames", 125)
    return state, before


def test_resolve_num_gpus():
    assert resolve_num_gpus("auto", visible_gpus=8) == 4
    assert resolve_num_gpus("auto", visible_gpus=2) == 2
    assert resolve_num_gpus("auto", visible_gpus=8, world_size=2) == 2
    assert resolve_num_gpus(4, visible_gpus=4) == 4
    with pytest.raises(ValueError, match="visible"):
        resolve_num_gpus(4, visible_gpus=2)
    with pytest.raises(ValueError, match="56 attention heads"):
        resolve_num_gpus(3, visible_gpus=4)


def test_holds_text_encoder_follows_offload_and_rank(monkeypatch):
    assert holds_text_encoder("stream") is True  # single GPU
    monkeypatch.setattr("hyperflow_h3_flash.distributed.parallel_world_size", lambda: 4)
    monkeypatch.setattr("hyperflow_h3_flash.distributed.parallel_rank", lambda: 2)
    assert holds_text_encoder("none", text_encoder_fsdp=True) is True
    assert holds_text_encoder("none", text_encoder_fsdp=False) is False
    assert holds_text_encoder("stream", text_encoder_fsdp=True) is False
    monkeypatch.setattr("hyperflow_h3_flash.distributed.parallel_rank", lambda: 0)
    assert holds_text_encoder("stream", text_encoder_fsdp=True) is True


def test_tile_plan_covers_a_768p_frame():
    tiles, columns, y_overlaps, x_overlaps = tile_plan(TiledVae(), 768 // 16, 1344 // 16)
    assert (len(tiles), columns) == (4 * 7, 7)
    assert all((ys.stop - ys.start, xs.stop - xs.start) == (16, 16) for ys, xs in tiles)
    assert (tiles[-1][0].stop, tiles[-1][1].stop) == (48, 84)
    assert (len(y_overlaps), len(x_overlaps)) == (3, 6)


def test_encode_tile_plan_covers_a_768p_frame():
    tiles, columns, y_overlaps, x_overlaps = encode_tile_plan(TiledVae(), 768, 1344)
    assert (len(tiles), columns) == (4 * 7, 7)
    assert all((ys.stop - ys.start, xs.stop - xs.start) == (256, 256) for ys, xs in tiles)
    assert (tiles[-1][0].stop, tiles[-1][1].stop) == (768, 1344)
    _, _, pixel_y_overlaps, pixel_x_overlaps = tile_plan(TiledVae(), 768 // 16, 1344 // 16)
    assert y_overlaps == [o // 16 for o in pixel_y_overlaps] and x_overlaps == [o // 16 for o in pixel_x_overlaps]


def tiny_vae() -> AutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(**TINY_VAE).eval()
    vae.enable_tiling(16, 16, 8, 8)
    return vae


def vae_inputs(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A keyframe ``(1, 3, 1, H, W)`` and a 22-frame video (two temporal clips), the same on every rank."""
    generator = torch.Generator().manual_seed(height * width)
    frame = torch.randn(1, 3, 1, height, width, generator=generator)
    video = torch.randn(1, 3, 22, height, width, generator=generator)
    return frame, video


@torch.no_grad()
def serial_and_parallel_vae(vae: AutoencoderKLMiniMaxH3, height: int, width: int) -> list[tuple]:
    """``(name, serial output, parallel output)`` of a keyframe encode, a video encode and its decode."""
    frame, video = vae_inputs(height, width)
    runs = {
        "encode_clip": lambda: vae._encode_clip(frame),
        "encode": lambda: vae._encode(video),
        "decode": lambda: vae.decode(vae._encode(video)[:, :4], return_dict=False)[0],
    }
    serial = {name: run() for name, run in runs.items()}
    enable_parallel_vae(vae)
    return [(name, serial[name], run()) for name, run in runs.items()]


def _parallel_vae_worker(rank: int, world_size: int, store: str) -> None:
    # Each rank would otherwise start one thread per host core; under a container CPU quota that stalls for minutes.
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world_size)
    try:
        for height, width in VAE_FRAMES:
            for name, serial, parallel in serial_and_parallel_vae(tiny_vae(), height, width):
                assert torch.equal(serial, parallel), f"{name} at {height}x{width} on rank {rank}"
    finally:
        dist.destroy_process_group()


def test_parallel_vae_is_bitwise_serial_on_two_ranks(tmp_path):
    mp.spawn(_parallel_vae_worker, args=(2, str(tmp_path / "store")), nprocs=2)


def test_parallel_encode_clip_fires_the_offload_hook(one_process_group):
    class Count(ModelHook):
        calls = 0

        def pre_forward(self, module, *args, **kwargs):
            Count.calls += 1
            return args, kwargs

    vae = tiny_vae()
    add_hook_to_module(vae, Count())
    enable_parallel_vae(vae)
    enable_parallel_vae(vae)  # idempotent
    with torch.no_grad():
        vae._encode_clip(vae_inputs(40, 56)[0])
    assert Count.calls == 1


def test_round_robin_restores_row_major_order():
    for num_tiles in (1, 3, 28, 29):
        for world in (1, 2, 4, 8):
            local = [list(range(num_tiles))[rank::world] for rank in range(world)]
            restored = [local[r][s] for r, s in (tile_source(i, world) for i in range(num_tiles))]
            assert restored == list(range(num_tiles))


def test_state_changes_carry_new_and_replaced_values_and_their_kwargs_types():
    state, before = encoded_state()
    changes = state_changes(before, state)
    assert set(changes["values"]) == {"prompt_embeds", "prompt_embeds_mask", "num_frames"}

    other = PipelineState()
    other.set("prompt", "a fox")
    other.set("num_frames", 124)
    apply_state_changes(other, changes)
    assert other.values.keys() == state.values.keys() and other.get("num_frames") == 125
    assert torch.equal(other.get("prompt_embeds"), state.get("prompt_embeds"))
    assert other.get_by_kwargs("denoiser_input_fields").keys() == {"prompt_embeds", "prompt_embeds_mask"}
    apply_state_changes(other, changes)  # idempotent
    assert other.kwargs_mapping["denoiser_input_fields"].count("prompt_embeds") == 1


def test_broadcast_round_trip_keeps_structure_and_cpu_tensors(one_process_group):
    state, before = encoded_state()
    changes = broadcast_from_rank0(state_changes(before, state), torch.device("cpu"))
    assert torch.equal(changes["values"]["prompt_embeds"], state.get("prompt_embeds"))
    assert changes["values"]["num_frames"] == 125


def test_rank0_text_encoder_block_writes_the_state_and_touches_one_instance(one_process_group):
    class Block:
        def __call__(self, components, state):
            state.set("prompt_embeds", torch.ones(1, 3, 4), kwargs_type="denoiser_input_fields")
            return components, state

    wrapped, plain = Block(), Block()

    class Blocks:
        sub_blocks = {"text_encoder": wrapped}

    text_encoder_on_rank0(Blocks(), torch.device("cpu"))
    assert type(wrapped) is not type(plain)
    state = PipelineState()
    _, state = wrapped("components", state)
    assert torch.equal(state.get("prompt_embeds"), torch.ones(1, 3, 4))
