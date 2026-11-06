# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Fit the pipeline across several GPUs of one machine.

Every rank runs the whole pipeline on the same inputs and ends with the same frames (rank 0 writes them):

* **Transformer (`enable_ulysses`)**: Ulysses context parallelism splits the packed sequence across the ranks.
* **Text encoder (`shard_text_encoder` / `text_encoder_on_rank0`)**: With ``offload="none"``, FSDP2 shards the 50 GiB
  Qwen3-VL conditioner across the ranks; each decoder layer and vision block gathers its weights just before it runs
  and frees them right after. When offloading (e.g. several 24 GB GPUs with ``offload="stream"``), only rank 0 loads
  and offloads the text encoder and broadcasts the encoded state (~2 MB) to the other ranks.
* **Video VAE (`enable_parallel_vae`)**: `AutoencoderKLMiniMaxH3` encodes and decodes each temporal clip as a grid
  of overlapping 256x256-pixel tiles (`_encode_clip`, `_decode_clip`) and blends them (`_stitch_tiles`). Rank ``k``
  runs tiles ``k, k + world, ...``, all-gathers them, and stitches the clip on every rank — bitwise identical to one
  GPU. Encoding covers the keyframes (``fl2va``) and the image and video references (``ref2va``); the stitched
  moments are sampled by the caller as before. Every rank has to encode the same pixels, as every rank runs the
  whole pipeline.
"""

from __future__ import annotations

import os
import socket
import sys
import types
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

#: Ulysses splits the attention heads over the GPUs, so the GPU count has to divide them.
NUM_ATTENTION_HEADS = 56
#: ``--gpus auto`` uses the visible GPUs up to this many.
AUTO_MAX_GPUS = 4


def world() -> tuple[int, int, int]:
    """``(rank, local_rank, world_size)`` from the torchrun environment; ``(0, 0, 1)`` outside it."""
    return int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def parallel_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def parallel_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def holds_text_encoder(offload: str, text_encoder_fsdp: bool = True) -> bool:
    """Whether this rank loads the text encoder (every rank for single-GPU or FSDP; rank 0 otherwise)."""
    if parallel_world_size() <= 1:
        return True
    return (offload == "none" and text_encoder_fsdp) or parallel_rank() == 0


def resolve_num_gpus(requested: int | str, visible_gpus: int, world_size: int = 1) -> int:
    """Resolve ``requested`` (an integer or ``"auto"``) against ``visible_gpus`` and validate the head split."""
    if world_size > 1:
        num_gpus = world_size
    elif requested == "auto":
        num_gpus = max(1, min(AUTO_MAX_GPUS, visible_gpus))
    else:
        num_gpus = int(requested)
        if num_gpus > visible_gpus:
            raise ValueError(f"--gpus {num_gpus} requested, but {visible_gpus} GPU(s) are visible")
    if NUM_ATTENTION_HEADS % num_gpus != 0:
        raise ValueError(
            f"{num_gpus} GPUs: Ulysses splits the {NUM_ATTENTION_HEADS} attention heads; use 1, 2, 4, 7 or 8"
        )
    return num_gpus


def relaunch_under_torchrun(num_gpus: int) -> None:
    """Replace this process by ``torchrun --nproc_per_node num_gpus -m hyperflow_h3_flash <same arguments>``."""
    if "WORLD_SIZE" in os.environ or num_gpus <= 1:
        return
    # A static rendezvous on the loopback: `--standalone` advertises the hostname, which containers may not resolve.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={num_gpus}"]
    cmd += ["--master_addr=127.0.0.1", f"--master_port={port}", "--module", "--", "hyperflow_h3_flash", *sys.argv[1:]]
    os.execv(sys.executable, cmd)


def enable_ulysses(transformer: nn.Module) -> None:
    """Split the transformer's sequence over every rank of the default process group."""
    from diffusers import ContextParallelConfig

    if getattr(type(transformer), "_cp_plan", None) is None:
        raise RuntimeError("this diffusers has no MiniMax-H3 context parallel plan; run on one GPU")
    # `ulysses_anything`: the packed sequence length (it includes the prompt) need not divide by the degree.
    config = ContextParallelConfig(ulysses_degree=dist.get_world_size(), ulysses_anything=True)
    transformer.enable_parallelism(config=config)


def shard_text_encoder(text_encoder: nn.Module, device: torch.device) -> None:
    """Shard the Qwen3-VL conditioner over every rank with FSDP2: each decoder layer and vision block, then root."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    model = text_encoder.model
    for layer in model.language_model.layers:
        fully_shard(layer, mesh=mesh)
    for block in getattr(getattr(model, "visual", None), "blocks", ()):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    model.to(device)


class _DeviceTensor:
    """A CUDA tensor in a broadcast payload: travels as CPU bytes, lands on the receiving rank's own GPU."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.cpu = tensor.cpu()


def _map_tensors(obj: Any, fn: Callable[[Any], Any]) -> Any:
    if torch.is_tensor(obj) or isinstance(obj, _DeviceTensor):
        return fn(obj)
    if isinstance(obj, list):
        return [_map_tensors(x, fn) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_map_tensors(x, fn) for x in obj)
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    return obj


def broadcast_from_rank0(payload: Any, device: torch.device) -> Any:
    """Rank 0's ``payload`` on every rank; CUDA tensors arrive on ``device``, CPU tensors stay on the CPU."""
    box = [_map_tensors(payload, lambda t: _DeviceTensor(t) if torch.is_tensor(t) and t.is_cuda else t)]
    dist.broadcast_object_list(box, src=0)
    return _map_tensors(box[0], lambda t: t.cpu.to(device) if isinstance(t, _DeviceTensor) else t)


def state_changes(before: dict, state: Any) -> dict:
    """What a block wrote into a `PipelineState`: new or replaced values, plus its kwargs-type mapping."""
    values = {k: v for k, v in state.values.items() if k not in before or before[k] is not v}
    return {"values": values, "kwargs_mapping": {k: list(v) for k, v in state.kwargs_mapping.items()}}


def apply_state_changes(state: Any, changes: dict) -> None:
    state.values.update(changes["values"])
    for kwargs_type, keys in changes["kwargs_mapping"].items():
        known = state.kwargs_mapping.setdefault(kwargs_type, [])
        known.extend(k for k in keys if k not in known)


def text_encoder_on_rank0(blocks: Any, device: torch.device) -> None:
    """Only rank 0 runs the text-encoder block of ``blocks``; the other ranks receive what it wrote into the state."""
    block = blocks.sub_blocks["text_encoder"]
    cls, rank = type(block), dist.get_rank()

    def call(self, components, state):
        if rank != 0:
            apply_state_changes(state, broadcast_from_rank0(None, device))
            return components, state
        before = dict(state.values)
        result = cls.__call__(self, components, state)
        broadcast_from_rank0(state_changes(before, state), device)
        return result

    block.__class__ = type(cls.__name__, (cls,), {"__call__": call})


def _pixel_tiles(
    vae: nn.Module, height: int, width: int, scale: int
) -> tuple[list[tuple[slice, slice]], int, list, list]:
    """`_split_tiles` over a ``height x width``-pixel frame: tile slices divided by ``scale``, row-major."""
    y_starts, y_lengths, y_overlaps = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    x_starts, x_lengths, x_overlaps = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    tiles = [
        (slice(y // scale, (y + h) // scale), slice(x // scale, (x + w) // scale))
        for y, h in zip(y_starts, y_lengths, strict=True)
        for x, w in zip(x_starts, x_lengths, strict=True)
    ]
    return tiles, len(x_starts), y_overlaps, x_overlaps


def tile_plan(
    vae: nn.Module, latent_height: int, latent_width: int
) -> tuple[list[tuple[slice, slice]], int, list, list]:
    """Decode: ``(latent slices of every tile in row-major order, columns, pixel overlaps in y, in x)``."""
    ratio = vae.spatial_compression_ratio
    return _pixel_tiles(vae, latent_height * ratio, latent_width * ratio, ratio)


def encode_tile_plan(vae: nn.Module, height: int, width: int) -> tuple[list[tuple[slice, slice]], int, list, list]:
    """Encode: ``(pixel slices of every tile in row-major order, columns, latent overlaps in y, in x)``."""
    ratio = vae.spatial_compression_ratio
    tiles, columns, y_overlaps, x_overlaps = _pixel_tiles(vae, height, width, 1)
    return tiles, columns, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps]


def tile_source(index: int, world_size: int) -> tuple[int, int]:
    """``(rank, slot on that rank)`` of tile ``index`` under the round-robin split ``tiles[rank::world_size]``."""
    return index % world_size, index // world_size


def _gather_tiles(
    local: list[torch.Tensor], num_tiles: int, world_size: int, device: torch.device
) -> list[torch.Tensor]:
    """All-gather every rank's round-robin tiles back into row-major order; all tiles have one shape."""
    per_rank = -(-num_tiles // world_size)
    if num_tiles >= world_size:
        shape, dtype = tuple(local[0].shape), local[0].dtype
    else:  # fewer tiles than ranks: some ranks decoded nothing, rank 0 always has a tile
        box = [(tuple(local[0].shape), local[0].dtype) if local else None]
        dist.broadcast_object_list(box, src=0)
        shape, dtype = box[0]
    stack = torch.zeros((per_rank, *shape), device=device, dtype=dtype)
    for slot, tile in enumerate(local):
        stack[slot].copy_(tile)
    # Gloo only takes the concatenated output layout; NCCL takes either.
    gathered = torch.empty((world_size * per_rank, *shape), device=device, dtype=dtype)
    dist.all_gather_into_tensor(gathered, stack)
    gathered = gathered.view(world_size, per_rank, *shape)
    return [gathered[tile_source(index, world_size)] for index in range(num_tiles)]


def _parallel_decode_clip(self: Any, z: torch.Tensor) -> torch.Tensor:
    if not self.use_tiling:
        return self._serial_decode_clip(z)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    tiles, columns, y_overlaps, x_overlaps = tile_plan(self, z.shape[-2], z.shape[-1])
    local = [self.decoder(self.post_quant_conv(z[..., ys, xs])) for ys, xs in tiles[rank::world_size]]
    decoded = _gather_tiles(local, len(tiles), world_size, z.device)
    del local
    rows = [decoded[i : i + columns] for i in range(0, len(decoded), columns)]
    return self._stitch_tiles(rows, y_overlaps, x_overlaps)


def _parallel_encode_clip(self: Any, x: torch.Tensor) -> torch.Tensor:
    if not self.use_tiling:
        return self._serial_encode_clip(x)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    tiles, columns, y_overlaps, x_overlaps = encode_tile_plan(self, x.shape[-2], x.shape[-1])
    local = [self.quant_conv(self.encoder(x[..., ys, xs])) for ys, xs in tiles[rank::world_size]]
    encoded = _gather_tiles(local, len(tiles), world_size, x.device)
    del local
    rows = [encoded[i : i + columns] for i in range(0, len(encoded), columns)]
    return self._stitch_tiles(rows, y_overlaps, x_overlaps)


def enable_parallel_vae(vae: nn.Module) -> None:
    """Make the video VAE's encode and decode divide their spatial tiles over every rank of the default group."""
    from diffusers.utils.accelerate_utils import apply_forward_hook

    if getattr(vae, "_serial_decode_clip", None) is not None:
        return
    for name in ("_split_tiles", "_stitch_tiles", "_encode_clip", "_decode_clip", "tile_sample_min_height"):
        if not hasattr(vae, name):
            raise RuntimeError(f"this diffusers AutoencoderKLMiniMaxH3 has no `{name}`; the parallel VAE needs it")
    vae._serial_encode_clip = vae._encode_clip
    # The pipeline calls `_encode_clip` directly for keyframes and image references, so it fires the offload hook.
    vae._encode_clip = types.MethodType(apply_forward_hook(_parallel_encode_clip), vae)
    vae._serial_decode_clip = vae._decode_clip
    vae._decode_clip = types.MethodType(_parallel_decode_clip, vae)


def apply_parallel(
    pipe: Any,
    components: dict[str, nn.Module],
    device: torch.device,
    offload: str = "none",
    *,
    text_encoder_fsdp: bool = True,
    vae_parallel: bool = True,
) -> None:
    """Configure ``pipe`` and ``components`` for the initialized process group before `apply_offload` runs."""
    if parallel_world_size() <= 1:
        return
    if offload == "none" and text_encoder_fsdp and "text_encoder" in components:
        shard_text_encoder(components.pop("text_encoder"), device)
    else:
        text_encoder_on_rank0(pipe._blocks, device)
    for name in ("transformer", "transformer_ref"):
        if (transformer := getattr(pipe, name, None)) is not None:
            enable_ulysses(transformer)
    if vae_parallel and getattr(pipe, "vae", None) is not None:
        enable_parallel_vae(pipe.vae)


__all__ = [
    "AUTO_MAX_GPUS",
    "NUM_ATTENTION_HEADS",
    "apply_parallel",
    "apply_state_changes",
    "broadcast_from_rank0",
    "enable_parallel_vae",
    "enable_ulysses",
    "encode_tile_plan",
    "holds_text_encoder",
    "parallel_rank",
    "parallel_world_size",
    "relaunch_under_torchrun",
    "resolve_num_gpus",
    "shard_text_encoder",
    "state_changes",
    "text_encoder_on_rank0",
    "tile_plan",
    "tile_source",
    "world",
]
