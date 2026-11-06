# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Fit the pipeline into one GPU.

Three placements, chosen with ``load_pipeline(offload=...)``:

* ``"none"``: every component stays on the GPU. Fastest.
* ``"model"``: one model component on the GPU at a time, the others wait in pinned host RAM. Before a component
  runs, the one that ran before it leaves. Peak memory is the largest component (the text encoder) plus its
  activations.
* ``"stream"``: like ``"model"``, but the text encoder and the transformer never move whole. Their weights stream
  from pinned host RAM onto the GPU one layer (text encoder) or one block (transformer) ahead of the compute. Peak
  memory is the video VAE, a few blocks and the activations.

Switching is deterministic: it follows the order the pipeline calls its components and never looks at free memory.
"""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Iterable
from typing import Any

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device
from diffusers.hooks import apply_group_offloading
from torch import nn

OFFLOAD_MODES = ("none", "model", "stream")

#: Components that run on the GPU, in the order the pipeline first calls them (``ref2va`` names its transformer
#: ``transformer_ref``).
MODEL_COMPONENTS = ("text_encoder", "vae", "transformer", "transformer_ref", "audio_vae")
#: Components whose weights stream under ``offload="stream"``, with the granularity they stream at.
STREAMED = {"text_encoder": "leaf_level", "transformer": "block_level", "transformer_ref": "block_level"}
_ALIGN = 256


def _tensors(module: nn.Module) -> Iterable[torch.Tensor]:
    return itertools.chain(module.parameters(), module.buffers())


def _pin(module: nn.Module) -> None:
    """Move the host tensors of ``module`` into one page-locked buffer of exactly their size.

    `Tensor.pin_memory` rounds every allocation up to a power of two, which costs about a quarter more host RAM.
    """
    tensors = [tensor for tensor in _tensors(module) if not tensor.is_pinned()]
    if not tensors:
        return
    sizes = [tensor.numel() * tensor.element_size() for tensor in tensors]
    offsets = list(itertools.accumulate((-(-size // _ALIGN) * _ALIGN for size in sizes[:-1]), initial=0))
    arena = torch.empty(offsets[-1] + sizes[-1], dtype=torch.uint8)
    for tensor, offset, size in zip(tensors, offsets, sizes, strict=True):
        view = arena[offset : offset + size].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor.data)
        tensor.data = view
    cudart = torch.cuda.cudart()
    if error := int(cudart.cudaHostRegister(arena.data_ptr(), arena.numel(), 0)):
        raise RuntimeError(f"cudaHostRegister of {arena.numel() / 2**30:.1f} GiB failed with CUDA error {error}")
    weakref.finalize(module, cudart.cudaHostUnregister, arena.data_ptr()).atexit = False


class _Swap:
    """A component's weights, kept in pinned host RAM, with a GPU copy made on `load` and dropped on `drop`.

    Inference never writes the weights, so `drop` points every tensor back at its host copy instead of copying back.
    """

    def __init__(self, module: nn.Module, device: torch.device) -> None:
        _pin(module)
        self.host = [(tensor, tensor.data) for tensor in _tensors(module)]
        self.device = device
        self.loaded = False

    def load(self) -> None:
        if not self.loaded:
            for tensor, host in self.host:
                tensor.data = host.to(self.device, non_blocking=True)
            self.loaded = True

    def drop(self) -> None:
        if self.loaded:
            for tensor, host in self.host:
                tensor.data = host
            self.loaded = False


class _OneOnDevice(ModelHook):
    """Before its component runs: every other swapped component leaves the GPU, then this one is loaded.

    A streamed component has no `_Swap`; its own group-offload hooks bring its weights in. It still evicts the
    others, so that a VAE used before the transformer does not stay on the GPU while it runs.
    """

    def __init__(self, device: torch.device, swap: _Swap | None, swaps: list[_Swap]) -> None:
        self.execution_device = device
        self.swap = swap
        self.swaps = swaps

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any):
        for other in self.swaps:
            if other is not self.swap:
                other.drop()
        if self.swap is not None:
            self.swap.load()
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


def apply_offload(components: dict[str, nn.Module], device: str | torch.device, mode: str) -> None:
    """Place ``components`` (name -> model) for ``mode``, one of `OFFLOAD_MODES`."""
    if mode not in OFFLOAD_MODES:
        raise ValueError(f"unknown offload mode {mode!r}; use one of {OFFLOAD_MODES}")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if mode == "none":
        for module in components.values():
            module.to(device)
        return

    swaps: list[_Swap] = []
    for name, module in components.items():
        module.to("cpu")
        if mode == "stream" and name in STREAMED:
            _pin(module)
            apply_group_offloading(
                module,
                onload_device=device,
                offload_type=STREAMED[name],
                num_blocks_per_group=1 if STREAMED[name] == "block_level" else None,
                use_stream=True,
                low_cpu_mem_usage=True,
            )
            swap = None
        else:
            swap = _Swap(module, device)
            swaps.append(swap)
        add_hook_to_module(module, _OneOnDevice(device, swap, swaps))


def model_components(pipe: Any, names: Iterable[str] = MODEL_COMPONENTS) -> dict[str, nn.Module]:
    """The loaded ``names`` of ``pipe`` that are `nn.Module`s, in pipeline order."""
    return {name: module for name in names if isinstance(module := getattr(pipe, name, None), nn.Module)}


__all__ = ["MODEL_COMPONENTS", "OFFLOAD_MODES", "apply_offload", "model_components"]
