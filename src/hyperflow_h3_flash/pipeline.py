# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""One call from Hub ids to a ready pipeline: ``pipe = load_pipeline("t2va")``.

The pipeline is the official MiniMax-H3 Modular Pipeline. Its text encoder, VAEs, tokenizer and processors come from
the MiniMax-H3 repository; only the transformer comes from the HyperFlow H3 Flash repository, and two blocks are
swapped (see `blocks.py`). The 62 GB MiniMax-H3 transformer is never downloaded.

Called in every rank of a torchrun job with an initialized process group of several GPUs, it builds the parallel
pipeline: Ulysses context parallelism in the transformer, the text encoder sharded with FSDP (or held on rank 0 when
offloading) and the video VAE's encode and decode tiles split over the ranks (see `distributed.py`).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
from pathlib import Path

import torch
from diffusers.modular_pipelines import ModularPipeline
from huggingface_hub.constants import HF_HUB_OFFLINE
from torch import nn

from .blocks import flash_blocks
from .distributed import apply_parallel, holds_text_encoder, parallel_rank, parallel_world_size
from .int8 import INT8_GEMMS, ROTATE_QUANTS, resolve_int8_gemm, set_kernel_options
from .offload import OFFLOAD_MODES, apply_offload, model_components
from .transformer import (
    FF_CHUNK_TOKENS,
    RANK_MULTIPLE,
    HyperFlowH3FlashTransformer3DModel,
    set_block_chunk,
    set_ff_chunk,
    set_lowrank_rank_multiple,
)

BASE_MODEL = "MiniMaxAI/MiniMax-H3"
WEIGHTS = "VideoRebirthAI/hyperflow-h3-flash"
#: The partition (subfolder) of `WEIGHTS` each workflow reads, and the pipeline component it fills.
PARTITIONS = {"t2va": "fl2va_int8_convrot", "fl2va": "fl2va_int8_convrot", "ref2va": "ref2va_int8_convrot"}
TRANSFORMER_COMPONENTS = {"t2va": "transformer", "fl2va": "transformer", "ref2va": "transformer_ref"}

#: ``offload="auto"`` picks the first mode whose GPU-memory floor (GiB) the device meets.
#: Single GPU: 87 GiB with ``"none"``, 49 GiB with ``"model"``. Multi-GPU (FSDP text encoder + split VAE): 59-70 GiB.
AUTO_OFFLOAD = (("none", 90), ("model", 60), ("stream", 0))
AUTO_OFFLOAD_MULTI_GPU = (("none", 65), ("model", 45), ("stream", 0))
#: Short names accepted by ``attention_backend``; anything else is passed to diffusers' `set_attention_backend` as is.
ATTENTION_BACKENDS = {"sdpa": "native", "fa3": "_flash_3_hub"}

# MiniMax-H3 geometry (diffusers `modular_pipelines/minimax_h3/`).
FPS = 24
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS = 768, 768 * 1344
AUDIO_LATENTS_PER_SECOND, AUDIO_CHANNELS = 40, 2
#: Frame counts are rounded up to ``17n + 5``. 124 frames is 5.2 s; a 15 s request (360 frames) rounds up to 362,
#: 15.08 s, which `admit_max_frames` admits past diffusers' 15.0 s cap.
MIN_FRAMES, MAX_FRAMES = 124, 362
#: The official FP8 export of MiniMax-H3's text encoder, Qwen3-VL-32B-Instruct, for ``text_encoder_precision="fp8"``.
TEXT_ENCODER_FP8_WEIGHTS = "Qwen/Qwen3-VL-32B-Instruct-FP8"
TEXT_ENCODER_PRECISIONS = ("bf16", "fp8")
FP8_GEMMS = ("triton", "deepgemm")

log = logging.getLogger(__name__)


def align_num_frames(num_frames: int) -> int:
    """Round up to the next ``17n + 5``, as diffusers does."""
    return num_frames + (LATENTS_PER_CHUNK - num_frames) % FRAMES_PER_CHUNK


def resolve_canvas(aspect_width: float = 16, aspect_height: float = 9) -> tuple[int, int]:
    """``(height, width)`` diffusers generates when neither is given: 768 short edge at this aspect, 768x1344 max."""
    ratio = aspect_width / aspect_height
    width, height = (
        (CANVAS_SHORT_EDGE * ratio, CANVAS_SHORT_EDGE)
        if ratio >= 1
        else (CANVAS_SHORT_EDGE, CANVAS_SHORT_EDGE / ratio)
    )
    if width * height > CANVAS_MAX_PIXELS:
        scale = (CANVAS_MAX_PIXELS / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def request_tokens(height: int, width: int, num_frames: int) -> int:
    """Generated rows of the packed sequence, video plus audio latents; the prompt adds a few hundred."""
    frames = align_num_frames(num_frames)
    latent_frames = (frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    video = latent_frames * (height // CANVAS_MULTIPLE) * (width // CANVAS_MULTIPLE)
    audio = round(frames / FPS * AUDIO_LATENTS_PER_SECOND) * AUDIO_CHANNELS
    return video + audio


def load_text_encoder(
    path: str,
    *,
    subfolder: str | None,
    layers: int,
    device: str | torch.device,
    local_files_only: bool,
) -> nn.Module:
    """Qwen3-VL with its first ``layers`` decoder layers only (the rest are never read from disk), in bf16 or as the
    checkpoint's FP8, on ``device``."""
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from transformers.utils import logging as hf_logging

    kwargs = {"local_files_only": local_files_only, **({"subfolder": subfolder} if subfolder else {})}
    config = AutoConfig.from_pretrained(path, **kwargs)
    text_config = config.text_config
    if layers < text_config.num_hidden_layers:
        text_config.num_hidden_layers = layers
        if getattr(text_config, "layer_types", None):
            text_config.layer_types = text_config.layer_types[:layers]
    # The checkpoint's layers past `layers` would each be listed as an unexpected key.
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, config=config, dtype=torch.bfloat16, device_map={"": str(device)}, **kwargs
        )
    finally:
        hf_logging.set_verbosity(verbosity)
    return model.eval()


def trim_text_encoder(text_encoder: nn.Module, layer: int) -> None:
    """Drop the parts of the Qwen3-VL conditioner that cannot reach ``hidden_states[layer]``, the only state read.

    Decoder layers after ``layer + 1`` and the language-model head never run on the path the pipeline uses
    (`text_encoder.model`). One extra layer stays so that ``hidden_states[layer]`` is a pre-norm layer output, as
    in the full stack, and not the final normed state. The conditioning is unchanged; host and GPU memory drop by
    about 14 GB.
    """
    language_model = text_encoder.model.language_model
    keep = layer + 1
    if len(language_model.layers) > keep:
        language_model.layers = language_model.layers[:keep]
        text_config = text_encoder.config.text_config
        text_config.num_hidden_layers = keep
        if getattr(text_config, "layer_types", None):
            text_config.layer_types = text_config.layer_types[:keep]
    if isinstance(getattr(text_encoder, "lm_head", None), nn.Linear):
        text_encoder.lm_head = nn.Identity()


def resolve_offload(offload: str, device: torch.device, world_size: int | None = None) -> str:
    """``offload`` with ``"auto"`` resolved from the VRAM of ``device`` and the GPU count (default: the process
    group's)."""
    if offload not in ("auto", *OFFLOAD_MODES):
        raise ValueError(f"unknown offload mode {offload!r}; use auto or one of {OFFLOAD_MODES}")
    if offload != "auto":
        return offload
    if device.type != "cuda":
        return "none"
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    world_size = parallel_world_size() if world_size is None else world_size
    table = AUTO_OFFLOAD_MULTI_GPU if world_size > 1 else AUTO_OFFLOAD
    return next(mode for mode, floor in table if total >= floor)


def resolve_attention(backend: str | None, device: torch.device) -> str | None:
    """Resolve ``"auto"``, ``"sdpa"``, ``"fa3"`` or a diffusers backend name for ``device``."""
    if backend is None:
        return None
    if backend == "auto":
        is_hopper = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9
        return "_flash_3_hub" if is_hopper and importlib.util.find_spec("kernels") is not None else None
    if backend in ("fa3", "_flash_3_hub") and importlib.util.find_spec("kernels") is None:
        log.warning("attention %s needs `pip install kernels`; falling back to sdpa", backend)
        return None
    return ATTENTION_BACKENDS.get(backend, backend)


def resolve_subfolder(weights: str, workflow: str, subfolder: str | None = None) -> str | None:
    """The subfolder of ``weights`` to read: ``subfolder`` if given (``""`` for the root), the root of a directory
    that holds a ``config.json`` itself, and the workflow's partition (`PARTITIONS`) otherwise."""
    if subfolder is not None:
        return subfolder or None
    if (Path(weights) / "config.json").is_file():
        return None
    return PARTITIONS[workflow]


def fp8_kernels_problem() -> str | None:
    """Why transformers cannot run the FP8 text encoder's GEMM (a missing or incompatible `kernels`), or None."""
    try:
        from transformers.utils.import_utils import is_kernels_available
    except ImportError:  # a transformers without the version-aware check
        is_kernels_available = None
    if is_kernels_available is not None:
        ok = is_kernels_available()
    else:
        ok = importlib.util.find_spec("kernels") is not None
    if ok:
        return None
    try:
        installed = f"kernels {importlib.metadata.version('kernels')} does not match this transformers"
    except importlib.metadata.PackageNotFoundError:
        installed = "kernels is not installed"
    return (
        f"the fp8 text encoder runs transformers' FP8 GEMM from the Hub, and {installed}: "
        'pip install "hyperflow-h3-flash[fp8]" (or "transformers[kernels]")'
    )


def admit_max_frames(pipe: ModularPipeline) -> None:
    """Raise the pipeline's duration cap to `MAX_FRAMES`: a 15 s request rounds up to 362 frames, 15.08 s."""
    cls = type(pipe)
    pipe.__class__ = type(cls.__name__, (cls,), {"max_duration": property(lambda _self: MAX_FRAMES / FPS)})


def load_pipeline(
    workflow: str = "t2va",
    *,
    weights: str = WEIGHTS,
    subfolder: str | None = None,
    base_model: str = BASE_MODEL,
    device: str | torch.device = "cuda",
    offload: str = "auto",
    attention_backend: str | None = "auto",
    local_files_only: bool | None = None,
    text_encoder_fsdp: bool = True,
    vae_parallel: bool = True,
    text_encoder_precision: str = "bf16",
    text_encoder_fp8_weights: str = TEXT_ENCODER_FP8_WEIGHTS,
    fp8_gemm: str = "triton",
    int8_gemm: str = "auto",
    rotate_quant: str = "factored",
    lowrank_rank_multiple: int = 8,
    ff_chunk: int = FF_CHUNK_TOKENS,
    block_chunk: int = 0,
) -> ModularPipeline:
    """Build the HyperFlow H3 Flash pipeline for ``workflow`` (``"t2va"``, ``"fl2va"``, or ``"ref2va"``).

    Args:
        weights: HyperFlow H3 Flash transformer Hub ID or local directory.
        subfolder: Partition subfolder within ``weights`` (defaults to ``fl2va_int8_convrot`` for ``t2va``/``fl2va``
            and ``ref2va_int8_convrot`` for ``ref2va``, or ``None`` if ``weights`` directly contains ``config.json``).
        base_model: MiniMax-H3 Hub ID or local snapshot for the text encoder, VAEs, tokenizer, and processors.
        device: Target CUDA device.
        offload: ``"none"``, ``"model"``, ``"stream"`` (see `hyperflow_h3_flash.offload`), or ``"auto"`` (selects
            ``"none"`` at ≥90 GiB single-GPU / ≥65 GiB multi-GPU, ``"model"`` at ≥60 GiB / ≥45 GiB, else ``"stream"``).
        attention_backend: ``"auto"`` (FlashAttention-3 on Hopper when ``kernels`` is installed, else PyTorch SDPA),
            ``"sdpa"``, ``"fa3"``, or any diffusers attention backend name.
        local_files_only: Whether to only load cached local files; defaults to ``HF_HUB_OFFLINE``.
        text_encoder_fsdp: On multi-GPU with ``offload="none"`` and ``text_encoder_precision="bf16"``, shard the text
            encoder across ranks via FSDP2; otherwise run the text encoder on rank 0 and broadcast embeddings.
        vae_parallel: On multi-GPU, distribute video VAE spatial encode and decode tiles across ranks (bitwise
            identical latents and frames).
        text_encoder_precision: ``"bf16"`` (MiniMax-H3 default) or ``"fp8"`` (loads ``text_encoder_fp8_weights``,
            saving ~22 GiB host RAM; requires ``kernels``).
        fp8_gemm: FP8 text encoder GEMM backend, ``"triton"`` or ``"deepgemm"`` (Hopper and newer).
        int8_gemm, rotate_quant: INT8 kernel options (see `hyperflow_h3_flash.int8`).
        lowrank_rank_multiple: Re-pad low-rank linears to this rank multiple at load (default: 8).
        ff_chunk: Feed-forward token slice (0: the whole sequence).
        block_chunk: Run every token-wise op of a block in slices of this many tokens (0: off).

    Returns:
        A ready `ModularPipeline` callable as ``pipe(prompt=..., num_frames=124, num_inference_steps=6,
        generator=..., output=["videos", "audio", "sampling_rate"])``.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is not None:
        # NCCL collectives, FSDP2 and the Triton kernels run on the current device, not on `device`.
        torch.cuda.set_device(device)
    world_size = parallel_world_size()
    offload = resolve_offload(offload, device)
    if text_encoder_precision not in TEXT_ENCODER_PRECISIONS:
        raise ValueError(f"text_encoder_precision must be one of {TEXT_ENCODER_PRECISIONS}")
    if fp8_gemm not in FP8_GEMMS:
        raise ValueError(f"fp8_gemm must be one of {FP8_GEMMS}")
    fp8 = text_encoder_precision == "fp8"
    if fp8:
        if problem := fp8_kernels_problem():
            raise ImportError(problem)
        text_encoder_fsdp = False
        # transformers reads this on every FP8 linear call.
        os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "0" if fp8_gemm == "deepgemm" else "1"
    set_kernel_options(int8_gemm=int8_gemm, rotate_quant=rotate_quant)
    resolved_attention = resolve_attention(attention_backend, device)
    local_files_only = HF_HUB_OFFLINE if local_files_only is None else local_files_only
    load_kwargs = {"local_files_only": local_files_only}
    keep_text_encoder = holds_text_encoder(offload, text_encoder_fsdp)

    pipe = flash_blocks(workflow).init_pipeline(base_model)
    admit_max_frames(pipe)
    component = TRANSFORMER_COMPONENTS[workflow]
    skip = {component} if keep_text_encoder and not fp8 else {component, "text_encoder"}
    names = [name for name in pipe.pretrained_component_names if name not in skip]
    component_kwargs = dict(load_kwargs, dtype=torch.bfloat16)
    if Path(base_model).is_dir():
        # The snapshot's modular_model_index.json names the Hub repo for every component; read the directory instead.
        component_kwargs["pretrained_model_name_or_path"] = base_model
    pipe.load_components(names=names, **component_kwargs)
    missing = [name for name in names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"MiniMax-H3 components did not load: {missing}; see the diffusers warnings above")

    transformer = HyperFlowH3FlashTransformer3DModel.from_pretrained(
        weights,
        subfolder=resolve_subfolder(weights, workflow, subfolder),
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    )
    # Before offloading and context parallelism, which hook the parameters and the blocks.
    repadded = set_lowrank_rank_multiple(transformer, lowrank_rank_multiple)
    set_ff_chunk(transformer, ff_chunk)
    set_block_chunk(transformer, block_chunk)
    pipe.update_components(**{component: transformer})
    if keep_text_encoder and fp8:
        # `trim_text_encoder` keeps `text_encoder_layer + 1` layers; load only those.
        text_encoder = load_text_encoder(
            text_encoder_fp8_weights,
            subfolder=None,
            layers=pipe.text_encoder_layer + 1,
            device="cpu",
            local_files_only=local_files_only,
        )
        trim_text_encoder(text_encoder, pipe.text_encoder_layer)
        pipe.update_components(text_encoder=text_encoder)
    elif keep_text_encoder:
        trim_text_encoder(pipe.text_encoder, pipe.text_encoder_layer)

    if resolved_attention is not None:
        # A Hub kernel is fetched and checked into one cache shared by the ranks: one rank at a time.
        for turn in range(world_size):
            if turn == parallel_rank():
                transformer.set_attention_backend(resolved_attention)
            if world_size > 1:
                torch.distributed.barrier()
        # set_attention_backend also moves the global default; the fp32 audio VAE needs SDPA.
        pipe.audio_vae.set_attention_backend("native")

    components = model_components(pipe)
    apply_parallel(
        pipe,
        components,
        device,
        offload,
        text_encoder_fsdp=text_encoder_fsdp,
        vae_parallel=vae_parallel,
    )
    apply_offload(components, device, offload)
    log.info(
        "HyperFlow H3 Flash %s on %s, offload=%s, attention=%s, text encoder %s; INT8 GEMM %s, rotate_quant %s, "
        "low-rank multiple %d (%d re-padded), ff_chunk %d, block_chunk %d%s",
        workflow,
        device,
        offload,
        "sdpa" if resolved_attention in (None, "native") else resolved_attention,
        text_encoder_precision,
        resolve_int8_gemm(device) if device.type == "cuda" else "torch",
        rotate_quant,
        lowrank_rank_multiple,
        repadded,
        ff_chunk,
        block_chunk,
        f", {world_size} GPUs" if world_size > 1 else "",
    )
    return pipe


__all__ = [
    "ATTENTION_BACKENDS",
    "AUTO_OFFLOAD",
    "AUTO_OFFLOAD_MULTI_GPU",
    "BASE_MODEL",
    "FF_CHUNK_TOKENS",
    "FP8_GEMMS",
    "FPS",
    "FRAMES_PER_CHUNK",
    "INT8_GEMMS",
    "LATENTS_PER_CHUNK",
    "MAX_FRAMES",
    "MIN_FRAMES",
    "PARTITIONS",
    "RANK_MULTIPLE",
    "ROTATE_QUANTS",
    "TEXT_ENCODER_FP8_WEIGHTS",
    "TEXT_ENCODER_PRECISIONS",
    "TRANSFORMER_COMPONENTS",
    "WEIGHTS",
    "admit_max_frames",
    "align_num_frames",
    "fp8_kernels_problem",
    "load_pipeline",
    "load_text_encoder",
    "request_tokens",
    "resolve_attention",
    "resolve_canvas",
    "resolve_offload",
    "resolve_subfolder",
    "trim_text_encoder",
]
