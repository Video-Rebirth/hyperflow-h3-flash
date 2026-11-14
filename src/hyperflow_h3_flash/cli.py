# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""``hyperflow-h3-flash``: generate one clip with audio from the command line.

    hyperflow-h3-flash --prompt "..."                                   # text to video+audio (t2va)
    hyperflow-h3-flash --prompt "..." --image first.png [--last-image last.png]   # keyframes to video+audio (fl2va)
    hyperflow-h3-flash --prompt "..." --reference a.png [--reference b.mp4 ...]   # references to video+audio (ref2va)
    hyperflow-h3-flash --prompt "..." --gpus 4                          # context parallel over 4 GPUs (or --gpus auto)

``--gpus N`` (alias ``--num-gpus``) runs the pipeline across N GPUs: the command re-launches itself under ``torchrun``,
the transformer splits its sequence with Ulysses, the text encoder shards with FSDP (or runs on rank 0 when
offloading), the video VAE splits its encode and decode tiles, and rank 0 writes the file.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .distributed import relaunch_under_torchrun, resolve_num_gpus, world
from .hostmem import check_host_memory
from .offload import OFFLOAD_MODES
from .pipeline import (
    BASE_MODEL,
    FP8_GEMMS,
    FPS,
    MAX_FRAMES,
    MIN_FRAMES,
    TEXT_ENCODER_PRECISIONS,
    WEIGHTS,
    align_num_frames,
    fp8_kernels_problem,
    load_pipeline,
    request_tokens,
    resolve_canvas,
    resolve_offload,
)
from .schedule import DEFAULT_STEPS, SIGMA_GRIDS

OUTPUTS = ["videos", "audio", "sampling_rate"]
#: ``--reference`` kinds by file suffix.
REFERENCE_SUFFIXES = {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
    "video": (".mp4", ".mov", ".webm", ".mkv", ".avi"),
    "audio": (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"),
}
#: Per-request ``--reference`` limits, the defaults of diffusers' MiniMax-H3 reference validation.
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}
MAX_REFERENCES = 12

log = logging.getLogger("hyperflow_h3_flash")


def gpus_arg(text: str) -> int | str:
    if text == "auto":
        return text
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer or auto, got {text!r}")
    return value


def non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or a positive integer, got {text!r}")
    return value


def reference_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return next((kind for kind, suffixes in REFERENCE_SUFFIXES.items() if suffix in suffixes), None)


def reference_problem(paths: list[Path]) -> str | None:
    """Why ``paths`` cannot form one ``ref2va`` request, checked before any weights load; ``None`` if they can."""
    for path in paths:
        if reference_kind(path) is None:
            known = ", ".join(suffix for suffixes in REFERENCE_SUFFIXES.values() for suffix in suffixes)
            return f"--reference {path}: unknown media type; use one of {known}"
    counts = Counter(reference_kind(path) for path in paths)
    for kind, limit in REFERENCE_LIMITS.items():
        if counts[kind] > limit:
            return f"{counts[kind]} {kind} references; a request takes at most {limit}"
    if len(paths) > MAX_REFERENCES:
        return f"{len(paths)} references; a request takes at most {MAX_REFERENCES}"
    if counts["audio"] and not (counts["image"] or counts["video"]):
        return "an audio reference needs at least one image or video reference"
    return None


def load_references(paths: list[Path]) -> list[Any]:
    """The ``ref2va`` references, in command-line order; a video brings its own soundtrack."""
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference as AudioReference
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference as ImageReference
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference as VideoReference

    loaders = {"image": ImageReference, "video": VideoReference, "audio": AudioReference}
    return [loaders[reference_kind(path)].from_file(str(path)) for path in paths]


def canvas(args: argparse.Namespace) -> tuple[int, int]:
    """The ``(height, width)`` diffusers will generate; ``ref2va`` keeps 16:9 whatever its references."""
    if args.height is not None:
        return args.height, args.width
    if args.image is not None:
        from PIL import Image

        with Image.open(args.image) as image:
            return resolve_canvas(image.width, image.height)
    return resolve_canvas()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hyperflow-h3-flash", description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompt", required=True, help="text prompt describing scene, motion, and sound")
    parser.add_argument("--image", type=Path, help="first keyframe image (selects the fl2va workflow)")
    parser.add_argument("--last-image", type=Path, help="last keyframe image (requires --image)")
    parser.add_argument(
        "--reference",
        dest="references",
        type=Path,
        action="append",
        default=[],
        help="image, video, or audio reference in prompt order (<Picture i>, <Video i>, <Audio i>); "
        "repeatable, selects the ref2va workflow (up to 9 images, 3 videos, 3 audio clips; 12 max)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=MIN_FRAMES,
        help=f"frame count from {MIN_FRAMES} to {MAX_FRAMES} at {FPS} fps (5.2 to 15 s); rounded up to 17n+5",
    )
    parser.add_argument("--height", type=int, help="video height (multiple of 32; default: 768 on short edge)")
    parser.add_argument("--width", type=int, help="video width (multiple of 32)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        choices=sorted(SIGMA_GRIDS),
        help="sampling steps: 4 (speed), 6 (default, balanced), or 8 (quality)",
    )
    parser.add_argument("--output", type=Path, default=Path("hyperflow_h3_flash.mp4"), help="output MP4 path")

    parser.add_argument(
        "--offload",
        default="auto",
        choices=("auto", *OFFLOAD_MODES),
        help="GPU memory placement: auto (by VRAM), none, model, or stream",
    )
    parser.add_argument(
        "--attention",
        "--attention-backend",
        dest="attention",
        default="auto",
        help="attention backend: auto (fa3 on Hopper when `kernels` is installed, else sdpa), sdpa, fa3, "
        "or any diffusers backend name",
    )
    parser.add_argument(
        "--gpus",
        "--num-gpus",
        dest="gpus",
        type=gpus_arg,
        default=1,
        help="number of GPUs for context and VAE parallelism: 1, 2, 4, 7, 8, or auto (up to 4 visible GPUs)",
    )
    parser.add_argument("--weights", default=WEIGHTS, help="HyperFlow H3 Flash transformer Hub ID or local path")
    parser.add_argument(
        "--subfolder",
        help="subfolder within --weights (default: fl2va_int8_convrot for t2va/fl2va, ref2va_int8_convrot for ref2va; "
        "fl2va_bf16 or ref2va_bf16 for the unquantized weights)",
    )
    parser.add_argument("--base-model", default=BASE_MODEL, help="MiniMax-H3 base model Hub ID or local path")
    parser.add_argument(
        "--text-encoder-precision",
        default="bf16",
        choices=TEXT_ENCODER_PRECISIONS,
        help="bf16 (default) or fp8 (Qwen/Qwen3-VL-32B-Instruct-FP8, ~22 GiB less host RAM; requires `kernels`)",
    )
    parser.add_argument("--fp8-gemm", default="triton", choices=FP8_GEMMS, help="FP8 text encoder GEMM backend")
    parser.add_argument(
        "--block-chunk",
        type=non_negative_int,
        default=0,
        help="slice every token-wise op of a transformer block to bound VRAM activations (default: 0, off; "
        "use 16384 for 15 s mixed ref2va on 24 GB GPUs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)
    if args.last_image is not None and args.image is None:
        parser.error("--last-image needs --image")
    if args.references and args.image is not None:
        parser.error("--reference (ref2va) and --image (fl2va) select different workflows; pass one of them")
    if problem := reference_problem(args.references):
        parser.error(problem)
    if (args.height is None) != (args.width is None):
        parser.error("pass both --height and --width, or neither")
    if args.height is not None and (args.height % 32 != 0 or args.width % 32 != 0):
        parser.error("--height and --width must be multiples of 32")
    aligned = align_num_frames(args.num_frames) if args.num_frames >= 1 else 0
    if not (MIN_FRAMES <= aligned <= MAX_FRAMES):
        parser.error(
            f"--num-frames {args.num_frames} (rounded up to {aligned}) is outside {MIN_FRAMES} to {MAX_FRAMES} "
            "(5.2 to 15 s)"
        )
    return args


def check_platform(num_gpus: int, platform: str | None = None) -> None:
    """Ensure multi-GPU has Linux (NCCL) and the INT8 kernels have Triton before downloading weights."""
    platform = platform or sys.platform
    if num_gpus > 1 and not platform.startswith("linux"):
        raise ValueError(f"--gpus {num_gpus} requires Linux or WSL2 (NCCL); on native {platform}, use --gpus 1")
    if importlib.util.find_spec("triton") is None:
        hint = "pip install triton-windows" if platform == "win32" else "install a CUDA build of PyTorch"
        raise ValueError(f"the INT8 kernels require Triton ({hint})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Read when the CUDA allocator starts; torchrun ranks inherit it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rank, local_rank, world_size = world()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    if not args.verbose:
        log.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    workflow = "ref2va" if args.references else "fl2va" if args.image else "t2va"

    visible = torch.cuda.device_count()
    try:
        num_gpus = resolve_num_gpus(args.gpus, visible_gpus=visible, world_size=world_size)
        check_platform(num_gpus)
        if args.text_encoder_precision == "fp8" and (problem := fp8_kernels_problem()):
            raise ValueError(problem)
    except ValueError as error:
        print(f"hyperflow-h3-flash: {error}", file=sys.stderr)
        return 2
    height, width = canvas(args)
    tokens = request_tokens(height, width, args.num_frames)
    offload = resolve_offload(args.offload, torch.device("cuda", 0), num_gpus)
    if "WORLD_SIZE" not in os.environ:
        check_host_memory(
            weights=args.weights,
            subfolder=args.subfolder,
            base_model=args.base_model,
            text_encoder_precision=args.text_encoder_precision,
            offload=offload,
            num_gpus=num_gpus,
            height=height,
            width=width,
            num_frames=args.num_frames,
            workflow=workflow,
        )
    log.info(
        "%s on %d GPU(s), offload %s, attention %s; %dx%d x %d frames = %d tokens",
        workflow,
        num_gpus,
        offload,
        args.attention,
        width,
        height,
        align_num_frames(args.num_frames),
        tokens,
    )
    relaunch_under_torchrun(num_gpus)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        torch.distributed.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(minutes=5))

    started = time.perf_counter()
    pipe = load_pipeline(
        workflow,
        weights=args.weights,
        subfolder=args.subfolder,
        base_model=args.base_model,
        device=device,
        offload=args.offload,
        attention_backend=args.attention,
        text_encoder_precision=args.text_encoder_precision,
        fp8_gemm=args.fp8_gemm,
        block_chunk=args.block_chunk,
    )
    log.info("loaded in %.0f s", time.perf_counter() - started)

    inputs = dict(
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.height is not None:
        inputs.update(height=args.height, width=args.width)
    if args.image is not None:
        from diffusers.utils import load_image

        inputs["image"] = load_image(str(args.image))
        if args.last_image is not None:
            inputs["last_image"] = load_image(str(args.last_image))
    if args.references:
        inputs["references"] = load_references(args.references)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = pipe(**inputs, output=OUTPUTS)
    elapsed = time.perf_counter() - started
    log.info("generated in %.1f s, peak GPU memory %.1f GiB", elapsed, torch.cuda.max_memory_allocated(device) / 2**30)

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank == 0:
        from diffusers.utils.export_utils import encode_video

        args.output.parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            result["videos"][0],
            fps=FPS,
            output_path=str(args.output),
            audio=result["audio"][0],
            audio_sample_rate=result["sampling_rate"],
        )
        log.info("wrote %s", args.output)
    return 0
