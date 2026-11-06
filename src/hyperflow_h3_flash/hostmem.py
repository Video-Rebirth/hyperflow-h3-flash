# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Before loading (and before the first download): will the run fit in the host RAM this machine has available?

``offload="model"`` and ``"stream"`` keep every weight in pinned host RAM for the whole run; with several GPUs (text
encoder on rank 0) that is rank 0's text encoder and every rank's transformer and VAEs. With ``"none"`` the weights
pass through memory-mapped files on their way to the GPU and do not stay. Each process also needs its CUDA context,
libraries and the decoded frames (every rank decodes them). The estimate reads local safetensors sizes when complete,
or falls back to the known checkpoint sizes so it also runs before the first download.

Calibration, H200, 74.7 GiB of weights after trimming, 362 frames: ``"none"`` peaks at 16 GiB per process at 768p
(one GPU and four) and 27 GiB at 1088p; ``"stream"`` at 90.6 GiB at 768p, ``"model"`` at 102 GiB at 1088p.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .pipeline import BASE_MODEL, TEXT_ENCODER_FP8_WEIGHTS, WEIGHTS, align_num_frames

log = logging.getLogger("hyperflow_h3_flash")

GIB = 2**30
#: Qwen3-VL decoder layers on disk and kept by `trim_text_encoder` (MiniMax-H3 reads ``hidden_states[50]``).
TEXT_ENCODER_LAYERS, TEXT_ENCODER_KEPT = 64, 51
#: Per process, on top of the pinned weights: CUDA context, libraries and load-time copies...
PROCESS_OVERHEAD_GIB = 5.25
#: ...plus this many times the decoded frames in float32 (the pipeline output and the copies made to write it).
FRAME_COPIES = 2.6
#: Known component sizes on disk (text encoder after trimming to 51 layers), used before weights are downloaded.
DEFAULT_SIZES = {
    "text_encoder_bf16": int(49.5 * GIB),
    "text_encoder_fp8": int(27.5 * GIB),
    "transformer": int(14.9 * GIB),
    "vae": int(9.7 * GIB),
    "audio_vae": int(0.56 * GIB),
}
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC = Path("/proc")
#: ``(limit, usage, page cache, shared memory)`` files and ``memory.stat`` keys of cgroup v2 and v1.
CGROUP_FILES = {
    "v2": ("memory.max", "memory.current", "file", "shmem"),
    "v1": ("memory.limit_in_bytes", "memory.usage_in_bytes", "total_cache", "total_shmem"),
}


def _local_dir(repo: str, subfolder: str | None = None) -> Path | None:
    path = Path(repo).expanduser()
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo, local_files_only=True, allow_patterns=["*.json"]))
        except Exception:  # not in the cache (yet): nothing to measure locally
            return None
    return path / subfolder if subfolder else path


def _safetensors_bytes(folder: Path | None) -> int | None:
    """Bytes of the safetensors in ``folder``; None when a shard its index lists is missing (a partial download)."""
    if folder is None or not folder.is_dir():
        return None
    files = list(folder.glob("*.safetensors"))
    for index in folder.glob("*.safetensors.index.json"):
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return None
        if not shards <= {file.name for file in files}:
            return None
    return sum(file.stat().st_size for file in files) or None


def weight_bytes(
    *,
    weights: str = WEIGHTS,
    subfolder: str | None = None,
    base_model: str = BASE_MODEL,
    text_encoder_precision: str = "bf16",
    text_encoder_fp8_weights: str = TEXT_ENCODER_FP8_WEIGHTS,
    workflow: str = "t2va",
) -> dict[str, int] | None:
    """On-disk bytes of every component, the text encoder after trimming; None unless all are local."""
    from .pipeline import resolve_subfolder

    text_encoder = (
        _local_dir(text_encoder_fp8_weights)
        if text_encoder_precision == "fp8"
        else _local_dir(base_model, "text_encoder")
    )
    sizes = {
        "text_encoder": _safetensors_bytes(text_encoder),
        "transformer": _safetensors_bytes(_local_dir(weights, resolve_subfolder(weights, workflow, subfolder))),
        "vae": _safetensors_bytes(_local_dir(base_model, "vae")),
        "audio_vae": _safetensors_bytes(_local_dir(base_model, "audio_vae")),
    }
    if any(size is None for size in sizes.values()):
        return None
    sizes["text_encoder"] = sizes["text_encoder"] * TEXT_ENCODER_KEPT // TEXT_ENCODER_LAYERS
    return sizes


def default_weight_bytes(text_encoder_precision: str = "bf16") -> dict[str, int]:
    """Known component sizes of the official checkpoints when local files are not yet cached."""
    te_key = "text_encoder_fp8" if text_encoder_precision == "fp8" else "text_encoder_bf16"
    return {
        "text_encoder": DEFAULT_SIZES[te_key],
        "transformer": DEFAULT_SIZES["transformer"],
        "vae": DEFAULT_SIZES["vae"],
        "audio_vae": DEFAULT_SIZES["audio_vae"],
    }


def estimate_gib(
    sizes: dict[str, int],
    *,
    offload: str,
    num_gpus: int,
    height: int,
    width: int,
    num_frames: int,
) -> float:
    """Peak host RAM of all the processes of the run, in GiB, for a resolved placement and request."""
    vaes = sizes["vae"] + sizes["audio_vae"]
    if offload not in ("model", "stream"):
        held = 0
    elif num_gpus == 1:
        held = sum(sizes.values())
    else:
        held = sizes["text_encoder"] + num_gpus * (sizes["transformer"] + vaes)
    frames = align_num_frames(num_frames) * height * width * 3 * 4 / GIB
    return held / GIB + num_gpus * (PROCESS_OVERHEAD_GIB + FRAME_COPIES * frames)


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _stat_values(path: Path) -> dict[str, int]:
    try:
        pairs = (line.split() for line in path.read_text().splitlines())
        return {pair[0]: int(pair[1]) for pair in pairs if len(pair) == 2 and pair[1].isdigit()}
    except OSError:
        return {}


def _cgroup_dirs(proc: Path, root: Path) -> list[tuple[str, Path]]:
    """This process's memory cgroup and its ancestors, innermost first: limits set on any of them apply."""
    try:
        lines = (proc / "self" / "cgroup").read_text().splitlines()
    except OSError:
        lines = []
    dirs = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative = parts
        if hierarchy == "0" and not controllers:
            version, base = "v2", root
        elif "memory" in controllers.split(","):
            version, base = "v1", root / "memory"
        else:
            continue
        path = base / relative.strip().lstrip("/")
        dirs += [(version, directory) for directory in (path, *path.parents) if directory.is_relative_to(base)]
    return dirs or [("v2", root), ("v1", root / "memory")]


def _cgroup_available(proc: Path, root: Path) -> int | None:
    """Bytes left under the tightest cgroup memory limit on this process, page cache (not shmem) counted as free."""
    left = None
    for version, directory in _cgroup_dirs(proc, root):
        limit, usage, cache, shmem = CGROUP_FILES[version]
        cap, used = _read_int(directory / limit), _read_int(directory / usage)
        if cap is None or used is None or cap >= 2**60:  # "max", missing, or v1's unlimited sentinel
            continue
        stats = _stat_values(directory / "memory.stat")
        reclaimable = max(stats.get(cache, 0) - stats.get(shmem, 0), 0)
        room = max(cap - (used - reclaimable), 0)
        left = room if left is None else min(left, room)
    return left


def available_gib(proc: Path = PROC, cgroup_root: Path = CGROUP_ROOT) -> float | None:
    """``MemAvailable``, capped by the cgroup limits on this process; None without ``/proc/meminfo``."""
    available = None
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    if available is None:
        return None
    cgroup = _cgroup_available(proc, cgroup_root)
    return min(available, cgroup if cgroup is not None else available) / GIB


def in_wsl(proc: Path = PROC) -> bool:
    try:
        return "microsoft" in (proc / "version").read_text().lower()
    except OSError:
        return False


def check_host_memory(
    *,
    weights: str = WEIGHTS,
    subfolder: str | None = None,
    base_model: str = BASE_MODEL,
    text_encoder_precision: str = "bf16",
    text_encoder_fp8_weights: str = TEXT_ENCODER_FP8_WEIGHTS,
    offload: str,
    num_gpus: int,
    height: int,
    width: int,
    num_frames: int,
    workflow: str = "t2va",
    proc: Path = PROC,
    cgroup_root: Path = CGROUP_ROOT,
) -> None:
    available = available_gib(proc, cgroup_root)
    if available is None:
        return
    sizes = weight_bytes(
        weights=weights,
        subfolder=subfolder,
        base_model=base_model,
        text_encoder_precision=text_encoder_precision,
        text_encoder_fp8_weights=text_encoder_fp8_weights,
        workflow=workflow,
    ) or default_weight_bytes(text_encoder_precision)
    need = estimate_gib(
        sizes,
        offload=offload,
        num_gpus=num_gpus,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    if need > available:
        wsl_hint = "; on WSL2, raise memory= in %UserProfile%\\.wslconfig" if in_wsl(proc) else ""
        log.warning(
            "estimated host RAM needed is ~%.0f GiB, but %.0f GiB is available "
            "(consider --text-encoder-precision fp8 or the GGUF runtime on <= 64 GB machines%s)",
            need,
            available,
            wsl_hint,
        )
    else:
        log.info("host memory: about %.0f GiB needed, %.0f GiB available", need, available)


__all__ = ["available_gib", "check_host_memory", "default_weight_bytes", "estimate_gib", "in_wsl", "weight_bytes"]
