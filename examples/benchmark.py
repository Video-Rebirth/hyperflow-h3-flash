#!/usr/bin/env python3
"""Stage timings of HyperFlow H3 Flash: text encoder, transformer (DiT), video and audio VAE decode, per GPU memory.

Run it under torchrun for several GPUs (it does not re-launch itself).

torchrun --nproc-per-node 4 examples/benchmark.py --frames 124,362
torchrun --nproc-per-node 4 examples/benchmark.py --attention fa3 --out results.jsonl
python examples/benchmark.py --frames 124 --reference person.png --reference scene.mp4  # ref2va

Each length runs ``--warmup`` times (Triton compiles and autotunes once per shape) and then ``--repeat`` times; rank 0
prints one JSON line per run and appends it to ``--out``. A stage is the wall time of its pipeline block between two
``torch.cuda.synchronize()``: ``text_encoder``, ``denoise`` (one transformer pass per step, also listed per step) and
``vae_video`` / ``vae_audio`` (the two halves of ``decode``). ``driver_peak_gib`` is the device-wide peak of
``cudaMemGetInfo``, polled every few milliseconds, so it includes the CUDA context and NCCL buffers.
``tokens`` counts the generated video and audio rows; ``sequence_rows`` is the whole packed sequence the transformer
saw (text, references and generated rows). ``peak_rss_gib`` is the process's peak resident host memory so far
(``ru_maxrss``), so it only grows across runs. ``--reference`` (repeatable, in order) selects the ``ref2va`` workflow,
as in the CLI.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from hyperflow_h3_flash.cli import (  # noqa: E402
    load_references,
    non_negative_int,
    reference_kind,
    reference_problem,
)
from hyperflow_h3_flash.distributed import world  # noqa: E402
from hyperflow_h3_flash.int8 import INT8_GEMMS, ROTATE_QUANTS, resolve_int8_gemm  # noqa: E402
from hyperflow_h3_flash.offload import OFFLOAD_MODES  # noqa: E402
from hyperflow_h3_flash.pipeline import (  # noqa: E402
    BASE_MODEL,
    FF_CHUNK_TOKENS,
    FP8_GEMMS,
    FPS,
    TEXT_ENCODER_PRECISIONS,
    TRANSFORMER_COMPONENTS,
    WEIGHTS,
    load_pipeline,
    request_tokens,
    resolve_canvas,
)
from hyperflow_h3_flash.schedule import DEFAULT_STEPS, SIGMA_GRIDS  # noqa: E402

PROMPT = (
    "A chef slides a steaming bowl of ramen across a wooden counter; rain taps on the window behind him and the "
    "customers murmur as he calls out the order."
)
#: The ``ref2va`` default prompt names the references of each kind in order.
REFERENCE_TAGS = {"image": "Picture", "video": "Video", "audio": "Audio"}
#: Nested blocks timed under their own names, matched on the class name.
NESTED = {"VideoDecodeStep": "vae_video", "AudioDecodeStep": "vae_audio"}


def wrap_call(block, wrapper) -> None:
    """Route ``block(components, state)`` through ``wrapper(original_call, components, state)``."""
    cls = type(block)
    original = cls.__call__

    def call(self, components, state):
        return wrapper(lambda: original(self, components, state), components, state)

    block.__class__ = type(cls.__name__, (cls,), {"__call__": call})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", default="124,362", help="comma-separated frame counts, run in this order")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--prompt", help="default: a ramen scene, or for ref2va one naming every reference")
    parser.add_argument("--reference", dest="references", type=Path, action="append", default=[], metavar="PATH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, choices=sorted(SIGMA_GRIDS))
    parser.add_argument("--tag", default="", help="label copied into every record")
    parser.add_argument("--out", type=Path, help="append the records to this JSONL file")
    parser.add_argument("--video-dir", type=Path, help="write the last clip of every length here")
    parser.add_argument("--poll-ms", type=float, default=5.0)

    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--subfolder")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--offload", default="auto", choices=("auto", *OFFLOAD_MODES))
    parser.add_argument("--attention", "--attention-backend", dest="attention", default="auto")
    parser.add_argument("--text-encoder-precision", default="bf16", choices=TEXT_ENCODER_PRECISIONS)
    parser.add_argument("--fp8-gemm", default="triton", choices=FP8_GEMMS)
    parser.add_argument("--int8-gemm", default="auto", choices=INT8_GEMMS)
    parser.add_argument("--rotate-quant", default="factored", choices=ROTATE_QUANTS)
    parser.add_argument("--lowrank-rank-multiple", type=non_negative_int, default=8)
    parser.add_argument("--ff-chunk", type=non_negative_int, default=FF_CHUNK_TOKENS)
    parser.add_argument("--block-chunk", type=non_negative_int, default=0)
    args = parser.parse_args()
    if problem := reference_problem(args.references):
        parser.error(problem)
    if args.prompt is None:
        args.prompt = default_prompt(args.references) if args.references else PROMPT
    return args


def default_prompt(references: list[Path]) -> str:
    counts: dict[str, int] = {}
    names = []
    for path in references:
        kind = reference_kind(path)
        counts[kind] = counts.get(kind, 0) + 1
        names.append(f"<{REFERENCE_TAGS[kind]} {counts[kind]}>")
    return (
        f"A scene built from {', '.join(names)}: the person walks slowly through it, looks around and smiles; the "
        "camera follows at shoulder height."
    )


class Stages:
    """Wall time per pipeline block and per transformer pass of the current run."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.steps: list[float] = []
        self.sequence: int | None = None

    def reset(self) -> None:
        self.times, self.steps, self.sequence = {}, [], None

    def timed(self, name: str):
        def wrapper(call, components, state):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = call()
            torch.cuda.synchronize()
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - started
            return result

        return wrapper

    def instrument(self, pipe, workflow: str) -> list[str]:
        names = []
        for name, block in pipe._blocks.sub_blocks.items():
            wrap_call(block, self.timed(name))
            names.append(name)
        stack = list(pipe._blocks.sub_blocks.values())
        while stack:
            block = stack.pop()
            for suffix, name in NESTED.items():
                if type(block).__name__.endswith(suffix):
                    wrap_call(block, self.timed(name))
                    names.append(name)
            stack.extend((getattr(block, "sub_blocks", None) or {}).values())
        started: list[float] = []

        def pre(_module, _args, kwargs):
            if (position_ids := kwargs.get("position_ids")) is not None:
                self.sequence = position_ids.shape[0]
            torch.cuda.synchronize()
            started.append(time.perf_counter())

        def post(_module, _args, _kwargs, _output):
            torch.cuda.synchronize()
            self.steps.append(time.perf_counter() - started.pop())

        transformer = getattr(pipe, TRANSFORMER_COMPONENTS[workflow])
        transformer.register_forward_pre_hook(pre, with_kwargs=True)
        transformer.register_forward_hook(post, with_kwargs=True)
        return names


class DriverPeak:
    """Device-wide used memory (``cudaMemGetInfo``), the peak since the last `reset`, polled on a thread."""

    def __init__(self, device: torch.device, interval: float) -> None:
        self.device, self.interval, self.peak = device, interval, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info(self.device)
            self.peak = max(self.peak, total - free)
            time.sleep(self.interval)

    def reset(self) -> None:
        free, total = torch.cuda.mem_get_info(self.device)
        self.peak = total - free

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def gather(obj, world_size: int) -> list:
    if world_size == 1:
        return [obj]
    out = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = world()
    height, width = resolve_canvas()
    frames = [int(f) for f in args.frames.split(",") if f]
    tokens = {f: request_tokens(height, width, f) for f in frames}

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(minutes=5))

    torch.cuda.synchronize()
    started = time.perf_counter()
    workflow = "ref2va" if args.references else "t2va"
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
        int8_gemm=args.int8_gemm,
        rotate_quant=args.rotate_quant,
        lowrank_rank_multiple=args.lowrank_rank_multiple,
        ff_chunk=args.ff_chunk,
        block_chunk=args.block_chunk,
    )
    load_s = time.perf_counter() - started
    inputs = {"references": load_references(args.references)} if args.references else {}
    stages = Stages()
    timed_blocks = stages.instrument(pipe, workflow)
    peak = DriverPeak(device, args.poll_ms / 1000)

    props = torch.cuda.get_device_properties(device)
    base = {
        "tag": args.tag,
        "workflow": workflow,
        "references": [path.name for path in args.references],
        "gpu": props.name,
        "gpu_total_gib": round(props.total_memory / 2**30, 2),
        "num_gpus": world_size,
        "resolved": {
            "steps": args.steps,
            "offload": args.offload,
            "attention": args.attention,
            "text_encoder_precision": args.text_encoder_precision,
            "int8_gemm": args.int8_gemm,
            "rotate_quant": args.rotate_quant,
            "lowrank_rank_multiple": args.lowrank_rank_multiple,
            "ff_chunk": args.ff_chunk,
            "block_chunk": args.block_chunk,
        },
        "int8_gemm_used": resolve_int8_gemm(device),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "height": height,
        "width": width,
        "load_s": round(load_s, 1),
    }
    try:
        import diffusers
        import transformers

        base["versions"].update(diffusers=diffusers.__version__, transformers=transformers.__version__)
    except ImportError:
        pass
    if rank == 0:
        print(f"# timed blocks: {timed_blocks}", flush=True)

    for num_frames in frames:
        for run in range(args.warmup + args.repeat):
            stages.reset()
            torch.cuda.reset_peak_memory_stats(device)
            peak.reset()
            if world_size > 1:
                dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipe(
                prompt=args.prompt,
                num_frames=num_frames,
                num_inference_steps=args.steps,
                generator=torch.Generator().manual_seed(args.seed),
                output=["videos", "audio", "sampling_rate"],
                **inputs,
            )
            torch.cuda.synchronize()
            total = time.perf_counter() - started
            memory = gather(
                {
                    "allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                    "reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
                    "driver_peak_gib": round(peak.peak / 2**30, 2),
                    "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30, 2),
                },
                world_size,
            )
            warmup = run < args.warmup
            if rank == 0:
                record = dict(
                    base,
                    num_frames=num_frames,
                    seconds=round(num_frames / FPS, 2),
                    tokens=tokens[num_frames],
                    sequence_rows=stages.sequence,
                    run=run,
                    warmup=warmup,
                    total_s=round(total, 2),
                    stages_s={k: round(v, 2) for k, v in stages.times.items()},
                    steps_s=[round(s, 3) for s in stages.steps],
                    memory=memory,
                )
                line = json.dumps(record)
                print(line, flush=True)
                if args.out is not None:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    with args.out.open("a") as f:
                        f.write(line + "\n")
                last = run == args.warmup + args.repeat - 1
                if last and args.video_dir is not None:
                    from diffusers.utils.export_utils import encode_video

                    args.video_dir.mkdir(parents=True, exist_ok=True)
                    path = args.video_dir / f"{args.tag or 'run'}_{num_frames}f.mp4"
                    encode_video(
                        result["videos"][0],
                        fps=FPS,
                        output_path=str(path),
                        audio=result["audio"][0],
                        audio_sample_rate=result["sampling_rate"],
                    )
                    print(f"# wrote {path}", flush=True)
            del result

    peak.stop()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
