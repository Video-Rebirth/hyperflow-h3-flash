import importlib.util

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from hyperflow_h3_flash import pipeline as pipeline_module
from hyperflow_h3_flash.cli import align_num_frames, check_platform, parse_args
from hyperflow_h3_flash.pipeline import (
    FPS,
    MAX_FRAMES,
    admit_max_frames,
    fp8_kernels_problem,
    load_pipeline,
    load_text_encoder,
    resolve_attention,
    resolve_offload,
    resolve_subfolder,
    trim_text_encoder,
)

LAYER = 2


@pytest.fixture
def text_encoder():
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=5,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        ),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32),
    )
    return Qwen3VLForConditionalGeneration(config).eval()


def conditioning(text_encoder) -> torch.Tensor:
    input_ids = torch.arange(1, 9)[None]
    with torch.no_grad():
        outputs = text_encoder.model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
    return outputs.hidden_states[LAYER]


def test_trimming_keeps_the_conditioning_bitwise(text_encoder):
    expected = conditioning(text_encoder)
    trim_text_encoder(text_encoder, LAYER)
    assert len(text_encoder.model.language_model.layers) == LAYER + 1
    assert text_encoder.config.text_config.num_hidden_layers == LAYER + 1
    torch.testing.assert_close(conditioning(text_encoder), expected, rtol=0, atol=0)


def test_trimming_drops_the_language_model_head(text_encoder):
    trim_text_encoder(text_encoder, LAYER)
    assert isinstance(text_encoder.lm_head, torch.nn.Identity)


def test_load_text_encoder_reads_only_the_kept_layers(text_encoder, tmp_path):
    text_encoder.to(torch.bfloat16).save_pretrained(tmp_path)
    expected = conditioning(text_encoder)
    loaded = load_text_encoder(str(tmp_path), subfolder=None, layers=LAYER + 1, device="cpu", local_files_only=True)
    assert len(loaded.model.language_model.layers) == LAYER + 1
    assert loaded.config.text_config.num_hidden_layers == LAYER + 1
    torch.testing.assert_close(conditioning(loaded), expected, rtol=0, atol=0)


def test_load_pipeline_rejects_bad_text_encoder_options(monkeypatch):
    with pytest.raises(ValueError, match="text_encoder_precision"):
        load_pipeline(device="cpu", text_encoder_precision="fp16")
    with pytest.raises(ValueError, match="fp8_gemm"):
        load_pipeline(device="cpu", fp8_gemm="cutlass")
    monkeypatch.setattr(pipeline_module, "fp8_kernels_problem", lambda: "no kernels")
    with pytest.raises(ImportError, match="no kernels"):
        load_pipeline(device="cpu", text_encoder_precision="fp8")


@pytest.mark.parametrize(
    "available, installed, expected",
    [
        (True, "0.17.3", None),
        (False, "0.18.1", "kernels 0.18.1 does not match this transformers"),
        (False, None, "kernels is not installed"),
    ],
)
def test_fp8_kernels_follow_the_transformers_pin(monkeypatch, available, installed, expected):
    import importlib.metadata

    from transformers.utils import import_utils

    def version(name):
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(import_utils, "is_kernels_available", lambda: available)
    monkeypatch.setattr(importlib.metadata, "version", version)
    problem = fp8_kernels_problem()
    if expected is None:
        assert problem is None
    else:
        assert expected in problem and "hyperflow-h3-flash[fp8]" in problem


def test_check_platform(monkeypatch):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: object() if name == "triton" else find_spec(name, *args),
    )
    check_platform(1, "linux")
    check_platform(4, "linux")
    check_platform(1, "win32")
    with pytest.raises(ValueError, match="requires Linux or WSL2"):
        check_platform(2, "win32")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "triton" else find_spec(name, *args),
    )
    with pytest.raises(ValueError, match="triton-windows"):
        check_platform(1, "win32")
    with pytest.raises(ValueError, match="CUDA build of PyTorch"):
        check_platform(1, "linux")


def test_admit_max_frames_raises_duration_cap():
    class FakePipe:
        max_duration = 15.0

    pipe = FakePipe()
    admit_max_frames(pipe)
    assert pipe.max_duration == pytest.approx(MAX_FRAMES / FPS)


def test_resolve_attention(monkeypatch):
    cpu = torch.device("cpu")
    assert resolve_attention(None, cpu) is None
    assert resolve_attention("sdpa", cpu) == "native"
    assert resolve_attention("auto", cpu) is None
    assert resolve_attention("flash_varlen", cpu) == "flash_varlen"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    for backend in ("fa3", "_flash_3_hub"):
        assert resolve_attention(backend, cpu) is None


def test_resolve_offload():
    cpu = torch.device("cpu")
    assert resolve_offload("auto", cpu) == "none"
    assert resolve_offload("stream", cpu) == "stream"
    with pytest.raises(ValueError, match="offload"):
        resolve_offload("modle", cpu)


def test_resolve_subfolder(tmp_path):
    hub = "VideoRebirthAI/hyperflow-h3-flash"
    assert resolve_subfolder(hub, "t2va") == "fl2va_int8_convrot"
    assert resolve_subfolder(hub, "fl2va") == "fl2va_int8_convrot"
    assert resolve_subfolder(hub, "ref2va") == "ref2va_int8_convrot"
    assert resolve_subfolder(hub, "ref2va", "custom") == "custom"
    assert resolve_subfolder(hub, "ref2va", "") is None
    assert resolve_subfolder(str(tmp_path), "fl2va") == "fl2va_int8_convrot"
    (tmp_path / "config.json").write_text("{}")
    assert resolve_subfolder(str(tmp_path), "fl2va") is None


def test_cli_reference_options():
    args = parse_args(["--prompt", "p", "--reference", "a.png", "--reference", "b.MP4", "--reference", "c.wav"])
    assert [path.name for path in args.references] == ["a.png", "b.MP4", "c.wav"]
    assert parse_args(["--prompt", "p", "--subfolder", "custom"]).subfolder == "custom"

    def refs(*names):
        return [arg for name in names for arg in ("--reference", name)]

    nine_images = [f"{i}.png" for i in range(9)]
    assert len(parse_args(["--prompt", "p", *refs(*nine_images, "a.mp4", "b.wav")]).references) == 11
    for bad in (
        ["--prompt", "p", "--reference", "a.png", "--image", "first.png"],
        ["--prompt", "p", "--reference", "notes.txt"],
        ["--prompt", "p", *refs(*[f"{i}.png" for i in range(10)])],
        ["--prompt", "p", *refs("a.png", *[f"{i}.mp4" for i in range(4)])],
        ["--prompt", "p", *refs(*nine_images, "a.mp4", "b.mp4", "c.wav", "d.wav")],
        ["--prompt", "p", *refs("a.wav")],
    ):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_cli_options():
    args = parse_args(["--prompt", "p"])
    assert (args.gpus, args.offload, args.attention, args.num_frames, args.block_chunk) == (1, "auto", "auto", 124, 0)
    assert parse_args(["--prompt", "p", "--num-gpus", "4"]).gpus == 4
    assert parse_args(["--prompt", "p", "--gpus", "auto"]).gpus == "auto"
    assert parse_args(["--prompt", "p", "--gpus", "4", "--offload", "stream"]).offload == "stream"
    assert parse_args(["--prompt", "p", "--num-frames", "360"]).num_frames == 360
    assert align_num_frames(360) == 362
    assert parse_args(["--prompt", "p", "--attention-backend", "fa3"]).attention == "fa3"
    assert (args.text_encoder_precision, args.fp8_gemm) == ("bf16", "triton")
    assert parse_args(["--prompt", "p", "--text-encoder-precision", "fp8"]).text_encoder_precision == "fp8"
    assert parse_args(["--prompt", "p", "--block-chunk", "16384"]).block_chunk == 16384
    for bad in (
        ["--prompt", "p", "--text-encoder-precision", "fp16"],
        ["--prompt", "p", "--gpus", "0"],
        ["--prompt", "p", "--gpus", "two"],
        ["--prompt", "p", "--num-frames", "363"],
        ["--prompt", "p", "--height", "768"],
        ["--prompt", "p", "--height", "760", "--width", "1344"],
        ["--prompt", "p", "--last-image", "last.png"],
        ["--prompt", "p", "--block-chunk", "-1"],
    ):
        with pytest.raises(SystemExit):
            parse_args(bad)
