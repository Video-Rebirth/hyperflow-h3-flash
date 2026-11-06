import sys

import pytest
import torch
from conftest import COMPRESSION, TINY_CONFIG, packed_inputs
from diffusers import MiniMaxH3Transformer3DModel

from hyperflow_h3_flash import HyperFlowH3FlashTransformer3DModel
from hyperflow_h3_flash.int8 import Int8RotLinear
from hyperflow_h3_flash.transformer import (
    ChunkedFeedForward,
    DroppedBlock,
    LowRankLinear,
    set_block_chunk,
    set_ff_chunk,
    set_lowrank_rank_multiple,
)


def flash(**overrides) -> HyperFlowH3FlashTransformer3DModel:
    torch.manual_seed(0)
    return HyperFlowH3FlashTransformer3DModel(**TINY_CONFIG, **(COMPRESSION | overrides)).eval()


def test_compressed_layers_are_built_from_the_config():
    model = flash()
    assert isinstance(model.transformer_blocks[1], DroppedBlock)
    assert list(model.transformer_blocks[1].parameters()) == []
    assert isinstance(model.transformer_blocks[0].attn.to_q, LowRankLinear)
    assert model.transformer_blocks[0].attn.to_q.rank == 8
    assert isinstance(model.transformer_blocks[0].attn.to_out[0], LowRankLinear)
    assert isinstance(model.transformer_blocks[2].attn.to_k, Int8RotLinear)
    assert isinstance(model.transformer_blocks[2].ff.net[2], Int8RotLinear)
    assert type(model.transformer_blocks[2].attn.to_q) is torch.nn.Linear


def test_unknown_quantization_method_is_rejected():
    with pytest.raises(ValueError, match="quantization"):
        flash(quantization={"method": "fp8", "group": 16, "linears": []})


def test_forward_needs_the_endpoint():
    model = flash()
    with pytest.raises(ValueError, match="endpoint_timestep"):
        model(**packed_inputs(model.config))


def test_forward_shapes():
    model = flash()
    inputs = packed_inputs(model.config)
    video, audio = model(**inputs, endpoint_timestep=torch.tensor([0.6]), return_dict=False)
    assert video.shape[:2] == inputs["hidden_states"].shape[:2]
    assert audio.shape == inputs["audio_hidden_states"].shape
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_an_endpoint_equal_to_t_reduces_to_the_base_model():
    """With ``emb_r == emb_t`` and ``r == t``, ``temb(t, r) = emb_t(t)``: the output is the base model's, bitwise."""
    torch.manual_seed(0)
    base = MiniMaxH3Transformer3DModel(**TINY_CONFIG).eval()
    model = HyperFlowH3FlashTransformer3DModel(**TINY_CONFIG).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and all(key.startswith("endpoint_time_embedder.") for key in missing)
    model.endpoint_time_embedder.load_state_dict(base.time_embedder.state_dict())

    inputs = packed_inputs(base.config)
    with torch.no_grad():
        expected = base(**inputs, return_dict=False)
        actual = model(**inputs, endpoint_timestep=inputs["timestep"].clone(), return_dict=False)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_feed_forward_chunks_keep_the_checkpoint_keys_and_the_function():
    """Chunks change only the row count of each GEMM, so the outputs agree to rounding, not bitwise."""
    model = flash()
    assert "transformer_blocks.0.ff.net.0.proj.weight" in model.state_dict()
    inputs = dict(packed_inputs(model.config), endpoint_timestep=torch.tensor([0.6]), return_dict=False)
    with torch.no_grad():
        whole = model(**inputs)
        for module in model.modules():
            if isinstance(module, ChunkedFeedForward):
                module.chunk_tokens = 5
        chunked = model(**inputs)
    for a, e in zip(chunked, whole, strict=True):
        torch.testing.assert_close(a, e)


def run(model, dtype=torch.float32):
    inputs = packed_inputs(model.config)
    inputs = {k: v.to(dtype) if v.is_floating_point() and k != "position_ids" else v for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs, endpoint_timestep=torch.tensor([0.6], dtype=dtype), return_dict=False)


def test_low_rank_re_padding_adds_exact_zeros():
    """Ranks padded to 64 at load; re-padding to 8 trims the zeros and computes the same function."""
    model = flash().double()
    to_q, to_out = model.transformer_blocks[0].attn.to_q, model.transformer_blocks[0].attn.to_out[0]
    # In the tiny model ranks 8 and 16 are full; zero the tail of one so the used rank is 5.
    to_out.down.weight.data[5:] = 0
    to_out.up.weight.data[:, 5:] = 0
    expected = run(model, torch.float64)

    assert model.pad_lowrank_ranks() == 2
    assert (to_q.rank, to_out.rank, to_q.down.weight.shape, to_q.up.weight.shape) == (64, 64, (64, 64), (64, 64))
    padded = run(model, torch.float64)
    assert set_lowrank_rank_multiple(model, 8) == 2
    assert (to_q.rank, to_out.rank, to_out.down.out_features, to_out.up.in_features) == (8, 8, 8, 8)
    assert set_lowrank_rank_multiple(model, 8) == 0
    trimmed = run(model, torch.float64)
    assert set_lowrank_rank_multiple(model, 0) == 1 and to_out.rank == 5
    for outputs in (padded, trimmed, run(model, torch.float64)):
        for a, e in zip(outputs, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-12)


def test_ff_chunk_zero_runs_the_whole_sequence():
    model = flash()
    set_ff_chunk(model, 0)
    assert all(m.chunk_tokens == sys.maxsize for m in model.modules() if isinstance(m, ChunkedFeedForward))
    set_ff_chunk(model, 7)
    assert all(m.chunk_tokens == 7 for m in model.modules() if isinstance(m, ChunkedFeedForward))


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_block_chunk_computes_the_same_block(chunk):
    model = flash()
    expected = run(model)
    set_block_chunk(model, chunk)
    assert all("forward" in vars(block) for block in model.transformer_blocks if not isinstance(block, DroppedBlock))
    for a, e in zip(run(model), expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)


def test_q_and_k_keep_their_quantized_input_for_the_next_linear():
    model = flash(
        quantization={
            "method": "int8_convrot",
            "group": 16,
            "linears": [f"transformer_blocks.2.attn.{name}" for name in ("to_q", "to_k", "to_v")],
        }
    )
    attn = model.transformer_blocks[2].attn
    assert (attn.to_q.keep_input, attn.to_k.keep_input, attn.to_v.keep_input) == (True, True, False)


def test_padding_the_ranks_keeps_the_function():
    """Zero rows of ``down`` and columns of ``up`` add zero terms; only the GEMM blocking can change the rounding."""
    model = flash()
    inputs = dict(packed_inputs(model.config), endpoint_timestep=torch.tensor([0.6]), return_dict=False)
    with torch.no_grad():
        expected = model(**inputs)
        assert model.pad_lowrank_ranks(multiple=64) == 2
        actual = model(**inputs)
    to_q = model.transformer_blocks[0].attn.to_q
    assert (to_q.rank, to_q.down.weight.shape[0], to_q.up.weight.shape[1]) == (64, 64, 64)
    assert set(model.config.lowrank_ranks.values()) == {64}
    assert model.pad_lowrank_ranks(multiple=64) == 0
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


def test_save_and_load_keep_the_storage_dtypes(tmp_path):
    model = flash()
    model.save_pretrained(tmp_path)
    loaded = HyperFlowH3FlashTransformer3DModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    layer = loaded.transformer_blocks[2].attn.to_k
    assert layer.qweight.dtype == torch.int8 and layer.wscale.dtype == torch.float32
    assert loaded.transformer_blocks[0].attn.to_q.down.weight.dtype == torch.bfloat16
    assert loaded.endpoint_time_embedder.linear_1.weight.dtype == torch.float32
    assert loaded.time_embedder.linear_1.weight.dtype == torch.float32
    assert loaded.config.dropped_blocks == [1]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].to(value.dtype), value, rtol=0, atol=1e-2)
