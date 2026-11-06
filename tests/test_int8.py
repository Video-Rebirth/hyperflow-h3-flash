import contextlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from hyperflow_h3_flash.int8 import (
    KERNEL_OPTIONS,
    Int8RotLinear,
    _kernels,
    hadamard_factors,
    regular_hadamard,
    resolve_int8_gemm,
    set_kernel_options,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@contextlib.contextmanager
def kernel_options(**options):
    saved = dict(KERNEL_OPTIONS)
    set_kernel_options(**options)
    try:
        yield
    finally:
        KERNEL_OPTIONS.update(saved)


def quantize(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The checkpoint format: ``qweight = round(W R / wscale)``, one scale per output channel."""
    rotation = torch.block_diag(*[regular_hadamard(group).double()] * (weight.shape[1] // group))
    rotated = weight.double() @ rotation
    wscale = rotated.abs().amax(1) / 127.0
    return (rotated / wscale[:, None]).round().clamp(-127, 127).to(torch.int8), wscale.float()


@pytest.mark.parametrize("group", [4, 16, 64, 256])
def test_regular_hadamard_is_orthonormal_and_symmetric(group):
    h = regular_hadamard(group).double()
    torch.testing.assert_close(h @ h.T, torch.eye(group, dtype=torch.float64))
    torch.testing.assert_close(h, h.T)


def test_regular_hadamard_rejects_non_powers_of_four():
    with pytest.raises(ValueError):
        regular_hadamard(32)


def test_int8_linear_matches_dense():
    torch.manual_seed(0)
    weight = torch.randn(96, 128) * 0.05
    x = torch.randn(5, 7, 128)
    layer = Int8RotLinear(128, 96, group=16)
    layer.qweight.copy_(quantize(weight, 16)[0])
    layer.wscale.copy_(quantize(weight, 16)[1])

    y = layer(x)
    reference = x @ weight.T
    assert y.shape == (5, 7, 96) and y.dtype == x.dtype
    error = (y.float() - reference).norm() / reference.norm()
    assert error < 0.02, error


def test_int8_linear_state_is_two_frozen_parameters():
    # Streamed group offloading in diffusers returns parameters to the host, but not buffers.
    layer = Int8RotLinear(64, 32, group=16)
    assert set(layer.state_dict()) == {"qweight", "wscale"}
    assert not list(layer.buffers())
    assert not any(p.requires_grad for p in layer.parameters())


@pytest.mark.parametrize("group", [256, 1024, 4096])
def test_hadamard_factors_rebuild_the_rotation(group):
    ga, gb = hadamard_factors(group)
    assert ga >= 16 and gb >= 16
    torch.testing.assert_close(torch.kron(regular_hadamard(ga), regular_hadamard(gb)), regular_hadamard(group))


def test_small_groups_have_no_factors():
    assert hadamard_factors(16) is None and hadamard_factors(64) is None


def test_kernel_options_are_checked_and_default_to_the_release_arithmetic():
    assert KERNEL_OPTIONS == {"int8_gemm": "auto", "rotate_quant": "dense"}
    with pytest.raises(ValueError, match="int8_gemm"):
        set_kernel_options(int8_gemm="cutlass")
    with pytest.raises(ValueError, match="rotate_quant"):
        set_kernel_options(rotate_quant="fast")


def layer_and_input(rows=1000, in_features=5120, out_features=5120, seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(out_features, in_features) * 0.02
    layer = Int8RotLinear(in_features, out_features, group=256)
    qweight, wscale = quantize(weight, 256)
    layer.qweight.copy_(qweight)
    layer.wscale.copy_(wscale)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16)
    x[:, ::97] *= 30  # outlier channels, as in the real activations
    return layer.cuda(), x.cuda()


@cuda
def test_factored_rotation_rounds_like_the_dense_one():
    kernels = _kernels()
    _, x = layer_and_input()
    with kernel_options(rotate_quant="dense"):
        q_dense, s_dense = kernels.rotate_quant(x, 256)
    with kernel_options(rotate_quant="factored"):
        q_factored, s_factored = kernels.rotate_quant(x, 256)
    torch.testing.assert_close(s_factored, s_dense, rtol=1e-5, atol=0)
    differ = q_factored != q_dense
    assert differ.float().mean() <= 1e-5
    assert (q_factored.int() - q_dense.int()).abs().max() <= 1


@cuda
@pytest.mark.skipif(importlib.util.find_spec("sgl_kernel") is None, reason="needs sgl_kernel")
def test_sgl_gemm_is_within_one_ulp_of_triton():
    layer, x = layer_and_input()
    with kernel_options(int8_gemm="triton"):
        expected = layer(x)
    with kernel_options(int8_gemm="sgl"):
        assert resolve_int8_gemm(x.device) == "sgl"
        actual = layer(x)
    ulp = torch.maximum(expected.float().abs(), actual.float().abs()) * 2.0**-7
    assert ((actual.float() - expected.float()).abs() <= ulp + 1e-30).all()


@cuda
@pytest.mark.parametrize("gemm", ["cublas", "bf16"])
def test_int8_gemms_are_bitwise_triton(gemm):
    if gemm == "cublas" and not hasattr(torch, "_int_mm"):
        pytest.skip("needs torch._int_mm")
    if gemm == "bf16" and tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 8):
        pytest.skip("needs torch.mm(out_dtype=), PyTorch 2.8+")
    # 14336 is the widest input of the transformer (the feed-forward down projection).
    layer, x = layer_and_input(rows=300, in_features=14336, out_features=256)
    with kernel_options(int8_gemm="triton"):
        expected = layer(x)
    with kernel_options(int8_gemm=gemm):
        assert resolve_int8_gemm(x.device) == gemm
        actual = layer(x)
    assert torch.equal(actual, expected)


@cuda
def test_q_k_v_share_one_quantization(monkeypatch):
    kernels = _kernels()
    calls = []
    rotate_quant = kernels.rotate_quant
    monkeypatch.setattr(kernels, "rotate_quant", lambda *a: calls.append(1) or rotate_quant(*a))
    to_q, x = layer_and_input(in_features=1024, out_features=512)
    to_k, _ = layer_and_input(in_features=1024, out_features=512, seed=1)
    to_v, _ = layer_and_input(in_features=1024, out_features=512, seed=2)
    to_q.keep_input = to_k.keep_input = True
    shared = [to_q(x), to_k(x), to_v(x)]
    assert len(calls) == 1 and not Int8RotLinear._last_input
    to_q.keep_input = to_k.keep_input = False
    separate = [to_q(x), to_k(x), to_v(x)]
    assert len(calls) == 4
    for a, b in zip(shared, separate, strict=True):
        assert torch.equal(a, b)
    # An in-place change or another tensor is quantized again.
    to_q.keep_input = True
    to_q(x)
    x.mul_(2)
    to_k(x)
    to_q(x)
    to_k(x.clone())
    assert len(calls) == 8


def test_shared_quantization_works_under_inference_mode():
    calls = []
    kernels = SimpleNamespace(rotate_quant=lambda x2, group: calls.append(1) or (x2, group))
    to_q, to_k = Int8RotLinear(16, 16, group=16), Int8RotLinear(16, 16, group=16)
    to_q.keep_input = True
    with torch.inference_mode():
        x = torch.randn(4, 16)
        to_q._quantize(x, x, kernels)
        to_k._quantize(x, x, kernels)
    assert len(calls) == 1 and not Int8RotLinear._last_input


@pytest.mark.parametrize("arch", [86, 89, 120])
def test_rotation_kernels_compile_for_consumer_gpus(arch):
    """sm_86 (RTX 3090), sm_89 (RTX 4090) and sm_120 (RTX 5090) have 99 KiB of shared memory per block."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    kernels = _kernels()
    if kernels is None or not torch.cuda.is_available():
        pytest.skip("needs Triton with the CUDA backend")
    tail = {"q_ptr": "*i8", "s_ptr": "*fp32", "M": "i32", "K": "i32", "stride_xm": "i32", "stride_qm": "i32"}
    dense = {"x_ptr": "*bf16", "h_ptr": "*bf16", **tail, "G": "constexpr", "BM": "constexpr", "BN": "constexpr"}
    factored = {"x_ptr": "*bf16", "ha_ptr": "*bf16", "hb_ptr": "*bf16", **tail}
    factored.update(GA="constexpr", GB="constexpr", BM="constexpr")
    for fn, signature, constexprs, warps in (
        (kernels.jit.rotate_quant, dense, {"G": 256, "BM": 64, "BN": 64}, 8),
        (kernels.jit.rotate_quant, dense, {"G": 256, "BM": 64, "BN": 128}, 8),
        (kernels.jit.rotate_quant_factored, factored, {"GA": 16, "GB": 16, "BM": 16}, 4),
    ):
        source = ASTSource(fn, signature, constexprs)
        compiled = triton.compile(source, target=GPUTarget("cuda", arch, 32), options={"num_warps": warps})
        assert compiled.metadata.shared <= 99 * 1024
