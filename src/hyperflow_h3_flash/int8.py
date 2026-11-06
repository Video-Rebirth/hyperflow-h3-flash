# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""INT8 W8A8 linears with a Hadamard-rotated input dimension.

A dense ``y = x @ W^T`` is stored as ``qweight = round(W R / wscale)`` (int8, one fp32 scale per output channel), where
``R`` is a block-diagonal regular Hadamard of size ``group``. ``R`` is orthonormal, so ``y = (x R) @ (W R)^T``: at run
time the activation is rotated and quantized per token, multiplied in int8 with an exact int32 accumulator, and
rescaled. The rotation spreads outlier channels across the group, which is what makes per-token int8 activations work.

On CUDA the rotation + quantization and the GEMM are Triton kernels (compiled on first use). Every kernel fits the
99 KiB of shared memory per block of Ampere, Ada and consumer Blackwell GPUs. The int8 GEMM runs as a Triton kernel or
as cuBLASLt (``torch._int_mm``) plus a Triton epilogue; both use the same int32 accumulator and the same fp32 multiply
order, so they give bitwise identical outputs. cuBLASLt is faster from compute capability 10 on, Triton on Hopper.
The B300 (compute capability 10.3) has INT8 tensor cores at about 1/20 of the B200's, so there the int8 codes are
multiplied on bf16 tensor cores into an fp32 accumulator: products are exact, and the sum matches the int32 one while
it stays below 2**24, which the rotated activations keep well clear of.
Elsewhere a plain torch implementation of the same scheme runs.

`KERNEL_OPTIONS` (set by `load_pipeline`, no weight change) picks the kernels at run time:

* ``int8_gemm``: ``"auto"`` (the choice above), ``"triton"``, ``"cublas"``, ``"bf16"``, or ``"sgl"``: CUTLASS
  ``sgl_kernel.int8_scaled_mm`` with the scales in its epilogue, within 1 ulp of Triton and 1.25 to 1.45x faster on
  H200 (Triton when sgl_kernel is missing).
* ``rotate_quant``: ``"dense"`` (the kernel above) or ``"factored"``: a 256-channel group rotated as ``H16 X H16``, two
  K = 16 dots in 8.5 KiB of shared memory, 1.6 to 2.0x faster on H200; it rounds the rotated activation like the dense
  kernel up to rare one-code differences.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

#: Run-time kernel choices.
INT8_GEMMS = ("auto", "triton", "cublas", "sgl", "bf16")
ROTATE_QUANTS = ("dense", "factored")

#: Rows below this go through the torch path: `torch._int_mm` needs more than 16 rows, and so few rows are not worth
#: a kernel launch.
_MIN_KERNEL_ROWS = 17

#: Run-time kernel choices for every `Int8RotLinear`; `load_pipeline` sets them.
KERNEL_OPTIONS: dict[str, Any] = {"int8_gemm": "auto", "rotate_quant": "dense"}


def set_kernel_options(*, int8_gemm: str | None = None, rotate_quant: str | None = None) -> None:
    if int8_gemm is not None:
        if int8_gemm not in INT8_GEMMS:
            raise ValueError(f"int8_gemm must be one of {INT8_GEMMS}, got {int8_gemm!r}")
        KERNEL_OPTIONS["int8_gemm"] = int8_gemm
    if rotate_quant is not None:
        if rotate_quant not in ROTATE_QUANTS:
            raise ValueError(f"rotate_quant must be one of {ROTATE_QUANTS}, got {rotate_quant!r}")
        KERNEL_OPTIONS["rotate_quant"] = rotate_quant


@functools.cache
def _sgl_int8_scaled_mm():
    """sgl_kernel's ``int8_scaled_mm``, or None when sgl_kernel is not importable."""
    try:
        from sgl_kernel import int8_scaled_mm
    except ImportError:
        return None
    return int8_scaled_mm


def _has_bf16_mm() -> bool:
    # `torch.mm(..., out_dtype=)` is PyTorch 2.8+.
    return tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 8)


@functools.cache
def _auto_int8_gemm(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 3) and _has_bf16_mm():
        return "bf16"
    if hasattr(torch, "_int_mm") and capability[0] >= 10:
        return "cublas"
    return "triton"


def resolve_int8_gemm(device: torch.device) -> str:
    """The GEMM ``KERNEL_OPTIONS["int8_gemm"]`` runs as on ``device``: ``"triton"``, ``"cublas"``, ``"bf16"`` or
    ``"sgl"``."""
    choice = KERNEL_OPTIONS["int8_gemm"]
    if choice == "auto":
        return _auto_int8_gemm(device)
    if (
        (choice == "sgl" and _sgl_int8_scaled_mm() is None)
        or (choice == "cublas" and not hasattr(torch, "_int_mm"))
        or (choice == "bf16" and not _has_bf16_mm())
    ):
        return "triton"
    return choice


@functools.cache
def regular_hadamard(group: int) -> Tensor:
    """``kron(H4 / 2, ..., H4 / 2)`` of size ``group`` (a power of 4): orthonormal, symmetric, rows sum to one."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64) / 2
    h, n = torch.ones(1, 1, dtype=torch.float64), 1
    while n < group:
        h, n = torch.kron(h, h4), n * 4
    if n != group:
        raise ValueError(f"Hadamard group {group} is not a power of 4")
    return h.float()


def hadamard_factors(group: int) -> tuple[int, int] | None:
    """``(GA, GB)`` with ``regular_hadamard(group) == kron(regular_hadamard(GA), regular_hadamard(GB))``, both at
    least 16 (the smallest bf16 ``tl.dot`` K); None when ``group`` cannot be split that way."""
    n = group.bit_length() - 1
    if group < 256 or group != 1 << n or n % 2:
        return None
    ga = 4 ** (n // 4)
    return ga, group // ga


@functools.cache
def _rotation(group: int, device: torch.device) -> Tensor:
    return regular_hadamard(group).to(device=device, dtype=torch.bfloat16)


class Int8RotLinear(nn.Module):
    """Bias-free W8A8 linear: ``y = (q(x R) @ qweight^T) * s_x * wscale``, ``R = blockdiag(regular Hadamard)``.

    Linears that read one input (``to_q``, ``to_k``, ``to_v``) quantize it once: a layer with ``keep_input = True``
    leaves its quantized input for the next layer, which reuses it if it receives the exact same, unmodified tensor.
    Any other layer clears it, so nothing outlives the projections.
    """

    _last_input: ClassVar[dict[str, Any]] = {}

    def __init__(self, in_features: int, out_features: int, group: int) -> None:
        super().__init__()
        if in_features % group:
            raise ValueError(f"in_features {in_features} is not a multiple of the Hadamard group {group}")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.keep_input = False
        # Parameters, not buffers: diffusers' streamed group offloading moves module buffers on but never back off.
        self.qweight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.wscale = nn.Parameter(torch.ones(out_features, dtype=torch.float32), requires_grad=False)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group={self.group}"

    def _quantize(self, x: Tensor, x2: Tensor, kernels: SimpleNamespace) -> tuple[Tensor, Tensor]:
        cache = Int8RotLinear._last_input
        # Inference tensors (`torch.inference_mode`) have no version counter.
        version = None if x.is_inference() else x._version
        hit = cache.get("x") is x and cache.get("version") == version and cache.get("group") == self.group
        quantized = cache["value"] if hit else kernels.rotate_quant(x2, self.group)
        cache.clear()
        if self.keep_input:
            cache.update(x=x, version=version, group=self.group, value=quantized)
        return quantized

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features).to(torch.bfloat16).contiguous()
        kernels = _kernels() if x2.is_cuda and x2.shape[0] >= _MIN_KERNEL_ROWS else None
        if kernels is None:
            Int8RotLinear._last_input.clear()
            y = self._forward_torch(x2, _rotation(self.group, x2.device))
        else:
            xq, sx = self._quantize(x, x2, kernels)
            y = kernels.int8_matmul(xq, self.qweight, sx, self.wscale)
        return y.reshape(*shape[:-1], self.out_features).to(x.dtype)

    def _forward_torch(self, x2: Tensor, rot: Tensor) -> Tensor:
        """The same scheme in plain torch; the rotated activation is rounded to bf16 before quantization."""
        g = self.group
        xr = (x2.reshape(-1, self.in_features // g, g) @ rot).reshape(x2.shape).float()
        sx = xr.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
        xq = (xr / sx).round().clamp(-127, 127)
        return ((xq @ self.qweight.float().t()) * sx * self.wscale[None, :]).to(torch.bfloat16)


@functools.cache
def _kernels() -> SimpleNamespace | None:
    """``rotate_quant`` and ``int8_matmul`` as Triton kernels; ``None`` when Triton is not installed."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    ROTATE_BM = 64
    FACTORED_BM = 16

    @triton.jit
    def rotate_quant_kernel(
        x_ptr, h_ptr, q_ptr, s_ptr, M, K, stride_xm, stride_qm,
        G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):  # fmt: skip
        # Two passes over the row block: the per-token scale needs the absmax of the whole rotated row first.
        # The Hadamard streams through in G x BN column tiles, so shared memory is (BM * G + G * BN) * 2 bytes.
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rows64 = rows.to(tl.int64)
        cols = tl.arange(0, G)
        tile = tl.arange(0, BN)
        rmask = rows < M
        amax = tl.zeros((BM,), dtype=tl.float32)
        for g in range(0, K, G):
            x = tl.load(x_ptr + rows64[:, None] * stride_xm + (g + cols)[None, :], mask=rmask[:, None], other=0.0)
            for n in range(0, G, BN):
                h = tl.load(h_ptr + cols[:, None] * G + (n + tile)[None, :])
                amax = tl.maximum(amax, tl.max(tl.abs(tl.dot(x, h)), axis=1))
        scale = tl.maximum(amax, 1e-12) / 127.0
        tl.store(s_ptr + rows, scale, mask=rmask)
        inv = 1.0 / scale
        for g in range(0, K, G):
            x = tl.load(x_ptr + rows64[:, None] * stride_xm + (g + cols)[None, :], mask=rmask[:, None], other=0.0)
            for n in range(0, G, BN):
                h = tl.load(h_ptr + cols[:, None] * G + (n + tile)[None, :])
                q = tl.extra.cuda.libdevice.rint(tl.dot(x, h) * inv[:, None])
                q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
                tl.store(q_ptr + rows64[:, None] * stride_qm + (g + n + tile)[None, :], q, mask=rmask[:, None])

    @triton.jit
    def rotate_quant_factored_kernel(
        x_ptr, ha_ptr, hb_ptr, q_ptr, s_ptr, M, K, stride_xm, stride_qm,
        GA: tl.constexpr, GB: tl.constexpr, BM: tl.constexpr,
    ):  # fmt: skip
        # Channel g + a * GB + b of a group is X[a, b]; the rotated group is H_GA X H_GB. Pass 0 takes the row absmax,
        # pass 1 rotates again and writes int8. The fp32 input of the second dot is split into three bf16 terms
        # (8 + 8 + 8 significand bits), so it is exact; a tf32 dot or a two-term split rounds it measurably.
        # The second dot is batched per row, so its result is already in channel order.
        G: tl.constexpr = GA * GB
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rows64 = rows.to(tl.int64)
        cols = tl.arange(0, G)
        rmask = rows < M
        ia = tl.arange(0, GA)
        ib = tl.arange(0, GB)
        ha = tl.broadcast_to(tl.load(ha_ptr + ia[:, None] * GA + ia[None, :])[None, :, :], (BM, GA, GA))
        hb = tl.load(hb_ptr + ib[:, None] * GB + ib[None, :])
        amax = tl.zeros((BM,), dtype=tl.float32)
        inv = tl.zeros((BM,), dtype=tl.float32)
        for p in tl.static_range(2):
            for g in range(0, K, G):
                x = tl.load(x_ptr + rows64[:, None] * stride_xm + (g + cols)[None, :], mask=rmask[:, None], other=0.0)
                z = tl.reshape(tl.dot(tl.reshape(x, (BM * GA, GB)), hb), (BM, GA, GB))
                z_hi = z.to(tl.bfloat16)
                z = z - z_hi.to(tl.float32)
                z_mid = z.to(tl.bfloat16)
                z_lo = (z - z_mid.to(tl.float32)).to(tl.bfloat16)
                y = tl.dot(ha, z_lo, acc=tl.dot(ha, z_mid, acc=tl.dot(ha, z_hi)))
                y = tl.reshape(y, (BM, G))
                if p == 0:
                    amax = tl.maximum(amax, tl.max(tl.abs(y), axis=1))
                else:
                    q = tl.extra.cuda.libdevice.rint(y * inv[:, None])
                    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
                    tl.store(q_ptr + rows64[:, None] * stride_qm + (g + cols)[None, :], q, mask=rmask[:, None])
            if p == 0:
                scale = tl.maximum(amax, 1e-12) / 127.0
                tl.store(s_ptr + rows, scale, mask=rmask)
                inv = 1.0 / scale

    @functools.cache
    def rotate_tile(device_index: int, group: int) -> int:
        # At group 256, 128 columns take 96 KiB: that fits every GPU from Ampere on (99 KiB per block).
        shared = triton.runtime.driver.active.utils.get_device_properties(device_index)["max_shared_mem"]
        tile = 128 if (ROTATE_BM * group + group * 128) * 2 <= shared else 64
        return min(tile, group)

    configs = [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_stages=stages, num_warps=warps)
        for bm, bn, bk, stages, warps in [
            (128, 256, 128, 3, 8),
            (256, 128, 128, 3, 8),
            (128, 128, 128, 4, 8),
            (128, 256, 64, 4, 8),
            (256, 128, 64, 4, 8),
            (128, 128, 64, 5, 4),
        ]
    ]

    @triton.autotune(configs=configs, key=["M_BUCKET", "N", "K"])
    @triton.jit
    def int8_mm_kernel(
        a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K, M_BUCKET, stride_am, stride_bn, stride_cm,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
    ):  # fmt: skip
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BM)
        num_n = tl.cdiv(N, BN)
        group = GROUP_M * num_n
        first_m = (pid // group) * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (pid % group) % size_m
        pid_n = (pid % group) // size_m
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm.to(tl.int64)[:, None] * stride_am + rk[None, :]
        b_ptrs = b_ptr + rn.to(tl.int64)[None, :] * stride_bn + rk[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0)
            b = tl.load(b_ptrs, mask=(rn[None, :] < N) & (k + rk[:, None] < K), other=0)
            acc += tl.dot(a, b, out_dtype=tl.int32)
            a_ptrs += BK
            b_ptrs += BK
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
        sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
        c = (acc.to(tl.float32) * sa[:, None] * sb[None, :]).to(tl.bfloat16)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :], c, mask=mask)

    @triton.jit
    def rescale_kernel(acc_ptr, sa_ptr, sb_ptr, c_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        offsets = rm.to(tl.int64)[:, None] * N + rn[None, :]
        acc = tl.load(acc_ptr + offsets, mask=mask, other=0)
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
        sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
        tl.store(c_ptr + offsets, (acc.to(tl.float32) * sa[:, None] * sb[None, :]).to(tl.bfloat16), mask=mask)

    def rotate_quant(x: Tensor, group: int) -> tuple[Tensor, Tensor]:
        M, K = x.shape
        q = torch.empty(M, K, device=x.device, dtype=torch.int8)
        s = torch.empty(M, device=x.device, dtype=torch.float32)
        factors = hadamard_factors(group) if KERNEL_OPTIONS["rotate_quant"] == "factored" else None
        if factors is not None:
            ga, gb = factors
            ha, hb = _rotation(ga, x.device), _rotation(gb, x.device)
            rotate_quant_factored_kernel[(triton.cdiv(M, FACTORED_BM),)](
                x, ha, hb, q, s, M, K, x.stride(0), q.stride(0), GA=ga, GB=gb, BM=FACTORED_BM, num_warps=4,
                num_stages=4,
            )  # fmt: skip
            return q, s
        BN = rotate_tile(x.device.index, group)
        rotate_quant_kernel[(triton.cdiv(M, ROTATE_BM),)](
            x, _rotation(group, x.device), q, s, M, K, x.stride(0), q.stride(0), G=group, BM=ROTATE_BM, BN=BN,
            num_warps=8, num_stages=1,
        )  # fmt: skip
        return q, s

    def matmul_triton(a: Tensor, b: Tensor, sa: Tensor, sb: Tensor, c: Tensor) -> None:
        M, K = a.shape
        N = b.shape[0]
        grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)  # noqa: E731
        m_bucket = 1 << max(M - 1, 1).bit_length()
        int8_mm_kernel[grid](a, b, c, sa, sb, M, N, K, m_bucket, a.stride(0), b.stride(0), c.stride(0))

    def matmul_cublas(a: Tensor, b: Tensor, sa: Tensor, sb: Tensor, c: Tensor) -> None:
        # Row chunks keep the int32 accumulator near 1 GiB.
        M, N = a.shape[0], b.shape[0]
        chunk = max(128, ((1 << 28) // N) // 128 * 128)
        for start in range(0, M, chunk):
            stop = min(M, start + chunk)
            if stop - start < _MIN_KERNEL_ROWS:
                matmul_triton(a[start:stop], b, sa[start:stop], sb, c[start:stop])
                continue
            acc = torch._int_mm(a[start:stop], b.t())
            grid = (triton.cdiv(stop - start, 64), triton.cdiv(N, 128))
            rescale_kernel[grid](acc, sa[start:stop], sb, c[start:stop], stop - start, N, BM=64, BN=128, num_warps=8)

    def matmul_bf16(a: Tensor, b: Tensor, sa: Tensor, sb: Tensor, c: Tensor) -> None:
        M, N = a.shape[0], b.shape[0]
        weight = b.to(torch.bfloat16)
        chunk = max(128, ((1 << 28) // N) // 128 * 128)
        for start in range(0, M, chunk):
            stop = min(M, start + chunk)
            acc = torch.mm(a[start:stop].to(torch.bfloat16), weight.t(), out_dtype=torch.float32)
            grid = (triton.cdiv(stop - start, 64), triton.cdiv(N, 128))
            rescale_kernel[grid](acc, sa[start:stop], sb, c[start:stop], stop - start, N, BM=64, BN=128, num_warps=8)

    def int8_matmul(a: Tensor, b: Tensor, sa: Tensor, sb: Tensor) -> Tensor:
        M, K = a.shape
        N = b.shape[0]
        gemm = resolve_int8_gemm(a.device)
        if gemm == "sgl" and K % 16 == 0 and N % 16 == 0:
            # b is the [N, K] row-major qweight; its transpose is the column-major [K, N] operand CUTLASS expects.
            return _sgl_int8_scaled_mm()(a, b.t(), sa, sb, torch.bfloat16)
        c = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
        if gemm == "bf16":
            matmul_bf16(a, b, sa, sb, c)
        elif gemm == "cublas" and K % 8 == 0 and N % 8 == 0:
            matmul_cublas(a, b, sa, sb, c)
        else:
            matmul_triton(a, b, sa, sb, c)
        return c

    jit = SimpleNamespace(
        rotate_quant=rotate_quant_kernel,
        rotate_quant_factored=rotate_quant_factored_kernel,
        int8_mm=int8_mm_kernel,
        rescale=rescale_kernel,
    )
    return SimpleNamespace(rotate_quant=rotate_quant, int8_matmul=int8_matmul, jit=jit, rotate_tile=rotate_tile)


__all__ = [
    "INT8_GEMMS",
    "KERNEL_OPTIONS",
    "ROTATE_QUANTS",
    "Int8RotLinear",
    "hadamard_factors",
    "regular_hadamard",
    "resolve_int8_gemm",
    "set_kernel_options",
]
