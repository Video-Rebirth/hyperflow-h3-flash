import pytest
import torch
from torch import nn

from hyperflow_h3_flash.int8 import Int8RotLinear
from hyperflow_h3_flash.offload import apply_offload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading moves weights to a GPU")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 64, dtype=torch.bfloat16)
        self.int8 = Int8RotLinear(64, 64, group=16)
        self.int8.qweight.random_(-127, 128)

    def forward(self, x):
        return x + self.int8(self.dense(x))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(Block() for _ in range(3))
        self.proj_out = nn.Linear(64, 64, dtype=torch.bfloat16)

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return self.proj_out(x)


def components():
    torch.manual_seed(0)
    return {"text_encoder": nn.Linear(64, 64, dtype=torch.bfloat16), "transformer": Tiny()}


def run(parts):
    x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    return parts["transformer"](parts["text_encoder"](x))


def devices(module):
    return {t.device.type for t in [*module.parameters(), *module.buffers()]}


@pytest.mark.parametrize("mode", ["model", "stream"])
def test_offload_matches_resident_and_returns_weights_to_pinned_host_memory(mode):
    resident = components()
    apply_offload(resident, "cuda", "none")
    torch.manual_seed(1)
    expected = run(resident)

    offloaded = components()
    apply_offload(offloaded, "cuda", mode)
    for _ in range(2):
        torch.manual_seed(1)
        torch.testing.assert_close(run(offloaded), expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        assert devices(offloaded["text_encoder"]) == {"cpu"}  # evicted when the transformer ran
        if mode == "stream":
            assert devices(offloaded["transformer"]) == {"cpu"}
    # diffusers moves the parameters outside the streamed blocks with a plain `.to`, so those come back unpinned.
    host = [*offloaded["text_encoder"].parameters()]
    if mode == "stream":
        host += offloaded["transformer"].transformer_blocks.parameters()
    assert all(t.is_pinned() for t in host)
