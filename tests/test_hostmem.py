import json

import pytest

from hyperflow_h3_flash import hostmem
from hyperflow_h3_flash.hostmem import available_gib, default_weight_bytes, estimate_gib, weight_bytes

GIB = 2**30
#: The safetensors sizes on disk, the text encoder trimmed to 51 of its 64 layers.
SIZES = {"text_encoder": 49.5 * GIB, "transformer": 14.9 * GIB, "vae": 9.7 * GIB, "audio_vae": 0.56 * GIB}


@pytest.mark.parametrize(
    "offload, num_gpus, height, width, measured",
    [
        ("stream", 1, 768, 1344, 90.6),
        ("model", 1, 1088, 1920, 102),
        ("none", 1, 768, 1344, 16),
        ("none", 1, 1088, 1920, 27),
        ("none", 4, 768, 1344, 4 * 16),
    ],
)
def test_estimate_matches_the_measured_peaks(offload, num_gpus, height, width, measured):
    estimate = estimate_gib(SIZES, offload=offload, num_gpus=num_gpus, height=height, width=width, num_frames=362)
    assert abs(estimate - measured) < 0.05 * measured + 1


def test_held_weights_follow_the_placement():
    def held(offload: str, num_gpus: int = 1):
        with_offload = estimate_gib(SIZES, offload=offload, num_gpus=num_gpus, height=768, width=1344, num_frames=362)
        resident = estimate_gib(SIZES, offload="none", num_gpus=num_gpus, height=768, width=1344, num_frames=362)
        return (with_offload - resident) * GIB

    vaes = SIZES["vae"] + SIZES["audio_vae"]
    assert held("stream") == pytest.approx(sum(SIZES.values()))
    # Several GPUs offloading: rank 0 alone holds the text encoder.
    assert held("stream", 4) == pytest.approx(SIZES["text_encoder"] + 4 * (SIZES["transformer"] + vaes))


def test_weights_are_measured_in_the_workflow_partition(tmp_path):
    def shard(folder, size):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"\0" * size)

    weights, base = tmp_path / "flash", tmp_path / "h3"
    shard(weights / "fl2va_int8_convrot", 100)
    shard(weights / "ref2va_int8_convrot", 200)
    shard(base / "text_encoder", 640)
    shard(base / "vae", 10)
    shard(base / "audio_vae", 1)
    assert weight_bytes(weights=str(weights), base_model=str(base), workflow="t2va") == {
        "text_encoder": 510,
        "transformer": 100,
        "vae": 10,
        "audio_vae": 1,
    }
    assert weight_bytes(weights=str(weights), base_model=str(base), workflow="ref2va")["transformer"] == 200
    assert (
        weight_bytes(weights=str(weights), subfolder="ref2va_int8_convrot", base_model=str(base), workflow="t2va")[
            "transformer"
        ]
        == 200
    )


def test_a_partial_download_is_not_measured(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 10)
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert hostmem._safetensors_bytes(tmp_path) is None
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 5)
    assert hostmem._safetensors_bytes(tmp_path) == 15


def test_default_weight_bytes_matches_known_sizes():
    bf16 = default_weight_bytes("bf16")
    fp8 = default_weight_bytes("fp8")
    assert bf16["text_encoder"] > fp8["text_encoder"]
    assert bf16["transformer"] == fp8["transformer"]


def meminfo(tmp_path, available_gib, version="Linux version 6.8.0"):
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    (proc / "meminfo").write_text(f"MemTotal: 999 kB\nMemAvailable: {int(available_gib * 2**20)} kB\n")
    (proc / "version").write_text(version)
    return proc


def limit(directory, cap_gib, used_gib, version="v2"):
    directory.mkdir(parents=True, exist_ok=True)
    cap, used = ("memory.max", "memory.current") if version == "v2" else hostmem.CGROUP_FILES["v1"][:2]
    (directory / cap).write_text(f"{int(cap_gib * GIB) if cap_gib else 'max'}\n")
    (directory / used).write_text(f"{int(used_gib * GIB)}\n")


def test_a_cgroup_limit_caps_the_available_memory(tmp_path):
    proc = meminfo(tmp_path, 100)
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    assert available_gib(proc, cgroup) == pytest.approx(100)
    (cgroup / "memory.max").write_text("max\n")
    (cgroup / "memory.current").write_text(f"{20 * GIB}\n")
    assert available_gib(proc, cgroup) == pytest.approx(100)
    (cgroup / "memory.max").write_text(f"{32 * GIB}\n")
    # Page cache is reclaimable, shmem is not: 20 GiB used, of which 6 GiB file pages minus 2 GiB shmem.
    (cgroup / "memory.stat").write_text(f"anon {14 * GIB}\nfile {6 * GIB}\nshmem {2 * GIB}\n")
    assert available_gib(proc, cgroup) == pytest.approx(32 - (20 - 4))


def test_the_limit_on_the_process_cgroup_and_its_ancestors_applies(tmp_path):
    """A systemd scope or a host slice: /proc/self/cgroup names a path below the root, and every level counts."""
    proc, cgroup = meminfo(tmp_path, 1000), tmp_path / "cgroup"
    scope = cgroup / "user.slice" / "user-1001.slice" / "run-1.scope"
    (proc / "self").mkdir()
    (proc / "self" / "cgroup").write_text("0::/user.slice/user-1001.slice/run-1.scope\n")
    limit(cgroup, None, 500)  # the root: no limit
    limit(scope, 80, 1)
    assert available_gib(proc, cgroup) == pytest.approx(79)
    limit(cgroup / "user.slice", 60, 30)  # a tighter ancestor
    assert available_gib(proc, cgroup) == pytest.approx(30)


def test_cgroup_v1_memory_controller(tmp_path):
    proc, cgroup = meminfo(tmp_path, 1000), tmp_path / "cgroup"
    (proc / "self").mkdir()
    (proc / "self" / "cgroup").write_text("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n0::/\n")
    limit(cgroup / "memory" / "docker" / "abc", 40, 10, version="v1")
    assert available_gib(proc, cgroup) == pytest.approx(30)
    (cgroup / "memory" / "docker" / "abc" / "memory.limit_in_bytes").write_text(f"{2**63 - 4096}\n")
    assert available_gib(proc, cgroup) == pytest.approx(1000)
