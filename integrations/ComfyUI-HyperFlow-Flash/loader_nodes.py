"""ComfyUI nodes: load a HyperFlow GGUF and emit the 4, 6 or 8-step schedule it samples."""

from __future__ import annotations

import functools
import logging

import comfy.sd
import folder_paths
import torch

from .gguf_io import GGUFBackend
from .model_patch import ENDPOINT_PREFIX, FactorizedPlan, PackedModel, install_two_time

# Raw interval knots by step count and the video flow shift they are mapped through.
RAW_SIGMAS = {
    4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
    6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
    8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
}
DEFAULT_STEPS = 6
VIDEO_SHIFT = 12.0
WEIGHTS_IN = ("auto", "ram", "disk")


def step_sigmas(steps: int = DEFAULT_STEPS) -> torch.Tensor:
    raw = torch.tensor(RAW_SIGMAS[steps], dtype=torch.float64)
    return (VIDEO_SHIFT * raw / (1 + (VIDEO_SHIFT - 1) * raw)).float()


def gguf_files() -> list[str]:
    names = set()
    for key in ("unet_gguf", "diffusion_models", "unet"):
        if key in folder_paths.folder_names_and_paths:
            names.update(name for name in folder_paths.get_filename_list(key) if name.endswith(".gguf"))
    return sorted(names)


def gguf_path(name: str) -> str:
    for key in ("unet_gguf", "diffusion_models", "unet"):
        if key in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(key, name)
            if path is not None:
                return path
    raise FileNotFoundError(name)


@functools.cache
def disk_patcher(base: type) -> type:
    """A GGUFModelPatcher whose offloaded weights stay in the memory-mapped file.

    When the transformer does not fit on the GPU, the stock patcher copies every weight into RAM ("release mmap") and
    pins what it offloads, so 9 GB of memory the OS cannot reclaim. Here the weights that do not fit stay in the file's
    page cache and stream to the GPU layer by layer (ComfyUI's lowvram path); under memory pressure the OS drops those
    pages and reads them again from disk. What fits on the GPU loads as usual.
    """

    class HyperFlowDiskPatcher(base):
        mmap_released = True

        def pin_weight_to_device(self, key):
            pass

        def clone(self, *args, **kwargs):
            clone = base.clone(self, *args, **kwargs)
            clone.__class__ = HyperFlowDiskPatcher
            # GGUFModelPatcher.clone copies the source's flag into the instance, which would hide the class's.
            clone.mmap_released = True
            return clone

    return HyperFlowDiskPatcher


def supports_disk(base: type) -> bool:
    """`disk_patcher` overrides these ComfyUI-GGUF internals; an older or newer GGUFModelPatcher may lack them."""
    return hasattr(base, "mmap_released") and hasattr(base, "pin_weight_to_device")


def weights_on_disk(path: str, weights_in: str) -> bool:
    if weights_in != "auto":
        return weights_in == "disk"
    try:
        import comfy.storage
    except ImportError:
        return False
    model_fast_disk = getattr(comfy.storage, "model_fast_disk", None)
    return model_fast_disk is not None and model_fast_disk([path])


def load_hyperflow_gguf(path: str, weights_in: str = "auto"):
    backend = GGUFBackend.find()
    state_dict = backend.read(path)
    gate = state_dict.pop("hyperflow_gate", None)
    sigmas = state_dict.pop("hyperflow_sigmas", None)
    endpoint_sd = {
        name.removeprefix(ENDPOINT_PREFIX): state_dict.pop(name)
        for name in list(state_dict)
        if name.startswith(ENDPOINT_PREFIX)
    }
    plan = FactorizedPlan(state_dict, backend)
    ops = backend.operations()
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options={"custom_operations": ops}, disable_dynamic=True
    )
    if patcher is None:
        raise RuntimeError(f"Could not detect the model type of {path}")
    model = patcher.model.diffusion_model
    if not isinstance(model, PackedModel):
        raise RuntimeError(f"{path} is not a packed video+audio DiT checkpoint")
    factorized = plan.install(model, ops)
    if gate is not None:
        install_two_time(model, endpoint_sd, float(backend.to_float(gate).flatten()[0]), ops)
    model.hyperflow_sigmas = step_sigmas() if sigmas is None else backend.to_float(sigmas).flatten()
    logging.info(
        "HyperFlow: %d factorized blocks, two-time=%s, %d-step schedule",
        factorized,
        gate is not None,
        model.hyperflow_sigmas.shape[0] - 1,
    )
    base = backend.nodes.GGUFModelPatcher
    on_disk = weights_on_disk(path, weights_in)
    if on_disk and not supports_disk(base):
        logging.warning("HyperFlow: this ComfyUI-GGUF cannot keep weights in the file; falling back to ram")
        on_disk = False
    patcher_class = disk_patcher(base) if on_disk else base
    patcher = patcher_class.clone(patcher)
    patcher.size = 0
    logging.info("HyperFlow: transformer weights in %s (weights_in=%s)", "disk" if on_disk else "ram", weights_in)
    return patcher


class HyperFlowGGUFLoader:
    TITLE = "HyperFlow GGUF Loader"
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"unet_name": (gguf_files(),)},
            "optional": {
                "weights_in": (
                    list(WEIGHTS_IN),
                    {
                        "default": "auto",
                        "tooltip": "disk: weights that do not fit on the GPU stay in the memory-mapped file and "
                        "stream from it, for machines with little RAM; ram: copy them into RAM; auto: disk with "
                        "--fast-disk or a fast NVMe drive",
                    },
                ),
            },
        }

    def load(self, unet_name, weights_in="auto"):
        return (load_hyperflow_gguf(gguf_path(unet_name), weights_in),)


class HyperFlowSigmas:
    TITLE = "HyperFlow Sigmas"
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    DESCRIPTION = (
        "Step schedule of the checkpoint: 4 steps favour speed, 6 (default) balance speed and quality, 8 favour "
        "quality. Use it with SamplerCustom, euler, cfg 1."
    )

    @classmethod
    def INPUT_TYPES(cls):
        steps = ("INT", {"default": DEFAULT_STEPS, "min": 4, "max": 8, "step": 2})
        return {"required": {"model": ("MODEL",)}, "optional": {"steps": steps}}

    def sigmas(self, model, steps=DEFAULT_STEPS):
        steps = int(steps)
        if steps not in RAW_SIGMAS:
            raise ValueError(f"HyperFlow samples {sorted(RAW_SIGMAS)} steps, got {steps}")
        schedule = getattr(model.model.diffusion_model, "hyperflow_sigmas", None)
        if steps != DEFAULT_STEPS or schedule is None:
            schedule = step_sigmas(steps)
        return (schedule.clone().cpu(),)


NODE_CLASS_MAPPINGS = {"HyperFlowGGUFLoader": HyperFlowGGUFLoader, "HyperFlowSigmas": HyperFlowSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {name: node.TITLE for name, node in NODE_CLASS_MAPPINGS.items()}
