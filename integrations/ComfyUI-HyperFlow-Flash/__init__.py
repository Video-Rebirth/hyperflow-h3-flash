"""ComfyUI custom node package for HyperFlow GGUF checkpoints; needs ComfyUI-GGUF installed next to it."""

from .loader_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
