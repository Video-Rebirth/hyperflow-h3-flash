# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""HyperFlow H3 Flash: a 13.7B, 4- to 8-step MiniMax-H3 for joint video and audio generation.

from hyperflow_h3_flash import load_pipeline

pipe = load_pipeline("t2va")
"""

from .blocks import WORKFLOWS, flash_blocks
from .pipeline import BASE_MODEL, WEIGHTS, load_pipeline
from .transformer import HyperFlowH3FlashTransformer3DModel

__version__ = "1.0.0"

__all__ = [
    "BASE_MODEL",
    "WEIGHTS",
    "WORKFLOWS",
    "HyperFlowH3FlashTransformer3DModel",
    "__version__",
    "flash_blocks",
    "load_pipeline",
]
