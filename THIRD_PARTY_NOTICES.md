# Third-party notices

Except for `integrations/` (see [GGUF runtimes](#gguf-runtimes-integrations)), the code in this repository is licensed
under the Apache License, Version 2.0 (see `LICENSE`). Parts of it are derived from other Apache-2.0 projects; their
notices are reproduced here as the license requires.

## diffusers (Hugging Face)

https://github.com/huggingface/diffusers/tree/v0.40.0 — Apache License 2.0.

### MiniMax-H3 Modular Pipeline

- `src/hyperflow_h3_flash/blocks.py`: `FlashSetTimestepsStep` and `FlashLoopDenoiser` subclass and rework
  `MiniMaxH3SetTimestepsStep` and `MiniMaxH3LoopDenoiser` (`src/diffusers/modular_pipelines/minimax_h3/`).
- `src/hyperflow_h3_flash/schedule.py`: `build_row_time_pairs` is derived from
  `MiniMaxH3SetTimestepsStep.build_row_timesteps`
  (`src/diffusers/modular_pipelines/minimax_h3/before_denoise.py`).

```
Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### MiniMax-H3 Scheduler

- `src/hyperflow_h3_flash/schedule.py`: `shift_sigmas` is derived from `MiniMaxH3Scheduler.set_timesteps`
  (`src/diffusers/schedulers/scheduling_minimax_h3.py`).

```
Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### MiniMax-H3 Transformer

- `src/hyperflow_h3_flash/transformer.py`: `HyperFlowH3FlashTransformer3DModel.forward` is derived from
  `MiniMaxH3Transformer3DModel.forward` (`src/diffusers/models/transformers/transformer_minimax_h3.py`).

```
Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## MiniMax-H3

The transformer weights (fl2va and ref2va partitions and their GGUF quants) are a derivative of MiniMax-H3 and
are distributed on their own Hugging Face repository, https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash,
under the MiniMax H3 Community License Agreement, not under this repository's code license. Agreement text:
https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash/blob/main/LICENSE (the copy shipped with the weights)
and https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE (upstream).

## GGUF runtimes (`integrations/`)

These are not under this repository's Apache-2.0 license:

- `integrations/stable-diffusion.cpp/hyperflow-two-time.patch` modifies stable-diffusion.cpp
  (https://github.com/leejet/stable-diffusion.cpp), MIT License, Copyright (c) 2023 leejet
  (`integrations/stable-diffusion.cpp/LICENSE`).
- `integrations/ComfyUI-HyperFlow-Flash/` builds on ComfyUI (https://github.com/Comfy-Org/ComfyUI) and
  ComfyUI-GGUF (https://github.com/city96/ComfyUI-GGUF) and is licensed under the GNU General Public License v3.0
  (`integrations/ComfyUI-HyperFlow-Flash/LICENSE`).
