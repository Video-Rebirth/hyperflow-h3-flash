# ComfyUI-HyperFlow-Flash

ComfyUI custom nodes and drag-and-drop workflows for [HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) GGUF checkpoints (`fl2va_gguf/` and `ref2va_gguf/`):

- **HyperFlow GGUF Loader**: loads pruned, low-rank factored, two-time `(t, r)` GGUF checkpoints (`weights_in: auto | disk | ram`).
- **HyperFlow Sigmas**: emits the 4-, 6- (default), or 8-step sigma schedule at `CFG = 1.0`.
- **Workflow templates** (`example_workflows/`): `hyperflow_t2v`, `hyperflow_i2v`, `hyperflow_flf2v`, and `hyperflow_ref2v` (automatically listed under **Workflow > Browse Templates > ComfyUI-HyperFlow-Flash**).

## Install

Requires a [ComfyUI](https://github.com/Comfy-Org/ComfyUI) build with native MiniMax-H3 support (`0.43.0` or newer) and [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF):

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash
pip install -r ComfyUI-HyperFlow-Flash/requirements.txt  # in ComfyUI's Python environment
```

## Model Files

Place the downloaded files in `ComfyUI/models/`:

```text
ComfyUI/
├── main.py
├── custom_nodes/
│   ├── ComfyUI-GGUF/
│   └── ComfyUI-HyperFlow-Flash/
└── models/
    ├── diffusion_models/
    │   ├── minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf   # from VideoRebirthAI/hyperflow-h3-flash (fl2va_gguf/)
    │   └── minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf  # optional, for ref2v (ref2va_gguf/)
    ├── text_encoders/
    │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       # from Comfy-Org/MiniMax-H3 (15.7 GB)
    └── vae/
        ├── minimax_h3_video_vae_fp16.safetensors              # from Comfy-Org/MiniMax-H3
        └── minimax_h3_audio_vae_fp32.safetensors              # from Comfy-Org/MiniMax-H3
```

From `ComfyUI/`, launch with `python main.py --fast-disk` on PCs with 16 GB or 32 GB of system RAM. For complete setup instructions, benchmarks, and troubleshooting, see [`integrations/README.md`](https://github.com/Video-Rebirth/hyperflow-h3-flash/blob/main/integrations/README.md).

## License

- **Code:** GPL-3.0 (see [`LICENSE`](LICENSE)).
- **Model weights:** Not included. The GGUF checkpoints are a Model Derivative of MiniMax-H3 under the [MiniMax H3 Community License Agreement](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash/blob/main/LICENSE), not GPL-3.0. The Agreement does not cover use in the European Union, the United Kingdom, the Republic of Korea, or the United States without MiniMax's authorization, and it includes an Acceptable Use Policy.
