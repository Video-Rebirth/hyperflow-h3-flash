# GGUF Runtimes

Both HyperFlow H3 Flash transformers are published as GGUF checkpoints in [`VideoRebirthAI/hyperflow-h3-flash`](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash), the recommended format for consumer GPUs and PCs with 16–64 GB of system RAM (measured on 24 GB and 32 GB GPUs; 16 GB GPUs are not measured yet):

- **`fl2va_gguf/`**: text to video (`t2v`), first frame to video (`i2v`), and first-and-last frame to video (`flf2v`).
- **`ref2va_gguf/`**: reference images to video (`ref2v`, up to 9 images). Video and audio references require the Python package (`hyperflow-h3-flash --reference ...`).

| Runtime | Contents | License |
|---|---|---|
| [ComfyUI](#comfyui) | [`ComfyUI-HyperFlow-Flash/`](ComfyUI-HyperFlow-Flash) ([`Video-Rebirth/ComfyUI-HyperFlow-Flash`](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash)): custom nodes and four drag-and-drop workflows | GPL-3.0 |
| [stable-diffusion.cpp](#stable-diffusioncpp) | [`stable-diffusion.cpp/`](stable-diffusion.cpp): patch for the `sd-cli` command-line tool | MIT |
| [Unsloth Studio](#unsloth-studio) | Uses the patched `stable-diffusion.cpp` binary | — |

All runtimes sample in 4, 6 (default), or 8 steps without classifier-free guidance (`CFG = 1`):

| Steps | Mode | ComfyUI | stable-diffusion.cpp | Unsloth Studio |
|---:|---|---|---|---|
| `4` | Speed | **HyperFlow Sigmas** (`steps = 4`) | `--steps 4 --sigmas ...` (see below) | — |
| `6` | Balanced (default) | **HyperFlow Sigmas** (`steps = 6`) | Default (stored in GGUF) | Default (always 6) |
| `8` | Quality | **HyperFlow Sigmas** (`steps = 8`) | `--steps 8 --sigmas ...` (see below) | — |

---

## ComfyUI

### 1. Update ComfyUI

Use a [ComfyUI](https://github.com/Comfy-Org/ComfyUI) version with native MiniMax-H3 support (0.43.0 or newer, which includes the `MiniMaxH3ImageToVideo` node). On PCs with 16 GB or 32 GB of system RAM, use a recent ComfyUI build that supports `--fast-disk` (see [Host RAM](#host-ram)).

### 2. Install Custom Nodes

Install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and [ComfyUI-HyperFlow-Flash](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash) in `ComfyUI/custom_nodes/`:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash
pip install -r ComfyUI-HyperFlow-Flash/requirements.txt  # in the Python environment ComfyUI uses
```

### 3. Download the Model Files

Pick one GGUF quant per task (`fl2va_gguf/` for text and keyframes, `ref2va_gguf/` for reference images):

| Quant | `fl2va` size | `ref2va` size | When to use |
|---|---:|---:|---|
| **`Q4_K_M`** | **9.1 GB** | **9.2 GB** | **Recommended default for 24 GB GPUs** |
| `Q8_0` | 14.9 GB | 15.2 GB | Closest to full precision when VRAM/RAM fits |
| `Q6_K` / `Q5_K_M` / `Q4_K_S` | 8.0–11.6 GB | 8.1–11.8 GB | Intermediate size/quality trade-offs |
| `Q3_K_M` | 7.2 GB | 7.4 GB | Smallest footprint |

From your `ComfyUI` directory, download the GGUF transformer plus the companion text encoder and VAEs from [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3):

```sh
# 1. HyperFlow GGUF transformer (replace Q4_K_M with your chosen quant)
hf download VideoRebirthAI/hyperflow-h3-flash \
  fl2va_gguf/minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf --local-dir hf-download
mv hf-download/fl2va_gguf/*.gguf models/diffusion_models/
# Optional: download ref2va_gguf/minimax_h3_ref2va_hyperflow_flash_14b-Q4_K_M.gguf for reference-to-video

# 2. Companion text encoder (15.7 GB NVFP4) and VAEs
hf download Comfy-Org/MiniMax-H3 \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir models
```

Your `ComfyUI/` directory should look like:

```text
ComfyUI/
├── main.py
├── custom_nodes/
│   ├── ComfyUI-GGUF/
│   └── ComfyUI-HyperFlow-Flash/
└── models/
    ├── diffusion_models/minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf
    ├── text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    └── vae/
        ├── minimax_h3_video_vae_fp16.safetensors
        └── minimax_h3_audio_vae_fp32.safetensors
```

> `Comfy-Org/MiniMax-H3` also provides `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27.1 GB) and `qwen3vl_32b_minimax_h3_bf16.safetensors` (51.5 GB) if you prefer a larger text encoder.

### 4. Start ComfyUI and Open a Workflow

From `ComfyUI/`, run `python main.py` (add `--fast-disk` on 16 GB or 32 GB RAM machines). In the UI, open **Workflow > Browse Templates > ComfyUI-HyperFlow-Flash**, or drag a workflow file from [`ComfyUI-HyperFlow-Flash/example_workflows/`](ComfyUI-HyperFlow-Flash/example_workflows) onto the canvas:

| Workflow file | Task | Required GGUF | Inputs |
|---|---|---|---|
| `hyperflow_t2v.json` | Text to video + audio | `fl2va` | Prompt |
| `hyperflow_i2v.json` | First frame to video + audio | `fl2va` | Prompt, first frame |
| `hyperflow_flf2v.json` | First + last frame to video + audio | `fl2va` | Prompt, first frame, last frame |
| `hyperflow_ref2v.json` | Reference images to video + audio | `ref2va` | Prompt with `<Picture 1>`, `<Picture 2>`, ..., reference images (up to 9) |

- Every MP4 saved by these workflows (`ComfyUI/output/hyperflow/`) embeds the workflow metadata; drag the MP4 back into ComfyUI to reload the graph.
- API-format JSON files for headless scripting are in `example_workflows/api/`:

```sh
cd ComfyUI/custom_nodes/ComfyUI-HyperFlow-Flash/example_workflows/api
curl -s http://127.0.0.1:8188/prompt -H "Content-Type: application/json" \
  -d "{\"prompt\": $(cat hyperflow_t2v_api.json)}"
```

### 5. Key Settings

Adjust your inputs in the **MiniMaxH3ImageToVideo** (or **MiniMaxH3ReferenceToVideo**) node:

- **`prompt`**: describe the scene, motion, camera movement, and sound in a single prompt (no negative prompt).
- **`width` / `height`**: multiples of 32 (e.g., `1344 × 768` landscape or `768 × 1152` portrait).
- **`length`**: frame count at 24 fps from `124` (5.2 s) to `362` (15 s), using `17n + 5` frames.
- **`steps`** (in **HyperFlow Sigmas**): `4` (speed), `6` (default), or `8` (quality).
- Keep **`KSamplerSelect`** set to `euler` and **`BasicGuider`** at `cfg = 1.0`.

### 6. Host RAM and `--fast-disk`

<a id="host-ram"></a>

On PCs with 16 GB or 32 GB of system RAM, launch ComfyUI with `--fast-disk`:

```sh
python main.py --fast-disk
# Windows portable build (in run_nvidia_gpu.bat):
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --fast-disk
```

With `--fast-disk`, ComfyUI and **HyperFlow GGUF Loader** (`weights_in = auto`) stream offloaded weights directly from the OS file page cache rather than copying them into non-reclaimable pinned RAM:

| System RAM | Launch command | Behavior |
|---|---|---|
| **≥ 64 GB** | `python main.py` | Models stay cached in RAM (34–39 GiB peak RSS). Fastest across repeated runs. |
| **32 GB** | `python main.py --fast-disk` | Anonymous RSS stays at 8–16 GiB; within ~5% of uncapped RAM on RTX 3090 / 4090 (~13% on RTX 5090). |
| **16 GB** | `python main.py --fast-disk` | Anonymous RSS stays around ~8 GiB; OS page cache uses the remaining memory. |

`ref2v` with limited system RAM (`Q4_K_M`, `ref_image_size` `match`, 2 images, `1344 × 768`, 6 steps). Each cell is total time for a **5.2 s / 15 s** clip:

| GPU | ≥ 64 GB RAM | 32 GB PC (default) | 32 GB PC (`--fast-disk`) | 16 GB PC (`--fast-disk`) |
|---|---|---|---|---|
| **RTX 5090 (32 GB)** | 135 s / 660 s | 164 s / 687 s | 139 s / 663 s | 157 s / 684 s |
| **RTX 4090 (24 GB)** | 194 s / 885 s | 191 s / 891 s | 180 s / 866 s | 178 s / 876 s |
| **RTX 3090 (24 GB)** | 401 s / 2059 s | 395 s / 2036 s | 385 s / 2010 s | 384 s / 2025 s |
| *Peak anonymous RSS* | *25–34 GiB* | *13–17 GiB* | *10–13 GiB* | *~8 GiB* |

- **Keep models on an NVMe SSD** when using `--fast-disk`, since uncached layers are re-read from disk between stages.
- **`weights_in` override on `HyperFlow GGUF Loader`:** `auto` (default) follows ComfyUI's storage policy (`fast_disk=True`), `disk` forces memory-mapped file streaming, and `ram` forces standard RAM pinning.
- **Check a Windows PC in Task Manager** (Performance > Memory) during a 15 s clip. **In use** should stay below the installed RAM. **Cached** holds the model files and may fill the rest. If **Committed** grows past the installed RAM, Windows pages to disk and every step slows down. Close other programs or use a shorter `length`.

### 7. Performance on Consumer RTX GPUs

<a id="what-to-expect"></a>

Measured with `Q4_K_M` GGUF + `qwen3vl_32b_minimax_h3_nvfp4_awq` at `1344 × 768`, 6 steps, with `--fast-disk` on a 32 GB RAM configuration:

| GPU | Frames (Duration) | `t2v` | `i2v` | `flf2v` | `ref2v` (2 images) | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| **RTX 4090 (24 GB)** | 124 (5.2 s) | 152 s | 166 s | 178 s | 180 s | 23 GiB |
| **RTX 5090 (32 GB)** | 124 (5.2 s) | 99 s | 108 s | — | 139 s | 28 GiB |
| **RTX 5090 (32 GB)** | 294 (12.2 s) | 426 s | 439 s | — | — | 23 GiB |
| **RTX 5090 (32 GB)** | 362 (15.0 s) | 609 s | 624 s | — | 663 s | 23 GiB |

Peak VRAM is for `t2v` and `i2v`; `ref2v` peaks at 30 GiB on the RTX 5090 and 22 GiB on the RTX 4090. These runs used a different machine from the [main README `t2va` table](../README.md#rtx-gpus-in-a-home-pc-32-gb-or-16-gb-of-host-ram-q4_k_m-comfyui), so timings differ (RTX 5090, 5.2 s `t2v`: 99 s here, 118 s there).

Every reference image adds its own tokens to every step. `ref_image_size` `match` (the default) downscales each reference, keeping its aspect ratio, to the output's pixel area. `max` downscales only above a 2048 short edge; on 2048 images this more than doubled the time of a 5.2 s clip. See the [`ref2va` benchmarks](../README.md#ref2va-on-consumer-rtx-gpus) for all three GPUs and the INT8 comparison, and [Host RAM](#host-ram) for 16 GB / 32 GB PCs.

### 8. Troubleshooting

| Symptom | Fix |
|---|---|
| **HyperFlow** nodes missing after restart | Verify `ComfyUI-GGUF` is installed in `custom_nodes/` and `pip install gguf` was run in ComfyUI's Python environment. |
| `MiniMaxH3ImageToVideo` node missing | Update ComfyUI to 0.43.0 or newer. |
| Loader dropdown is empty or red | Place `.gguf` files in `ComfyUI/models/diffusion_models/`, `.safetensors` text encoders in `models/text_encoders/`, and VAEs in `models/vae/`, then restart ComfyUI. |
| Out of system RAM on 16 GB / 32 GB PC | Start ComfyUI with `--fast-disk` or set `weights_in` to `disk` on **HyperFlow GGUF Loader**. |
| Out of GPU memory | Use `Q4_K_M` or `Q3_K_M`, keep the `nvfp4` text encoder, or reduce `length` / resolution. |
| `CUDA error: invalid argument` in `ref2v` with `ref_image_size` `max` | Start ComfyUI with `--disable-cuda-malloc`. |
| Stock UNet or GGUF loader fails to load | Use **HyperFlow GGUF Loader**; stock loaders do not support the two-time embedder or factorized linear layers. |

---

## stable-diffusion.cpp

Apply `stable-diffusion.cpp/hyperflow-two-time.patch` to upstream commit [`3be2378`](https://github.com/leejet/stable-diffusion.cpp/commit/3be23788fc0c9f35a2c24b1211caae11d4a54dbc) and build with CUDA enabled:

```sh
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git checkout 3be23788fc0c9f35a2c24b1211caae11d4a54dbc
git submodule update --init --recursive
git apply /path/to/hyperflow-h3-flash/integrations/stable-diffusion.cpp/hyperflow-two-time.patch
cmake -B build -DSD_CUDA=ON
cmake --build build --config Release -j
```

The patch adds support for the two-time `(t, r)` embedder, low-rank linear layers, the default 6-step schedule stored in `hyperflow_sigmas`, and token-axis chunking on wide projections to avoid 32-bit CUDA matmul offset overflow on long clips. Standard checkpoints continue to use the unmodified code path.

### Running `sd-cli`

Download a GGUF transformer (`fl2va_gguf/` or `ref2va_gguf/`), the companion VAEs from [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3), and either the BF16 text encoder (`qwen3vl_32b_minimax_h3_bf16.safetensors`) or a GGUF text encoder from [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF):

```sh
./build/bin/sd-cli -M vid_gen \
  --diffusion-model minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf \
  --llm qwen3vl_32b_minimax_h3_bf16.safetensors \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --sampling-method euler --cfg-scale 1.0 --steps 6 \
  -W 1344 -H 768 --video-frames 124 --fps 24 -s 42 \
  -p "A chef slides a steaming bowl of ramen across a wooden counter; rain taps on the window." \
  -o output.webm
```

- **4 or 8 steps:** pass the pre-shifted video sigmas explicitly:
  ```sh
  # 4 steps
  --steps 4 --sigmas 1.0,0.962575793,0.755781889,0.312497854,0.0

  # 8 steps
  --steps 8 --sigmas 1.0,0.993909776,0.9842875,0.966063738,0.923076928,0.8349424,0.696852028,0.468753308,0.0
  ```
  When `--sigmas` is omitted, `sd-cli` automatically uses the 6-step schedule stored in the GGUF (`1.0,0.982655942,0.914642572,0.674755692,0.385223746,0.159055322,0.0`).
- **Keyframes (`fl2va`):** add `-i first.png` for first frame, or `-i first.png --end-img last.png` for first and last frame.
- **Reference images (`ref2va`):** use a `minimax_h3_ref2va_...` GGUF and pass `--ref-image img1.png --ref-image img2.png` in the order referenced by `<Picture 1>`, `<Picture 2>`, ... (up to 9).
- **VRAM offloading & MP4 conversion:** add `--offload-to-cpu` on smaller GPUs. Convert `.webm` output to `.mp4` with `ffmpeg -i output.webm -c:v libx264 -crf 18 -c:a aac output.mp4`.

---

## Unsloth Studio

Point Unsloth Studio at the patched `sd-cli` binary from [stable-diffusion.cpp](#stable-diffusioncpp) before launching:

```sh
export SD_CLI_PATH=/path/to/stable-diffusion.cpp/build/bin/sd-cli
# or: export UNSLOTH_SD_CPP_PATH=/path/to/stable-diffusion.cpp
unsloth studio
```

Load `VideoRebirthAI/hyperflow-h3-flash` in Studio and select a `minimax_h3_fl2va_hyperflow_flash_14b-*.gguf` file from `fl2va_gguf/` (or `ref2va_gguf/`). Studio downloads the companion text encoder and VAEs from [`unsloth/MiniMax-H3-GGUF`](https://huggingface.co/unsloth/MiniMax-H3-GGUF) automatically. Keep `CFG = 1.0` and `flow_shift = 12`; the patched binary automatically runs the 6-step schedule stored in the GGUF.
