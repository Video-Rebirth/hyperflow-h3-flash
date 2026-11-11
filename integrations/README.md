# GGUF Runtimes

Both HyperFlow H3 Flash transformers are published as GGUF checkpoints in [`VideoRebirthAI/hyperflow-h3-flash`](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash), the recommended format for consumer GPUs and PCs with 16–64 GB of system RAM (measured on 24 GB and 32 GB GPUs; 16 GB GPUs are not measured yet):

- **`fl2va_gguf/`**: text to video (`t2v`), first frame to video (`i2v`), and first-and-last frame to video (`flf2v`).
- **`ref2va_gguf/`**: reference images to video (`ref2v`, up to 9 images). Video and audio references require the Python package (`hyperflow-h3-flash --reference ...`).

ComfyUI also loads the INT8 convrot transformers as single safetensors files (`fl2va_int8_convrot_comfyui/`, `ref2va_int8_convrot_comfyui/`) on its own INT8 kernels, without ComfyUI-GGUF. In ComfyUI on 24 GB+ GPUs they are faster than `Q4_K_M` and closer to full precision, also on 16 GB PCs; see [INT8 safetensors](#comfyui-int8).

| Runtime | Contents | License |
|---|---|---|
| [ComfyUI](#comfyui) | [`ComfyUI-HyperFlow-Flash/`](ComfyUI-HyperFlow-Flash) ([`Video-Rebirth/ComfyUI-HyperFlow-Flash`](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash)): custom nodes and eight drag-and-drop workflows (four GGUF, four INT8) | GPL-3.0 |
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

Install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (GGUF files only) and [ComfyUI-HyperFlow-Flash](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash) in `ComfyUI/custom_nodes/`:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF  # not needed for the INT8 safetensors
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

#### INT8 safetensors instead of a GGUF

<a id="comfyui-int8"></a>

`fl2va_int8_convrot_comfyui/` (16.0 GB) and `ref2va_int8_convrot_comfyui/` (16.4 GB) hold the INT8 convrot transformers of `fl2va_int8_convrot/` and `ref2va_int8_convrot/` in ComfyUI's tensor layout. Their INT8 linears are ComfyUI `int8_tensorwise` + convrot layers with the published INT8 codes and scales; the low-rank linears stay bf16. **HyperFlow Loader** reads them; ComfyUI-GGUF is not needed:

```sh
hf download VideoRebirthAI/hyperflow-h3-flash \
  fl2va_int8_convrot_comfyui/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors --local-dir hf-download
mv hf-download/fl2va_int8_convrot_comfyui/*.safetensors models/diffusion_models/
# Optional: ref2va_int8_convrot_comfyui/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors
```

- ComfyUI runs the INT8 layers on comfy-kitchen: its CUDA backend with a PyTorch built for CUDA 13.0 or newer, its Triton backend with `--enable-triton-backend`, and a PyTorch fallback otherwise, about 50% slower.
- **HyperFlow Loader** runs the INT8 layers of long clips in row chunks: comfy-kitchen's Triton INT8 kernels use 32-bit offsets and return wrong values once a layer's output passes 2^31 elements (`fl2va` at 294 frames, `1344 × 768`). Other INT8 loaders for this model need the same limit.
- ComfyUI 0.44.0 ran the file with 12 GiB of system RAM left to it (a 16 GB PC) on the RTX 5090 and 4090, without `--fast-disk`.

<a id="int8-performance"></a>

Sampling time on one H200 with `--enable-triton-backend`, `1344 × 768`, 6 steps; 294 frames for `fl2va`, 192 frames for `ref2v`:

| Transformer | `t2v` | `i2v` | `flf2v` | `ref2v` |
|---|---:|---:|---:|---:|
| INT8 safetensors | 198 s | 159 s | 217 s | 109 s |
| `Q8_0` GGUF | 192 s | 170 s | 231 s | 109 s |
| `BF16` GGUF | 194 s | 170 s | 231 s | 108 s |

The INT8 file runs as fast as the GGUFs on this GPU; the PyTorch fallback took 298 s for the same `t2v`.

End-to-end `t2v` time without saving the MP4 (`5.2 s / 15 s`; ComfyUI 0.44.0, PyTorch 2.13 + CUDA 13.0, `nvfp4_awq` text encoder, one RunPod pod per GPU):

| GPU, system RAM | INT8 safetensors | `Q4_K_M` GGUF | `Q8_0` GGUF |
|---|---:|---:|---:|
| RTX PRO 6000 (96 GB), 188 GB | 53 s / 333 s | – | – |
| RTX 5090 (32 GB), 60 GB | 83 s / 518 s | 96 s / 562 s | 98 s / – |
| RTX 5090, 16 GB PC¹ | 88 s / – | 102 s / – | – |
| RTX 4090 (24 GB), 31 GB | 152 s / 747 s | 160 s / 834 s² | 173 s / – |
| RTX 4090, 16 GB PC¹ | 153 s / 821 s | 163 s / – | – |

- **¹ 16 GB PC:** 12 GiB of system RAM left to ComfyUI; GGUF with `--fast-disk` (and `--disable-cuda-malloc` on the 4090).
- **² 15 s with `--disable-cuda-malloc`:** with the default CUDA allocator, this run crashed.
- **Peak VRAM, INT8, 5.2 s:** 39 GiB on the PRO 6000, 31 GiB on the 5090, 21 GiB on the 4090. Triton and CUDA backends ran at the same speed.
- **Quality:** against the `BF16` GGUF (one H200, same seed, `t2v` 5.2 s), mean frame PSNR is 18.0 dB for INT8, 17.5 dB for `Q8_0`, and 14.8 dB for `Q4_K_M`.

### 4. Start ComfyUI and Open a Workflow

From `ComfyUI/`, run `python main.py` (add `--fast-disk` on 16 GB or 32 GB RAM machines). In the UI, open **Workflow > Browse Templates > ComfyUI-HyperFlow-Flash**, or drag a workflow file from [`ComfyUI-HyperFlow-Flash/example_workflows/`](ComfyUI-HyperFlow-Flash/example_workflows) onto the canvas:

| Workflow file | Task | Required transformer | Inputs |
|---|---|---|---|
| `hyperflow_t2v.json` | Text to video + audio | `fl2va` | Prompt |
| `hyperflow_i2v.json` | First frame to video + audio | `fl2va` | Prompt, first frame |
| `hyperflow_flf2v.json` | First + last frame to video + audio | `fl2va` | Prompt, first frame, last frame |
| `hyperflow_ref2v.json` | Reference images to video + audio | `ref2va` | Prompt with `<Picture 1>`, `<Picture 2>`, ..., reference images (up to 9) |

- The workflows load a `Q4_K_M` GGUF; `hyperflow_t2v_int8.json`, `hyperflow_i2v_int8.json`, `hyperflow_flf2v_int8.json`, and `hyperflow_ref2v_int8.json` are the same graphs with the INT8 safetensors. **HyperFlow Loader** takes either file, so you can also switch the file in any of them.
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

With `--fast-disk`, ComfyUI and **HyperFlow Loader** (`weights_in = auto`, GGUF files) stream offloaded weights directly from the OS file page cache rather than copying them into non-reclaimable pinned RAM:

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
- **`weights_in` override on `HyperFlow Loader` (and `HyperFlow GGUF Loader`), for GGUF files:** `auto` (default) follows ComfyUI's storage policy (`fast_disk=True`), `disk` forces memory-mapped file streaming, and `ram` forces standard RAM pinning.
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
| **HyperFlow** nodes missing after restart | Run `pip install -r ComfyUI-HyperFlow-Flash/requirements.txt` in ComfyUI's Python environment and check the ComfyUI log for the import error. |
| "needs the ComfyUI-GGUF custom node" | Install `ComfyUI-GGUF` in `custom_nodes/` to load GGUF files, or use the INT8 safetensors. |
| `MiniMaxH3ImageToVideo` node missing | Update ComfyUI to 0.43.0 or newer. |
| Loader dropdown is empty or red | Place `.gguf` and HyperFlow INT8 `.safetensors` files in `ComfyUI/models/diffusion_models/`, `.safetensors` text encoders in `models/text_encoders/`, and VAEs in `models/vae/`, then restart ComfyUI. **HyperFlow Loader** lists only safetensors files with HyperFlow metadata. |
| "uses HyperFlow layout N; this ComfyUI-HyperFlow-Flash reads up to M" | Update ComfyUI-HyperFlow-Flash. |
| INT8 safetensors much slower than a GGUF | Use a PyTorch built for CUDA 13.0 or newer, or start ComfyUI with `--enable-triton-backend`. |
| Dark, grid-patterned, or NaN frames with INT8 on long clips | Update ComfyUI-HyperFlow-Flash to 1.1.0 or newer and load the file with **HyperFlow Loader**. |
| Out of system RAM on 16 GB / 32 GB PC | Start ComfyUI with `--fast-disk` and use a GGUF with `weights_in` set to `auto` or `disk` on **HyperFlow Loader**. |
| Out of GPU memory | Use `Q4_K_M` or `Q3_K_M`, keep the `nvfp4` text encoder, or reduce `length` / resolution. |
| `CUDA error: invalid argument` in `ref2v` with `ref_image_size` `max` | Start ComfyUI with `--disable-cuda-malloc`. |
| Stock UNet, GGUF, or safetensors loader fails to load | Use **HyperFlow Loader**; stock loaders do not support the two-time embedder or factorized linear layers. |

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
