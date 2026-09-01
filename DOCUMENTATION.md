# HunyuanWorld 2.0 — Documentation
This document provides detailed usage guides, parameter references, and output format specifications for each component of HunyuanWorld 2.0.

## Table of Contents
- [WorldMirror 2.0 (World Reconstruction)](#worldmirror-20-world-reconstruction)
  - [Overview](#overview)
  - [Python API](#python-api)
    - [`WorldMirrorPipeline.from_pretrained`](#worldmirrorpipelinefrom_pretrained)
    - [`WorldMirrorPipeline.__call__`](#worldmirrorpipelinecall)
  - [CLI Reference](#cli-reference)
  - [Output Format](#output-format)
    - [File Structure](#file-structure)
    - [Prediction Dictionary](#prediction-dictionary)
  - [Prior Injection](#prior-injection)
    - [Camera Parameters (JSON)](#camera-parameters-json)
    - [Depth Maps (Folder)](#depth-maps-folder)
    - [Combining Priors](#combining-priors)
  - [Multi-GPU Inference](#multi-gpu-inference)
  - [Advanced Options](#advanced-options)
    - [Disabling Prediction Heads](#disabling-prediction-heads)
    - [Mask Filtering](#mask-filtering)
    - [Point Cloud Compression](#point-cloud-compression)
  - [Gradio App](#gradio-app)
- [Panorama Generation (HY-Pano 2.0)](#panorama-generation-hy-pano-20)
  - [Overview](#overview-1)
  - [Backend 1 — HunyuanImage-3](#backend-1--hunyuanimage-3)
    - [`HunyuanPanoPipeline.from_pretrained` (Backend 1)](#hunyuanpanopipelinefrom_pretrained-backend-1)
    - [`HunyuanPanoPipeline.__call__` (Backend 1)](#hunyuanpanopipelinecall-backend-1)
    - [CLI Reference (Backend 1)](#cli-reference-backend-1)
  - [Backend 2 — Qwen-Image-Edit](#backend-2--qwen-image-edit)
    - [`HunyuanPanoPipeline.from_pretrained` (Backend 2)](#hunyuanpanopipelinefrom_pretrained-backend-2)
    - [`HunyuanPanoPipeline.__call__` (Backend 2)](#hunyuanpanopipelinecall-backend-2)
    - [CLI Reference (Backend 2)](#cli-reference-backend-2)
  - [Output Format](#output-format-1)
- [World Generation](#world-generation)

---
## WorldMirror 2.0 (World Reconstruction)
### Overview
WorldMirror 2.0 is a unified feed-forward model for comprehensive 3D geometric prediction from multi-view images or video. It simultaneously generates:
- **3D point clouds** in world coordinates
- **Per-view depth maps** in camera frame
- **Surface normals** in camera coordinates
- **Camera poses** (c2w) and **intrinsics**
- **3D Gaussian Splatting** attributes (means, scales, rotations, opacities, SH coefficients)

Key improvements over WorldMirror 1.0:
- **Normalized RoPE** for flexible resolution inference
- **Depth mask prediction** for robust invalid pixel handling
- **Sequence Parallel + FSDP + BF16** for efficient multi-GPU inference

---
### Python API
#### `WorldMirrorPipeline.from_pretrained`
Factory method to load the model and create a pipeline instance.

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

pipeline = WorldMirrorPipeline.from_pretrained(
    pretrained_model_name_or_path="tencent/HY-World-2.0",
    subfolder="HY-WorldMirror-2.0",
    config_path=None,
    ckpt_path=None,
    use_fsdp=False,
    enable_bf16=False,
    fsdp_cpu_offload=False,
    disable_heads=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | `str` | `"tencent/HY-World-2.0"` | HuggingFace repo ID or local path |
| `subfolder` | `str` | `"HY-WorldMirror-2.0"` | Subfolder inside the repo containing WorldMirror checkpoint (`model.safetensors` + config) |
| `config_path` | `str` | `None` | Training config YAML (used with `ckpt_path` for custom checkpoints) |
| `ckpt_path` | `str` | `None` | Checkpoint file (`.ckpt` / `.safetensors`). When provided with `config_path`, loads model from local checkpoint instead of HuggingFace |
| `use_fsdp` | `bool` | `False` | Shard parameters across GPUs via Fully Sharded Data Parallel |
| `enable_bf16` | `bool` | `False` | Use bfloat16 precision (except numerically critical layers) |
| `fsdp_cpu_offload` | `bool` | `False` | Offload FSDP parameters to CPU (saves GPU memory at the cost of speed) |
| `disable_heads` | `list[str]` | `None` | Heads to disable and free from memory. Options: `"camera"`, `"depth"`, `"normal"`, `"points"`, `"gs"` |

**Notes:**
- Distributed mode is auto-detected from `WORLD_SIZE` environment variable (set by `torchrun`).
- When using multi-GPU, each rank must call `from_pretrained` — the method handles `dist.init_process_group` internally.

---
#### `WorldMirrorPipeline.__call__`
Run inference on a set of images or a video.

```python
result = pipeline(
    input_path,
    output_path="inference_output",
    **kwargs,
)
```

Returns the output directory path (`str`), or `None` if the input was skipped.

**Inference Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | *(required)* | Directory of images or path to a video file |
| `output_path` | `str` | `"inference_output"` | Root output directory |
| `target_size` | `int` | `952` | Maximum inference resolution (longest edge). Images are resized + center-cropped to the nearest multiple of 14 |
| `fps` | `int` | `1` | FPS for extracting frames from video input |
| `video_strategy` | `str` | `"new"` | Video frame extraction strategy: `"new"` (motion-aware) or `"old"` (uniform FPS) |
| `video_min_frames` | `int` | `1` | Minimum number of frames to extract from video |
| `video_max_frames` | `int` | `32` | Maximum number of frames to extract from video |

**Save Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_depth` | `bool` | `True` | Save per-view depth maps (PNG visualization + NPY raw values) |
| `save_normal` | `bool` | `True` | Save per-view surface normal maps (PNG) |
| `save_gs` | `bool` | `True` | Save 3D Gaussian Splatting as `gaussians.ply` |
| `save_camera` | `bool` | `True` | Save camera parameters as `camera_params.json` |
| `save_points` | `bool` | `True` | Save depth-derived point cloud as `points.ply` |
| `save_colmap` | `bool` | `False` | Save COLMAP-format sparse reconstruction (`sparse/0/`) |
| `save_conf` | `bool` | `False` | Save depth confidence maps |
| `save_sky_mask` | `bool` | `False` | Save sky segmentation masks |

**Mask Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_sky_mask` | `bool` | `True` | Filter out sky regions from point clouds and Gaussians |
| `apply_edge_mask` | `bool` | `True` | Filter out edge/discontinuity regions |
| `apply_confidence_mask` | `bool` | `False` | Filter out low-confidence predictions |
| `sky_mask_source` | `str` | `"auto"` | Sky mask method: `"auto"` (ONNX + model fusion), `"model"` (model predictions only), `"onnx"` (external segmentation only) |
| `model_sky_threshold` | `float` | `0.45` | Threshold for model-based sky detection |
| `confidence_percentile` | `float` | `10.0` | Percentile threshold for confidence filtering (bottom N% removed) |
| `edge_normal_threshold` | `float` | `1.0` | Normal edge detection tolerance |
| `edge_depth_threshold` | `float` | `0.03` | Depth edge detection relative tolerance |

**Compression Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compress_pts` | `bool` | `True` | Compress point clouds via voxel merging + random sampling |
| `compress_pts_max_points` | `int` | `2,000,000` | Maximum number of points after compression |
| `compress_pts_voxel_size` | `float` | `0.002` | Voxel size for point cloud merging |
| `max_resolution` | `int` | `1920` | Maximum resolution for saved output images |
| `compress_gs_max_points` | `int` | `5,000,000` | Maximum number of Gaussians after voxel pruning |

**Prior Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_cam_path` | `str` | `None` | Path to camera parameters JSON file |
| `prior_depth_path` | `str` | `None` | Path to directory containing depth map files |

**Rendered Video Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_rendered` | `bool` | `False` | Render interpolated fly-through video from Gaussian splats |
| `render_interp_per_pair` | `int` | `15` | Number of interpolated frames between each camera pair |
| `render_depth` | `bool` | `False` | Also render a depth visualization video |

**Misc Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_time` | `bool` | `True` | Print timing report and save `pipeline_timing.json` |
| `strict_output_path` | `str` | `None` | If set, save results directly to this path without `<case_name>/<timestamp>` subdirectories |

---
### CLI Reference
All `__call__` parameters are exposed as CLI arguments:

```bash
python -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --output_path inference_output \
    --target_size 952 \
    --prior_cam_path path/to/camera_params.json \
    --prior_depth_path path/to/depth_dir/ \
```

**Boolean flag conventions:**

| Enable | Disable |
|--------|---------|
| `--save_colmap` | *(omit)* |
| `--save_conf` | *(omit)* |
| `--save_sky_mask` | *(omit)* |
| `--apply_sky_mask` (default on) | `--no_sky_mask` |
| `--apply_edge_mask` (default on) | `--no_edge_mask` |
| `--apply_confidence_mask` | *(omit)* |
| `--compress_pts` (default on) | `--no_compress_pts` |
| `--log_time` (default on) | `--no_log_time` |
| *(default on)* `save_depth` | `--no_save_depth` |
| *(default on)* `save_normal` | `--no_save_normal` |
| *(default on)* `save_gs` | `--no_save_gs` |
| *(default on)* `save_camera` | `--no_save_camera` |
| *(default on)* `save_points` | `--no_save_points` |
| `--save_rendered` | *(omit)* |
| `--render_depth` | *(omit)* |

**Additional CLI-only arguments:**

| Argument | Description |
|----------|-------------|
| `--config_path` | Training config YAML for custom checkpoint loading |
| `--ckpt_path` | Local checkpoint file path |
| `--use_fsdp` | Enable FSDP multi-GPU sharding |
| `--enable_bf16` | Enable bfloat16 mixed precision |
| `--fsdp_cpu_offload` | Offload FSDP params to CPU |
| `--disable_heads` | Space-separated list of heads to disable (e.g. `--disable_heads camera normal`) |
| `--no_interactive` | Exit after first inference (skip interactive prompt loop) |

---
### Output Format
#### File Structure

```
inference_output/
└── <case_name>/
    └── <timestamp>/
        ├── depth/
        │   ├── depth_0000.png      # Normalized depth visualization
        │   ├── depth_0000.npy      # Raw float32 depth values [H, W]
        │   └── ...
        ├── normal/
        │   ├── normal_0000.png     # Normal map visualization (RGB)
        │   └── ...
        ├── camera_params.json      # Camera extrinsics & intrinsics
        ├── gaussians.ply           # 3D Gaussian Splatting (standard format)
        ├── points.ply              # Colored point cloud
        ├── sparse/                 # COLMAP format (if --save_colmap)
        │   └── 0/
        │       ├── cameras.bin
        │       ├── images.bin
        │       └── points3D.bin
        ├── rendered/               # Rendered video (if --save_rendered)
        │   ├── rendered_rgb.mp4
        │   └── rendered_depth.mp4  # (if --render_depth)
        └── pipeline_timing.json    # Performance timing report
```

#### Prediction Dictionary
When using the Python API, `pipeline(...)` internally produces a `predictions` dictionary with the following keys:

```python
# Geometry
predictions["depth"]        # [B, S, H, W, 1]  — Z-depth in camera frame
predictions["depth_conf"]   # [B, S, H, W]     — Depth confidence
predictions["normals"]      # [B, S, H, W, 3]  — Surface normals in camera coords
predictions["normals_conf"] # [B, S, H, W]     — Normal confidence
predictions["pts3d"]        # [B, S, H, W, 3]  — 3D point maps in world coords
predictions["pts3d_conf"]   # [B, S, H, W]     — Point cloud confidence
# Camera
predictions["camera_poses"] # [B, S, 4, 4]     — Camera-to-world (c2w), OpenCV convention
predictions["camera_intrs"] # [B, S, 3, 3]     — Camera intrinsic matrices
predictions["camera_params"]# [B, S, 9]        — Compact camera vector (translation, quaternion, fov_v, fov_u)
# 3D Gaussian Splatting
predictions["splats"]["means"]      # [B, N, 3] — Gaussian centers
predictions["splats"]["scales"]     # [B, N, 3] — Gaussian scales
predictions["splats"]["quats"]      # [B, N, 4] — Gaussian rotations (quaternions)
predictions["splats"]["opacities"]  # [B, N]    — Gaussian opacities
predictions["splats"]["sh"]         # [B, N, 1, 3] — Spherical harmonics (degree 0)
predictions["splats"]["weights"]    # [B, N]    — Per-Gaussian confidence weights
```

Where `B` = batch size (always 1 for inference), `S` = number of input views, `H, W` = image dimensions, `N` = total Gaussians (`S × H × W`).

---
### Prior Injection
WorldMirror 2.0 accepts three types of geometric priors as conditioning inputs. Priors are automatically detected from the provided files.

| Prior Type | Condition | Input Format |
|------------|-----------|--------------|
| Camera Pose | `cond_flags[0]` | c2w 4×4 matrix (OpenCV convention) |
| Depth Map | `cond_flags[1]` | Per-view float depth maps |
| Intrinsics | `cond_flags[2]` | 3×3 intrinsic matrix |

#### Camera Parameters (JSON)
The camera parameter file follows the same format as the `camera_params.json` output by the pipeline:

```json
{
  "num_cameras": 2,
  "extrinsics": [
    {
      "camera_id": 0,
      "matrix": [
        [0.98, 0.01, -0.17, 0.52],
        [-0.01, 0.99, 0.01, -0.03],
        [0.17, -0.01, 0.98, 1.20],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ],
  "intrinsics": [
    {
      "camera_id": 0,
      "matrix": [
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `camera_id` | Integer index (`0`, `1`, `2`, ...) or image filename stem without extension (e.g., `"image_0001"`) |
| `extrinsics.matrix` | 4×4 camera-to-world (c2w) transformation matrix, OpenCV coordinate convention |
| `intrinsics.matrix` | 3×3 camera intrinsic matrix in pixels (`fx, fy` = focal lengths; `cx, cy` = principal point) |

**Important notes:**
- `extrinsics` and `intrinsics` lists can be provided independently or together. An empty list `[]` or missing key means that prior is unavailable.
- **Intrinsics resolution:** Values should correspond to the **original image resolution**. The pipeline automatically adjusts for inference-time resize + center-crop.
- **Extrinsics alignment:** The pipeline automatically normalizes all extrinsics relative to the first view, consistent with training behavior.
#### Depth Maps (Folder)
Depth maps are stored as individual files in a directory. Filenames should match the input image filenames. Supported formats: `.npy`, `.exr`, `.png` (16-bit).

```
prior_depth/
├── image_0001.npy    # float32, shape [H, W]
├── image_0002.npy
└── ...
```

#### Combining Priors
Priors can be freely combined. Examples:

```bash
# Only intrinsics
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_cam_path camera_intrinsics_only.json
# Only depth
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_depth_path depth_maps/
# Camera poses + intrinsics + depth
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_cam_path camera_params.json \
    --prior_depth_path depth_maps/
```

---
### Multi-GPU Inference
WorldMirror 2.0 supports **Sequence Parallel (SP)** inference across multiple GPUs, where token sequences are sharded across ranks in the ViT backbone, and DPT heads process frames in parallel.

> **Requirement:** The number of input images must be **>= the number of GPUs** (`nproc_per_node`). For example, with 8 GPUs you need at least 8 input images. The pipeline will raise an error if this condition is not met.

```bash
# 2 GPUs with FSDP + bf16
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
# 4 GPUs
torchrun --nproc_per_node=4 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
# Python API (inside a torchrun script)
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
pipeline = WorldMirrorPipeline.from_pretrained(
    'tencent/HY-World-2.0',
    use_fsdp=True,
    enable_bf16=True,
)
pipeline('path/to/images')
```

**What happens under the hood:**
1. `from_pretrained` auto-detects `WORLD_SIZE > 1` and initializes `torch.distributed`.
2. The model is loaded on rank 0 and broadcast via `sync_module_states=True`.
3. FSDP shards parameters across the SP process group.
4. DPT prediction heads split frames across ranks and `AllGather` results.
5. Post-processing (mask computation, saving) runs on rank 0 only.

---
### Advanced Options
#### Disabling Prediction Heads
To save memory when you only need specific outputs:

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

pipeline = WorldMirrorPipeline.from_pretrained(
    'tencent/HY-World-2.0',
    disable_heads=["normal", "points"],  # free ~200M params
)
```

Available heads: `"camera"`, `"depth"`, `"normal"`, `"points"`, `"gs"`.
#### Mask Filtering
The pipeline supports three types of output filtering to improve point cloud and Gaussian quality:
1. **Sky mask** (`apply_sky_mask=True`): Removes sky regions using an ONNX-based segmentation model, optionally fused with model-predicted depth masks.
2. **Edge mask** (`apply_edge_mask=True`): Removes points at depth/normal discontinuities (object boundaries).
3. **Confidence mask** (`apply_confidence_mask=False`): Removes the bottom N% of points by prediction confidence.
These masks are applied independently to both the `points.ply` (depth-based) and `gaussians.ply` (GS-based) outputs. The GS output uses its own depth predictions for edge detection when available.
#### Point Cloud Compression
When `compress_pts=True` (default), the depth-derived point cloud undergoes:
1. **Voxel merging**: Points within each voxel (size controlled by `compress_pts_voxel_size`) are merged via weighted averaging.
2. **Random subsampling**: If the result exceeds `compress_pts_max_points`, points are uniformly subsampled.
Similarly, Gaussians are voxel-pruned (weighted averaging of means, scales, quaternions, colors, opacities) and optionally subsampled to `compress_gs_max_points`.

---
### Gradio App
An interactive web demo for WorldMirror 2.0. Upload images or videos and visualize 3DGS, point clouds, depth maps, normal maps, and camera parameters in your browser.
**Quick start:**

```bash
# Single GPU
python -m hyworld2.worldrecon.gradio_app

# Multi-GPU
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.gradio_app \
    --use_fsdp --enable_bf16
```

**With a local checkpoint:**

```bash
python -m hyworld2.worldrecon.gradio_app \
    --config_path /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint.safetensors
```

**With a public link (e.g., for Colab or remote servers):**

```bash
python -m hyworld2.worldrecon.gradio_app --share
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `8081` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--share` | `False` | Create a public Gradio link |
| `--examples_dir` | `./examples/worldrecon` | Path to example scenes directory |
| `--config_path` | `None` | Training config YAML (used with `--ckpt_path`) |
| `--ckpt_path` | `None` | Local checkpoint file (`.ckpt` / `.safetensors`) |
| `--use_fsdp` | `False` | Enable FSDP multi-GPU sharding |
| `--enable_bf16` | `False` | Enable bfloat16 mixed precision |
| `--fsdp_cpu_offload` | `False` | Offload FSDP params to CPU (saves GPU memory) |

> **Important:** In multi-GPU mode, the number of input images must be **>= the number of GPUs**.

---
## Panorama Generation (HY-Pano 2.0)
### Overview
HY-Pano 2.0 is a panorama generation model that converts a single perspective image (or a text prompt) into a 360° equirectangular panorama (ERP). It offers two backends:

- **Backend 1 — HunyuanImage-3** (`pipeline.py`): Full reasoning pipeline with chain-of-thought recaptioning. The model internally rewrites the user prompt via a think → recaption → diffuse workflow before running the diffusion process, producing higher-quality, semantically coherent panoramas.
- **Backend 2 — Qwen-Image-Edit** (`pipeline_with_qwen_image.py`): A lighter `diffusers`-based backend built on top of Qwen-Image-Edit with a LoRA adapter. Faster to load, easier to integrate into diffusers workflows.

Both backends expose the same high-level `HunyuanPanoPipeline` class with `from_pretrained` / `__call__` interfaces. Model weights are hosted at [tencent/HY-World-2.0/tree/main/HY-Pano-2.0](https://huggingface.co/tencent/HY-World-2.0/tree/main/HY-Pano-2.0).

---
### Backend 1 — HunyuanImage-3

#### `HunyuanPanoPipeline.from_pretrained` (Backend 1)
Factory method to load the HunyuanImage-3 model and return a ready-to-use pipeline.

```python
from pipeline import HunyuanPanoPipeline

pipeline = HunyuanPanoPipeline.from_pretrained(
    pretrained_model_name_or_path="tencent/HY-World-2.0",
    subfolder="HY-Pano-2.0",
    attn_impl="sdpa",
    moe_impl="eager",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | `str` | `"tencent/HY-World-2.0"` | HuggingFace repo ID or local path. Model files are expected under `{path}/{subfolder}/` or directly under `{path}/` |
| `subfolder` | `str` | `"HY-Pano-2.0"` | Subfolder inside the repo containing the model checkpoint |
| `attn_impl` | `str` | `"sdpa"` | Attention implementation. Options: `"sdpa"`, `"flash_attention_2"` (requires FlashAttention) |
| `moe_impl` | `str` | `"eager"` | MoE implementation. Options: `"eager"`, `"flashinfer"` (requires FlashInfer) |

---
#### `HunyuanPanoPipeline.__call__` (Backend 1)
Run panorama generation and return the output image.

```python
output = pipeline(
    image,
    prompt="Expand this image to a 360-degree equirectangular panorama.",
    seed=None,
    height=960,
    width=1952,
    diff_infer_steps=50,
    bot_task="think_recaption",
    use_system_prompt="en_unified",
    system_prompt=None,
    blend_width=32,
    verbose=2,
    max_new_tokens=2048,
    infer_align_image_size=False,
    # Taylor Cache
    use_taylor_cache=False,
    taylor_cache_interval=5,
    taylor_cache_order=2,
)
output.save("output_panorama.png")
```

Returns a `PIL.Image` (the generated panorama after circular edge blending).

**Core Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str` | *(required)* | Path to the input perspective image |
| `prompt` | `str` | `"Expand this image to a 360-degree equirectangular panorama."` | Text prompt. The panorama instruction is prepended automatically if not already present |
| `seed` | `int` | `None` | Random seed (`None` = random each run) |
| `height` | `int` | `960` | Output panorama height in pixels |
| `width` | `int` | `1952` | Output panorama width in pixels |
| `diff_infer_steps` | `int` | `50` | Number of diffusion denoising steps |
| `blend_width` | `int` | `32` | Pixel-space blending width for seamless left–right panorama wrapping |

**Task & Prompt Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bot_task` | `str` | `"think_recaption"` | Generation task mode. Options: `"image"` (direct generation), `"auto"` (text generation), `"recaption"` (rewrite → image), `"think_recaption"` (think → rewrite → image) |
| `use_system_prompt` | `str` | `"en_unified"` | System prompt type. Options: `"None"` (no system prompt), `"dynamic"` (determined by `bot_task`), `"en_vanilla"`, `"en_recaption"`, `"en_think_recaption"`, `"en_unified"` (recommended), `"custom"` (requires `system_prompt`) |
| `system_prompt` | `str` | `None` | Custom system prompt text. Only used when `use_system_prompt="custom"` |
| `max_new_tokens` | `int` | `2048` | Maximum number of tokens generated in the reasoning/recaption stage |
| `verbose` | `int` | `2` | Verbosity level (0 = silent, 1 = basic, 2 = full) |
| `infer_align_image_size` | `bool` | `False` | Align output resolution to the input image size |

**Taylor Cache Parameters** (speed–quality trade-off for diffusion sampling):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_taylor_cache` | `bool` | `False` | Enable Taylor Cache to accelerate sampling |
| `taylor_cache_interval` | `int` | `5` | Steps between full attention recomputation |
| `taylor_cache_order` | `int` | `2` | Polynomial approximation order |
| `taylor_cache_enable_first_enhance` | `bool` | `False` | Enable full recomputation for the first few steps |
| `taylor_cache_first_enhance_steps` | `int` | `3` | Number of first-step enhancement steps (must be > 2) |
| `taylor_cache_enable_tailing_enhance` | `bool` | `False` | Enable full recomputation for the last few steps |
| `taylor_cache_tailing_enhance_steps` | `int` | `1` | Number of tailing enhancement steps |
| `taylor_cache_low_freqs_order` | `int` | `2` | Taylor order for low-frequency attention components |
| `taylor_cache_high_freqs_order` | `int` | `2` | Taylor order for high-frequency attention components |

---
#### CLI Reference (Backend 1)

```bash
# Basic panorama generation
python pipeline.py --image input.png

# Specify prompt and output path
python pipeline.py --image input.png \
    --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \
    --save output_panorama.png

# Customize inference steps and task type
python pipeline.py --image input.png \
    --diff-infer-steps 50 --bot-task think_recaption --use-system-prompt en_unified

# Reproducible generation with a fixed seed
python pipeline.py --image input.png --seed 42 --reproduce

# Use Taylor Cache to speed up sampling
python pipeline.py --image input.png \
    --use-taylor-cache --taylor-cache-interval 5 --taylor-cache-order 2

# Load from a local path with flash attention
python pipeline.py --image input.png \
    --pretrained-model-name-or-path /path/to/HY-Pano-2.0 --subfolder "" \
    --attn-impl flash_attention_2
```

**All CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Path to the input image |
| `--prompt` | `"Expand this image..."` | Text prompt |
| `--seed` | `None` | Random seed |
| `--height` | `960` | Output height |
| `--width` | `1952` | Output width |
| `--diff-infer-steps` | `50` | Diffusion inference steps |
| `--bot-task` | `think_recaption` | Task mode (`image` / `auto` / `recaption` / `think_recaption`) |
| `--use-system-prompt` | `en_unified` | System prompt type |
| `--system-prompt` | `None` | Custom system prompt (used with `--use-system-prompt custom`) |
| `--max-new-tokens` | `2048` | Max new tokens in reasoning stage |
| `--verbose` | `2` | Verbosity level |
| `--blend-width` | `32` | Edge blending width |
| `--infer-align-image-size` | `False` | Align output size to input image |
| `--use-taylor-cache` | `False` | Enable Taylor Cache |
| `--taylor-cache-interval` | `5` | Taylor Cache update interval |
| `--taylor-cache-order` | `2` | Taylor Cache polynomial order |
| `--taylor-cache-enable-first-enhance` | `False` | Enable first-step enhancement |
| `--taylor-cache-first-enhance-steps` | `3` | First-step enhancement steps |
| `--taylor-cache-enable-tailing-enhance` | `False` | Enable tailing enhancement |
| `--taylor-cache-tailing-enhance-steps` | `1` | Tailing enhancement steps |
| `--taylor-cache-low-freqs-order` | `2` | Low-frequency Taylor order |
| `--taylor-cache-high-freqs-order` | `2` | High-frequency Taylor order |
| `--pretrained-model-name-or-path` | `tencent/HY-World-2.0` | HuggingFace repo ID or local path |
| `--subfolder` | `HY-Pano-2.0` | Subfolder containing model weights |
| `--attn-impl` | `sdpa` | Attention implementation (`sdpa` / `flash_attention_2`) |
| `--moe-impl` | `eager` | MoE implementation (`eager` / `flashinfer`) |
| `--save` | `<input_stem>_panorama.png` | Output path |
| `--reproduce` | `False` | Fix all RNGs for reproducibility |

---
### Backend 2 — Qwen-Image-Edit

#### `HunyuanPanoPipeline.from_pretrained` (Backend 2)
Factory method to load the Qwen-Image-Edit base model with a LoRA adapter.

```python
from pipeline_with_qwen_image import HunyuanPanoPipeline

pipeline = HunyuanPanoPipeline.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen-Image-Edit-2509",
    lora_path="tencent/HY-World-2.0",
    lora_subfolder="HY-Pano-2.0",
    torch_dtype=torch.bfloat16,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | `str` | `"Qwen/Qwen-Image-Edit-2509"` | HuggingFace repo ID or local path to the base Qwen-Image-Edit model |
| `lora_path` | `str` | `"tencent/HY-World-2.0"` | Local path or HuggingFace repo ID for the LoRA weights. Pass `None` to skip LoRA loading |
| `lora_subfolder` | `str` | `"HY-Pano-2.0"` | Subfolder inside `lora_path` containing `pytorch_lora_weights.safetensors` |
| `torch_dtype` | `torch.dtype` | `torch.bfloat16` | Model precision |

---
#### `HunyuanPanoPipeline.__call__` (Backend 2)
Run panorama generation and return the output image.

```python
output = pipeline(
    image,
    prompt="A sunny outdoor scene.",
    negative_prompt="",
    seed=42,
    height=960,
    width=1952,
    num_inference_steps=40,
    guidance_scale=1.0,
    blend_width=32,
    crop_border=0.0,
)
output.save("output_panorama.png")
```

Returns a `PIL.Image` (the generated panorama after circular edge blending).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str` | *(required)* | Path to the input perspective image |
| `prompt` | `str` | `""` | User-provided scene description, appended to the internal positive prompt template |
| `negative_prompt` | `str` | `""` | Additional negative prompt, appended to the internal default negative prompt |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `height` | `int` | `960` | Output panorama height in pixels |
| `width` | `int` | `1952` | Output panorama width in pixels |
| `num_inference_steps` | `int` | `40` | Number of diffusion denoising steps |
| `guidance_scale` | `float` | `1.0` | Classifier-free guidance scale |
| `blend_width` | `int` | `32` | Pixel-space edge blending width for seamless left–right panorama wrapping |
| `crop_border` | `float` | `0.0` | Fraction of image border to crop before inference (removes compression artefacts on edges) |

---
#### CLI Reference (Backend 2)

```bash
# Basic panorama generation
python pipeline_with_qwen_image.py --image input.png

# Specify prompt, seed and output path
python pipeline_with_qwen_image.py --image input.png \
    --prompt "A sunny outdoor scene." --seed 42 --save output_panorama.png

# Customize inference steps and guidance
python pipeline_with_qwen_image.py --image input.png \
    --num-inference-steps 40 --guidance-scale 1.0

# Load LoRA from a local path
python pipeline_with_qwen_image.py --image input.png \
    --lora-path /path/to/lora --lora-subfolder ""
```

**All CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Path to the input image |
| `--prompt` | `""` | Scene description prompt |
| `--negative-prompt` | `""` | Additional negative prompt |
| `--seed` | `42` | Random seed |
| `--height` | `960` | Output height |
| `--width` | `1952` | Output width |
| `--num-inference-steps` | `40` | Diffusion denoising steps |
| `--guidance-scale` | `1.0` | CFG guidance scale |
| `--blend-width` | `32` | Edge blending width |
| `--crop-border` | `0.0` | Border crop fraction |
| `--pretrained-model-name-or-path` | `Qwen/Qwen-Image-Edit-2509` | HuggingFace repo ID or local path to base model |
| `--lora-path` | `tencent/HY-World-2.0` | LoRA weights path or HuggingFace repo ID |
| `--lora-subfolder` | `HY-Pano-2.0` | Subfolder containing LoRA weights |
| `--save` | `<input_stem>_panorama.png` | Output path |
| `--reproduce` | `False` | Fix all RNGs for reproducibility |

---
### Output Format
Both backends return a single `PIL.Image` object. The panorama is in **equirectangular projection (ERP)** format with the default output resolution of **1920 × 960** pixels (after removing the blending overlap):

| Property | Value |
|----------|-------|
| Format | Equirectangular (ERP) |
| Default size | 1920 × 960 px (1952 − 32 blending overlap × 960) |
| Color space | RGB |
| File format | PNG (when saved via `.save()`) |

The circular blending step (`blend_width=32`) wraps the left and right edges to ensure seamless panorama rendering in 360° viewers.

---
## World Generation
*Coming soon.*
This section will document the world generation pipeline, including:
- Trajectory planning configuration
- World expansion with memory-driven video generation
- World composition (point cloud expansion + 3DGS optimization)
- End-to-end generation from text/image to navigable 3D world