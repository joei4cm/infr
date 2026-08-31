# Docker images for infr

Three images: **CPU** (reference backend, no GPU needed), **Vulkan/Mesa**
(Intel Arc ANV + AMD RADV — the recommended first GPU try), and **SYCL/oneAPI**
(Intel native, Level Zero — MVP: real SYCL device init + correctness-first
execute, oneDNN GEMM wired in as the fast path lands; see `docs/intel-gpu.md`).

**New to this?** Start with [`../RELEASE.md`](../RELEASE.md) at the repo root
— a copy-paste, prerequisites-first guide for trying a release on real Intel
Arc hardware. This file documents the Docker image internals; RELEASE.md
documents the end-to-end user path.

Build context is the repo root. Prefer the helper script:

```bash
./scripts/docker-build.sh cpu        # → infr:cpu
./scripts/docker-build.sh vulkan     # → infr:vulkan
./scripts/docker-build.sh sycl       # → infr:sycl-intel
./scripts/docker-build.sh all
```

Or call Docker directly:

```bash
docker build -f docker/Dockerfile.cpu    -t infr:cpu .
docker build -f docker/Dockerfile.vulkan -t infr:vulkan .
docker build -f docker/Dockerfile.sycl   -t infr:sycl-intel .
```

Package a full release (build + `docker save` + checksums) with
`../scripts/docker-release.sh` — see [`../RELEASE.md`](../RELEASE.md) §10.

`Dockerfile.cpu` / `Dockerfile.vulkan` build on **Ubuntu 26.04** so `glslc` is
new enough (Ubuntu 24.04's shaderc is too old for the dp4a shaders — same pin
as CI). `Dockerfile.sycl` builds on Intel's `deep-learning-essentials` image
(Ubuntu 24.04) instead — for shaders, it copies a modern `glslc` in from an
Ubuntu 26.04 stage rather than switching the whole base (see that Dockerfile's
header comment and [`../RELEASE.md`](../RELEASE.md) §9).

## Run — CPU

```bash
docker run --rm -it \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  -e INFR_DEV=cpu \
  infr:cpu run unsloth/Qwen3-0.6B-GGUF:Q4_K_M "hello"
```

`HF_HOME` defaults to `/data/huggingface` in the image. Mount the host HF cache
there so pulls from `infr`, `llama.cpp`, and `huggingface_hub` stay shared.
Gated repos need `-e HF_TOKEN=…`.

## Run — Vulkan (Intel Arc / AMD)

Pass the host DRM devices and the render group:

```bash
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)

docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  -e INFR_DEV=Vulkan0 \
  infr:vulkan devices
```

Compose equivalent:

```bash
docker compose -f docker/compose.vulkan.yml run --rm infr devices
```

- **Intel Arc / iGPU** → Mesa **ANV**
- **AMD** → Mesa **RADV**

Without `/dev/dri` (or on a host with no GPU), Mesa may still expose
**llvmpipe** (CPU software Vulkan). That is useful for packaging smoke tests,
not for real throughput — pin the real device with `infr devices` /
`INFR_DEV=VulkanN` once DRI is passed through.

NVIDIA is not the focus of these Mesa images (use a host ICD / different stack
if you need it).

## Environment

| Variable     | Role |
| ------------ | ---- |
| `INFR_DEV`   | Device pick: `cpu`, `Vulkan0`, `Vulkan1`, `sycl`, … (same as `--dev`) |
| `HF_HOME`    | HuggingFace home; hub cache is `$HF_HOME/hub` |
| `HF_HUB_CACHE` | Override hub cache path directly |
| `HF_TOKEN`   | Auth for gated/private HF repos |
| `RUST_LOG`   | Optional tracing filter |
| `ONEAPI_DEVICE_SELECTOR` | SYCL image only — device filter, default `level_zero:gpu` |
| `ZES_ENABLE_SYSMAN` | SYCL image only — Level Zero Sysman, set to `1` |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | SYCL image only — larger Level Zero allocations, set to `1` |

## Troubleshooting

**No Vulkan devices / empty `infr devices`**
- Confirm `/dev/dri/renderD*` exists on the host and is passed with `--device /dev/dri`.
- Check ICDs inside the container: `ls /usr/share/vulkan/icd.d/` and `vulkaninfo` (image includes `vulkan-tools`).
- Wrong group: add the render GID (`stat -c '%g' /dev/dri/renderD128`) via `--group-add`. Permission denied on the render node usually means this.

**Wrong GPU when several are present**
- Run `infr devices`, then set `INFR_DEV=VulkanN` (or `--dev VulkanN`) for the ANV/RADV device you want.

**`glslc` / build failures**
- Do not switch the build stage to Ubuntu 24.04 — its shaderc is too old. Stay on 26.04.

## Run — SYCL / oneAPI (Intel native)

`infr:sycl-intel` uses Intel's oneAPI/SYCL/Level Zero stack instead of Vulkan.
**Status: MVP** — a real SYCL device is initialized and the compute graph
executes correctly (CPU-reference interpreter under the hood for
correctness-first execution, oneDNN Linear GEMM wired in when the toolchain
supports it — see `docs/intel-gpu.md`). Not yet a performance path; try
`infr:vulkan` first on Arc unless you specifically want the native oneAPI
stack.

Same DRM node pattern as Vulkan — Level Zero also needs the render node:

```bash
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)

docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  infr:sycl-intel devices
```

`devices` only enumerates Vulkan today; to see which SYCL/Level Zero device
was actually selected, run the CLI directly and check the startup log line
(`INFR_DEV` defaults to `sycl` in this image):

```bash
docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  infr:sycl-intel run unsloth/Qwen3-0.6B-GGUF:Q4_K_M "hello"
```

Look for `[sycl backend] device: <name> (Level Zero GPU)` in the startup log.
If it instead reports a CPU/host fallback, see the SYCL entry in
[`../RELEASE.md`](../RELEASE.md) §8 Troubleshooting.

Compose equivalent:

```bash
docker compose -f docker/compose.sycl.yml run --rm infr devices
```

Build pins (base image, Level Zero version, how `glslc` gets into an
Ubuntu-24.04-based image) are documented at the top of `Dockerfile.sycl` and
in [`../RELEASE.md`](../RELEASE.md) §9.
