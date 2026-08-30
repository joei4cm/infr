# Docker images for infr

Phase 1 ships a **CPU** image and a **Vulkan/Mesa** image (Intel Arc ANV + AMD
RADV). A Phase-2 **SYCL/Intel** Dockerfile exists as a scaffold only — it does
not run inference until a Rust SYCL Backend lands (see `docs/intel-gpu.md`).

Build context is the repo root. Prefer the helper script:

```bash
./scripts/docker-build.sh cpu        # → infr:cpu
./scripts/docker-build.sh vulkan     # → infr:vulkan
./scripts/docker-build.sh sycl       # → infr:sycl-intel (scaffold)
./scripts/docker-build.sh all
```

Or call Docker directly:

```bash
docker build -f docker/Dockerfile.cpu    -t infr:cpu .
docker build -f docker/Dockerfile.vulkan -t infr:vulkan .
docker build -f docker/Dockerfile.sycl   -t infr:sycl-intel .
```

Build uses **Ubuntu 26.04** so `glslc` is new enough (Ubuntu 24.04's shaderc is
too old for the dp4a shaders — same pin as CI).

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
| `INFR_DEV`   | Device pick: `cpu`, `Vulkan0`, `Vulkan1`, … (same as `--dev`) |
| `HF_HOME`    | HuggingFace home; hub cache is `$HF_HOME/hub` |
| `HF_HUB_CACHE` | Override hub cache path directly |
| `HF_TOKEN`   | Auth for gated/private HF repos |
| `RUST_LOG`   | Optional tracing filter |

## Troubleshooting

**No Vulkan devices / empty `infr devices`**
- Confirm `/dev/dri/renderD*` exists on the host and is passed with `--device /dev/dri`.
- Check ICDs inside the container: `ls /usr/share/vulkan/icd.d/` and `vulkaninfo` (image includes `vulkan-tools`).
- Wrong group: add the render GID (`stat -c '%g' /dev/dri/renderD128`) via `--group-add`. Permission denied on the render node usually means this.

**Wrong GPU when several are present**
- Run `infr devices`, then set `INFR_DEV=VulkanN` (or `--dev VulkanN`) for the ANV/RADV device you want.

**`glslc` / build failures**
- Do not switch the build stage to Ubuntu 24.04 — its shaderc is too old. Stay on 26.04.

## SYCL scaffold (`infr:sycl-intel`)

```bash
docker run --rm infr:sycl-intel
```

Prints that the SYCL Backend is not implemented and exits non-zero. Do not use
this image for inference; use `infr:vulkan` on Intel Arc until a Backend exists.
