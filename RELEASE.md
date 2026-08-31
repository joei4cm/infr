# infr — Release try guide (Intel Arc)

This is a **copy-paste guide** for trying an infr release on a machine with an
Intel Arc GPU (integrated **iGPU**, e.g. Core Ultra, or discrete **dGPU**,
e.g. Arc A- / B-series). It is written for people who are new to Docker and to
infr — every command below is meant to be pasted as-is into a terminal on
**Linux**.

If a term below is unfamiliar: **iGPU** = the GPU built into your CPU chip
(no separate graphics card). **dGPU** = a separate graphics card (its own
PCIe slot). **`/dev/dri`** = the Linux device files a GPU driver exposes;
Docker needs to be told to pass them into the container. **DRM render node**
= the specific `/dev/dri/renderD1XX` file compute workloads use.

---

## 1. What this release includes

Three Docker images, one per backend, all built from the same source tree:

| Image tag        | Dockerfile                | Backend                       | Status |
| ----------------- | -------------------------- | ------------------------------ | ------ |
| `infr:cpu`         | `docker/Dockerfile.cpu`     | CPU reference (no GPU needed)  | Stable |
| `infr:vulkan`      | `docker/Dockerfile.vulkan`  | Vulkan / Mesa (Intel ANV, AMD RADV) | Stable — **recommended first try on Arc** |
| `infr:sycl-intel`  | `docker/Dockerfile.sycl`    | SYCL / oneAPI / Level Zero (Intel native) | **MVP** — real SYCL device init + correctness-first execute (see §5 and [`docs/intel-gpu.md`](docs/intel-gpu.md)) |

All three run the exact same `infr` CLI (`pull`, `run`, `serve`, `bench`,
`devices`, …) — only the compute backend selected by `--dev` / `INFR_DEV`
differs. See the root [`README.md`](README.md) for what infr does and which
model families it supports.

---

## 2. Prerequisites

- **Linux** with a recent kernel (Battlemage/Xe2 dGPUs want a recent `xe`
  kernel driver; iGPUs generally work on whatever ships with your distro).
- **Docker** installed (`docker --version`). No GPU-specific Docker runtime is
  required — infr talks to the GPU through ordinary device files, not
  `nvidia-docker`-style plugins.
- **`/dev/dri`** present on the host:

  ```bash
  ls -l /dev/dri
  ```

  You should see at least one `renderD1XX` entry (e.g. `renderD128`). If this
  directory is empty or missing, your GPU driver is not loaded — install/update
  your Intel graphics driver first (this is a host, not a Docker, problem).
- An **Intel Arc iGPU or dGPU**. AMD Vulkan (RADV) also works with
  `infr:vulkan`, but this guide focuses on Intel.
- Enough disk space to pull/build images (a few GB each) and to hold whatever
  model you download (models are **not** bundled in the image — see §4).

You do **not** need Rust, `cargo`, or a oneAPI install on your machine to
*run* these images — only to build them from source (§4, Option A).

---

## 3. Get the images

### Option A — Build from source

```bash
git clone https://github.com/kryptic-sh/infr.git
cd infr

./scripts/docker-build.sh cpu       # -> infr:cpu
./scripts/docker-build.sh vulkan    # -> infr:vulkan
./scripts/docker-build.sh sycl      # -> infr:sycl-intel
# or build all three:
./scripts/docker-build.sh all
```

Build time is dominated by compiling the Rust workspace in release mode
(several minutes on a modern machine); `infr:sycl-intel` additionally pulls a
multi-GB `intel/deep-learning-essentials` base image the first time.

### Option B — Load from a release tarball

If you received `dist/*.tar.gz` files (produced by
`scripts/docker-release.sh`, see §10) or downloaded them from the GitHub
Release page instead of a git checkout:

```bash
# CPU + Vulkan (single files):
gunzip -c infr-cpu-0.1.0.tar.gz        | docker load
gunzip -c infr-vulkan-0.1.0.tar.gz     | docker load

# SYCL is split into two parts (GitHub Release asset limit is 2 GiB):
cat infr-sycl-intel-0.1.0.tar.gz.part-* > infr-sycl-intel-0.1.0.tar.gz
gunzip -c infr-sycl-intel-0.1.0.tar.gz | docker load

# Verify (compare against the SHA256SUMS file that ships alongside):
sha256sum -c SHA256SUMS
```

`docker load` registers the images under the tags baked into the tarball
(`infr:cpu`, `infr:vulkan`, `infr:sycl-intel`, plus a `-<VERSION>` tag — see
§10). Confirm they are present:

```bash
docker images | grep infr
```

---

## 4. Path A — Vulkan/Mesa (recommended first try)

This is the safest first path on Intel Arc: Mesa's Vulkan driver (**ANV**) is
mature and the same code path infr already validates on AMD (**RADV**).

```bash
# Find the render node group so the container can open it as non-root-safe:
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)

docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  -e INFR_DEV=Vulkan0 \
  infr:vulkan devices
```

Expected output: one or more lines starting `Vulkan0: ...` naming your Intel
GPU. If you see more than one Vulkan device (e.g. an iGPU *and* a dGPU in the
same box), pick the one you want with `INFR_DEV=VulkanN`.

Now run a small model end-to-end:

```bash
docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  -e INFR_DEV=Vulkan0 \
  infr:vulkan run unsloth/Qwen3-0.6B-GGUF:Q4_K_M "What is the capital of France?"
```

The first run downloads the model into your HuggingFace cache (the `-v`
mount above), so subsequent runs are fast and reused by other tools
(llama.cpp, `huggingface_hub`) that share the same cache layout.

Compose equivalent:

```bash
docker compose -f docker/compose.vulkan.yml run --rm infr devices
```

---

## 5. Path B — SYCL/oneAPI (native Intel)

`infr:sycl-intel` uses Intel's own oneAPI/SYCL/Level Zero stack instead of
Vulkan. **Current status is MVP** (see [`docs/intel-gpu.md`](docs/intel-gpu.md)):
a real SYCL device is initialized (you will see a genuine Intel GPU name, not
a placeholder) and the compute graph executes correctly — but execution
currently rides the CPU-reference interpreter for correctness first, with an
oneDNN-accelerated GEMM primitive wired in as the fast path lands. Practically:
**expect this path to be correct but not yet faster than Vulkan** — try it for
a second opinion / compatibility check, not (yet) for peak throughput.

```bash
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)

docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  infr:sycl-intel devices
```

The `sycl` backend does not have its own `devices` listing (there is exactly
one Level Zero selection today, chosen automatically); instead run the CLI
directly and check the startup log line, which names the real device:

```bash
docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  infr:sycl-intel run unsloth/Qwen3-0.6B-GGUF:Q4_K_M "What is the capital of France?"
```

Look for a startup line like:

```
[sycl backend] device: Intel(R) Arc(TM) ... Graphics (Level Zero GPU)
```

If it instead says `SYCL CPU device / host fallback`, Level Zero did not find
a GPU inside the container — double check `--device /dev/dri` and
`--group-add` (§7 Troubleshooting). If it says `cpu fallback (built without a
SYCL toolchain)`, the image was built without a working `icpx`/oneDNN — this
should not happen with the shipped `Dockerfile.sycl` on the pinned base image,
but would happen if you rebuilt with a different `BASE_IMAGE` that lacks them.

Compose equivalent:

```bash
docker compose -f docker/compose.sycl.yml run --rm infr devices
```

---

## 6. Path C — CPU baseline

No GPU, no `/dev/dri`, works anywhere Docker runs. Useful as a correctness
baseline or on a machine without a supported GPU driver yet.

```bash
docker run --rm -it \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  -e INFR_DEV=cpu \
  infr:cpu run unsloth/Qwen3-0.6B-GGUF:Q4_K_M "What is the capital of France?"
```

This will be noticeably slower than the GPU paths — that's expected, it's the
reference/fallback backend, not a performance path.

---

## 7. Verify checklist

Run through these in order; each should succeed before moving to the next.

1. **Host GPU node exists**: `ls -l /dev/dri` shows at least one `renderD1XX`.
2. **Image built/loaded**: `docker images | grep infr` shows the tag you need.
3. **Container sees the device**: `infr:vulkan devices` (§4) lists your GPU —
   not "no Vulkan physical devices found".
4. **A small model runs and produces coherent text** (§4/§5/§6) — not
   garbage tokens, not a crash.
5. **(SYCL only)** the startup log names a real GPU, not a "fallback" string
   (§5).
6. **(Optional) benchmark**: `infr bench` reports plausible (non-zero,
   non-error) tokens/sec:

   ```bash
   docker run --rm -it --device /dev/dri --group-add "$RENDER_GID" \
     -v "$HOME/.cache/huggingface:/data/huggingface" -e INFR_DEV=Vulkan0 \
     infr:vulkan bench unsloth/Qwen3-0.6B-GGUF:Q4_K_M -p 512 -n 128
   ```

If every step above passes, the release is working correctly on your machine.

---

## 8. Troubleshooting

**`docker: Error response from daemon: ... /dev/dri: no such file or directory`**
The host has no GPU device nodes — your GPU driver isn't loaded. This is a
host/driver problem, not something Docker or infr can fix. Confirm with
`ls /dev/dri` on the bare host (outside any container).

**`no Vulkan physical devices found`**
- Confirm `--device /dev/dri` is on the `docker run` command — it is easy to
  forget.
- Check ICDs inside the container: `docker run --rm infr:vulkan sh -c 'ls
  /usr/share/vulkan/icd.d/'` should list a Mesa ICD.
- Permission denied on the render node almost always means the missing
  `--group-add "$RENDER_GID"` step — re-run `stat -c '%g' /dev/dri/renderD128`
  and pass that number.

**SYCL: `infr_sycl_init failed` or falls back to a CPU/host device**
- Same `--device /dev/dri` / `--group-add` checklist as Vulkan above — Level
  Zero needs the same render node.
- Confirm the base image's own compute-runtime is intact:
  `docker run --rm infr:sycl-intel sh -c 'ls /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so*'`
- If you rebuilt with `--build-arg BASE_IMAGE=...`, make sure that base still
  ships `icpx` + oneDNN (`intel/deep-learning-essentials` images do; a plain
  Ubuntu base will not).

**Wrong GPU when several are present (Vulkan)**
Run `infr devices` first, then set `INFR_DEV=VulkanN` (or `--dev VulkanN`) to
the one you want.

**Build fails on `glslc` / shader compile errors**
Do not build the CPU/Vulkan images on anything but the pinned Ubuntu 26.04
build stage — Ubuntu 24.04's `shaderc` package is too old for the dp4a
shaders infr ships. `Dockerfile.sycl` works around the same issue by copying a
modern `glslc` in from an Ubuntu 26.04 stage (see §9).

**`infr-cli` build fails with `does not contain this feature: sycl`**
The `sycl` Cargo feature comes from `crates/infr-sycl` — if that crate is
missing or not yet wired into `infr-cli`/`infr-llama` on the checkout you are
building, `Dockerfile.sycl`'s final `cargo build` step will fail with exactly
this error. Use `infr:cpu` / `infr:vulkan` in the meantime; the SYCL Backend
is tracked in [`docs/intel-gpu.md`](docs/intel-gpu.md).

**Model download is slow / stalls**
Models come from HuggingFace over resumable HTTP Range requests — re-running
the same `infr run`/`infr pull` command resumes rather than restarting. Gated
repos need `-e HF_TOKEN=<your token>`.

**Out of memory / model too big**
Pick a smaller quant with the `:quant` suffix, e.g.
`unsloth/Qwen3-1.7B-GGUF:Q4_K_M` instead of a larger model, or a
smaller model entirely. See the root [`README.md`](README.md#supported-models)
for validated model × quant combinations.

---

## 9. Image pins / versions

| Component | Pin | Where |
| --------- | --- | ----- |
| Vulkan/CPU build stage | `ubuntu:26.04` | `docker/Dockerfile.cpu`, `docker/Dockerfile.vulkan` (shaderc 2025+, needed for dp4a shaders) |
| SYCL base image | `intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04` | `docker/Dockerfile.sycl` (`BASE_IMAGE` build-arg; override to bump, e.g. `2026.1.2-devel-ubuntu24.04`, confirmed pullable but not yet hardware-validated) |
| SYCL build glslc source | `ubuntu:26.04` (binary + `libshaderc.so.1` copied in) | `docker/Dockerfile.sycl` |
| Level Zero | **1.28.2**, pinned `.deb`s from `github.com/oneapi-src/level-zero/releases` | `docker/Dockerfile.sycl` (`LEVEL_ZERO_VERSION` build-arg) — installed over whatever the base image's own PPA snapshot ships |
| GPU compute-runtime (`libze-intel-gpu1`, `intel-opencl-icd`, `libigc2`) | whatever the DLE base image ships | already present in `intel/deep-learning-essentials`, not separately pinned |
| Rust toolchain | `stable` via `rustup`, resolved at build time | all three Dockerfiles |

Rebuild with a different base image or Level Zero pin:

```bash
docker build -f docker/Dockerfile.sycl -t infr:sycl-intel \
  --build-arg BASE_IMAGE=intel/deep-learning-essentials:2026.1.2-devel-ubuntu24.04 \
  --build-arg LEVEL_ZERO_VERSION=1.28.2 \
  .
```

---

## 10. How we built this release

Building and packaging every image for a release is one command:

```bash
VERSION=0.1.0 ./scripts/docker-release.sh
```

This:

1. Builds `infr:cpu`, `infr:vulkan`, `infr:sycl-intel` via
   [`scripts/docker-build.sh`](scripts/docker-build.sh) (the same script used
   in §3 Option A).
2. Tags each image additionally as `infr:<backend>-<VERSION>`.
3. Runs `docker save <image> | gzip` into `dist/infr-<backend>-<VERSION>.tar.gz`.
4. Writes `dist/SHA256SUMS` covering every tarball.

To only build/package a subset:

```bash
./scripts/docker-release.sh cpu vulkan      # skip sycl
./scripts/docker-release.sh --skip-sycl     # equivalent
```

`dist/` is not committed to git (release artifacts, not source) — see
`.gitignore`. Distribute the `dist/*.tar.gz` files plus `dist/SHA256SUMS`
together; §3 Option B is the corresponding load-and-verify recipe.

### v0.1.0 artifact checksums

| File | Size | SHA256 |
| ---- | ---- | ------ |
| `infr-cpu-0.1.0.tar.gz` | 57 MB | `2fb468e06ce23b67ca866b2c47652b56d371a52085fbb4ddada7d631274a9450` |
| `infr-vulkan-0.1.0.tar.gz` | 149 MB | `e56389a3e6c9cc736122f7df2b87a943f0cb78cc4e0a39419805c39c8b966fa9` |
| `infr-sycl-intel-0.1.0.tar.gz` | 2.9 GB | `06996010cd68c0299f24d325fca0d8c259d40a4ca8da4c21e86a399fc234b186` |
| `infr-sycl-intel-0.1.0.tar.gz.part-0` | 1.5 GB | `1dc15633a1a7ced3ae1785930d7dedc8268913288966b0ac55465cfa57d1b843` |
| `infr-sycl-intel-0.1.0.tar.gz.part-1` | 1.4 GB | `caba3ab35ad64cfa2d395c29ba4c96a068119fc45c7b5c417764228266422cb6` |

GitHub Release assets ship the SYCL image as **two parts** (2 GiB upload
limit). Reassemble with `cat …part-* > infr-sycl-intel-0.1.0.tar.gz` before
`docker load` (see §3 Option B).

The SYCL image is large because it embeds the Intel
`deep-learning-essentials` oneAPI runtime. Prefer `infr:vulkan` for everyday
use; use `infr:sycl-intel` when you specifically want the Level Zero path.

---

For engine internals, model support, and performance numbers (not
Docker-specific), see the root [`README.md`](README.md). For the Intel
Arc/SYCL design rationale and current implementation status, see
[`docs/intel-gpu.md`](docs/intel-gpu.md). For the Docker image internals
(what each Dockerfile stage does and why), see
[`docker/README.md`](docker/README.md).
