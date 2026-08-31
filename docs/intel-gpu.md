# Intel GPU support — analysis, roadmap, Docker

Design note for running **infr** on Intel Arc **iGPU** and **dGPU**, and for
shipping Docker images that make that path reproducible. Companion operational
docs: [`docker/README.md`](../docker/README.md), [`igpu.md`](igpu.md),
[`perf/vulkan-review.md`](perf/vulkan-review.md). **End-to-end try guide (all
three images, copy-paste commands): [`../RELEASE.md`](../RELEASE.md).**

> **Status (2026-08-31).** Phase 0/1 Docker images (`infr:cpu`, `infr:vulkan`)
> are stable. **Phase 2 SYCL Backend is now implemented as an MVP**:
> `crates/infr-sycl` initializes a real SYCL/Level Zero device through a thin
> C++ shim (`icpx` when available, degrading gracefully to `clang++`/`c++` or
> a pure host build when it isn't — see that crate's `build.rs`), and
> `--dev sycl` / `INFR_DEV=sycl` runs the full compute graph — correctness
> rides the CPU-reference interpreter end-to-end (`SyclBackend` forwards every
> `Backend` method except identity to a wrapped `CpuBackend`), with an
> oneDNN-backed `gemm_f32` primitive exposed for wiring into `Op::Linear` as
> the accelerated fast path (three-tier fallback in the shim: oneDNN → SYCL
> `parallel_for` → host loop, so results are correct at every tier).
> `docker/Dockerfile.sycl` is a real multi-stage build (not a scaffold) —
> see [`../RELEASE.md`](../RELEASE.md) for the user-facing guide. The engine
> also has Intel-oriented Vulkan paths (ANV, UMA/iGPU submit splitting,
> non-coopmat prefill tiers, opt-in 8×8×16 XMX). **TileLang is not an Intel
> runtime path** for infr today.

---

## 1. Goal

| Goal | Meaning |
| ---- | ------- |
| Run on Arc dGPU | Alchemist (A-series) and Battlemage (B-series) |
| Run on Arc iGPU | Core Ultra (MTL/ARL/LNL) and older Iris Xe where ANV works |
| Ship Docker | Reproducible images + `/dev/dri` passthrough recipes |
| Optional native | SYCL/oneAPI only if Vulkan leaves a measured gap |

Non-goals for this phase: a TileLang runtime backend; a Vulkan↔oneDNN hybrid;
changing the default AMD-tuned kernel shapes without Intel hardware numbers.

---

## 2. Findings (research summary)

### 2.1 What already exists in infr

- **Device pick:** `--dev` / `INFR_DEV` / `[device] dev` with `VulkanN` |
  `cpu` | `metal` (`crates/infr-cli`, `crates/infr-vulkan`).
- **Intel capability path:** `VENDOR_INTEL`, `DeviceArch::IntelXe1` /
  `IntelXe2`, coopmat trust rules, PCI CU table, `sg_pref=16` when
  `subgroup_min ≤ 16` (`crates/infr-vulkan/src/caps.rs`).
- **Prefill on Arc without trusted 16×16×16:** `nc_mmq` / `nc_fma` / `nc_fa`
  tiers; XMX `_cm8` behind `INFR_CM_8X8=1` / `kernels.vulkan.coopmat_8x8`.
- **iGPU/UMA:** shared-heap budget + per-submit dispatch caps (`docs/igpu.md`)
  — validated on **AMD** RADV iGPU; Intel APU is the same *class*, not yet
  measured on ANV.
- **No Docker** existed before this work; CI builds on Ubuntu 26.04 for
  `glslc` but never attaches a GPU.

### 2.2 Vulkan / Mesa (ANV) — Phase 1 path

```
infr → ash (Vulkan) → Mesa ANV ICD → xe/i915 → Arc iGPU/dGPU
```

This is the **intended** Intel path today: same `infr-vulkan` Backend as AMD
RADV. Packaging = Mesa userspace in the image + host DRM node passthrough.

Known soft spots (must re-measure on real Arc):

1. Default Alchemist coopmat is **off** (`INFR_CM_8X8`); users may sit on the
   nc_* tier until Mesa + `_cm8` is proven.
2. ANV `maxComputeWorkGroupCount[0]` is tight (backlog B65).
3. No live `DeviceProbe` from Intel hardware in CI or the validation box.

### 2.3 SYCL / oneAPI / Level Zero — Phase 2 candidate

llama.cpp’s Intel-native stack is approximately:

```
ggml-sycl → DPC++ / SYCL → Level Zero → compute-runtime → Arc
                 ↘ oneDNN (default GEMM / fused SDPA)
                 ↘ oneMKL (BLAS / optional XMX flash-attn)
```

Why it can beat Vulkan on some Intel workloads: **stable XMX / oneDNN / oneMKL
paths**, especially prompt processing. Why it is not automatic for infr:

| Factor | Implication |
| ------ | ----------- |
| New Backend | Memory, queues, dequant, FA, MoE, paging — Metal-scale work |
| Rust + SYCL | Practical shape is a **C++ SYCL/oneDNN shim** + thin Rust FFI |
| Dual stack | Vulkan↔L0 DMA-BUF hybrid is *possible* but not “thin” |
| Moving target | Community SYCL vs Vulkan wins flip with Mesa / IGC / concurrency |

**Status: implemented as an MVP** (`crates/infr-sycl`, `--dev sycl` /
`INFR_DEV=sycl`, real `infr:sycl-intel` image — see
[`../RELEASE.md`](../RELEASE.md)). As planned, it is **library-first**: a thin
C++ shim picks `icpx` when available (falling back to `clang++`/`c++`, or a
pure host build with neither SYCL nor a C++ toolchain that supports it),
initializes a real SYCL/Level Zero queue, and exposes an oneDNN-backed GEMM
primitive (`SyclBackend::gemm_f32`) with a SYCL `parallel_for` and a plain host
loop as further fallback tiers — never a hand-written kernel rewrite. Today
`SyclBackend` forwards every op except device identity to the CPU-reference
interpreter (`infr_cpu::CpuBackend`), so a `--dev sycl` run is byte-identical
to `--dev cpu` while still exercising the real device init/GEMM path; wiring
`gemm_f32` into `Op::Linear` itself is the natural next step (needs a seam in
the CPU interpreter to intercept just that op, or a thin `infr-sycl`-side
interpreter layered in front of it). No Vulkan↔oneDNN hybrid was built, per
the original decision.

### 2.4 TileLang — not a substitute

[TileLang](https://github.com/tile-ai/tilelang) is a Python/TVM **kernel DSL**.
Shipped targets: CUDA, HIP, Metal, experimental LLVM/WebGPU. **No Intel /
SYCL / SPIR-V product backend** (roadmap SPIR-V/OpenCL still open).

Realistic role for infr:

- Authoring aid on CUDA/Metal when inventing kernels to later port by hand
  into SPIR-V / MSL.
- **Not** a Docker runtime, **not** an Arc Backend, **not** a SYCL replacement.

---

## 3. Phased plan

| Phase | Deliverable | Runs inference? |
| ----- | ----------- | --------------- |
| **0** | `infr:cpu` image | Yes (CPU) |
| **1** | `infr:vulkan` image (Mesa ANV + RADV) + docs + compose | Yes (Intel/AMD Vulkan) |
| **1b** | Hardware validation matrix (below); optional `INFR_CM_8X8` default flip | Yes |
| **2** | SYCL/L0 Backend (`crates/infr-sycl`, MVP) + real `Dockerfile.sycl` | **Yes (MVP)** — device init + correctness-first execute; oneDNN `Op::Linear` wiring is the next step |
| **—** | TileLang | Out of scope until SPIR-V/Intel codegen exists |

Build helper: [`scripts/docker-build.sh`](../scripts/docker-build.sh).
Operator guide: [`docker/README.md`](../docker/README.md).

### Phase 1 run shape

```bash
./scripts/docker-build.sh vulkan

RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
docker run --rm -it \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v "$HOME/.cache/huggingface:/data/huggingface" \
  -e INFR_DEV=Vulkan0 \
  infr:vulkan devices
```

### Phase 1 validation commands (on real Arc)

```bash
# Stack
vulkaninfo --summary   # INTEL_OPEN_SOURCE_MESA / Arc name

# Device bind
infr devices
INFR_DEV=Vulkan0 infr run unsloth/Qwen3-1.7B-GGUF:Q4_K_M \
  "What is the capital of France?" --max-new 40

# Prefill A/B (XMX opt-in)
INFR_DEV=Vulkan0 infr bench "$M" -p 512 -n 0 -r 3
INFR_CM_8X8=1 INFR_DEV=Vulkan0 infr bench "$M" -p 512 -n 0 -r 3

# Optional parity (ignored in CI)
INFR_DEV=Vulkan0 cargo test -p infr-vulkan --release -- --ignored --test-threads=1
```

Pass bar: coherent generation; `sg_pref=16` on Xe1/Xe2-class; iGPU shows
UNIFIED MEMORY when appropriate; no TDR with default submit caps.

---

## 4. Hardware ask (please confirm what you can provide)

We need **at least one** discrete Arc **or** one Arc iGPU to close Phase 1b.
Ideal set:

| Device | Why we want it |
| ------ | -------------- |
| **Arc A770 / A750** (Alchemist) | Baseline discrete; XMX 8×8×16 trust; VRAM behaviour |
| **Arc B580 / B70** (Battlemage / Xe2) | Newest Mesa/IGC sensitivity; Xe2 coopmat trust path |
| **Core Ultra Arc iGPU** (MTL/ARL/LNL) | UMA / shared DDR; device selector next to a dGPU |
| **Iris Xe / 12–13th gen iGPU** (optional) | Older floor — “works but slow” |
| **AMD RDNA3/4** (optional) | Same `infr:vulkan` image RADV regression check |

**Minimum ask to you:** which of the above can you run Docker + `/dev/dri`
passthrough on? Prefer sharing:

1. `vulkaninfo --summary` (or Windows equivalent)
2. `ls -l /dev/dri`
3. Kernel version (`uname -r`) — Battlemage wants a recent `xe` stack
4. Whether the box is Linux (ANV) or Windows (Intel proprietary Vulkan)

Until that hardware is available, Phase 1 images can still be **built** and
CPU-tested; GPU claims stay provisional.

---

## 5. Docker matrix (as shipped)

| Tag | Dockerfile | Purpose |
| --- | ---------- | ------- |
| `infr:cpu` | `docker/Dockerfile.cpu` | CPU reference; CI-friendly |
| `infr:vulkan` | `docker/Dockerfile.vulkan` | Mesa Vulkan — Intel ANV + AMD RADV |
| `infr:sycl-intel` | `docker/Dockerfile.sycl` | SYCL/oneAPI/Level Zero — **real MVP inference image** |

Full user-facing walkthrough (prerequisites, exact commands, troubleshooting,
release packaging): **[`../RELEASE.md`](../RELEASE.md)**.

CPU/Vulkan build stage pins **Ubuntu 26.04** so `glslc` matches CI (24.04
shaderc is too old for dp4a shaders). Runtime Vulkan image installs
`mesa-vulkan-drivers`, `libvulkan1`, GLVND bits, and `vulkan-tools`.

SYCL image base: `intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04`
(`BASE_IMAGE` build-arg; a newer `2026.1.2-devel-ubuntu24.04` tag is confirmed
pullable but not yet hardware-validated) — the llama.cpp Intel Docker pattern.
Since that base is Ubuntu 24.04, `Dockerfile.sycl` copies a modern `glslc` in
from an Ubuntu 26.04 stage rather than switching bases (infr-vulkan stays a
build-time dependency for shader compilation even in the SYCL image). Level
Zero is pinned to **1.28.2** via upstream `.deb`s from
`github.com/oneapi-src/level-zero/releases`, installed over whatever the base
image's own PPA snapshot ships, in both the build and runtime stages.

---

## 6. Open decisions

1. **When to flip `coopmat_8x8` default on Arc** — after A770/B580 benches +
   parity, not before.
2. **`Op::Linear` → oneDNN wiring** — `SyclBackend::gemm_f32` is implemented
   and tested (`crates/infr-sycl/tests/gemm.rs`) but not yet called from the
   compute graph interpreter; needs either a CPU-interpreter override hook or
   a thin `infr-sycl`-side `Op::Linear` layer. This is what would let SYCL
   compete with Vulkan on throughput, not just correctness.
3. **CI** — optional `workflow_dispatch` Docker build (no GPU runner assumed);
   ignored Vulkan tests stay local/hardware-gated. No CI currently builds
   `--features sycl` (needs the Intel oneAPI toolchain, not just `glslc`).
4. **TileLang** — revisit only if upstream ships SPIR-V/Intel codegen; do not
   block Arc Docker or SYCL planning on it.

---

## 7. References

- Root overview: [`README.md`](../README.md)
- **Release try guide (copy-paste, all three images): [`../RELEASE.md`](../RELEASE.md)**
- iGPU campaign: [`igpu.md`](igpu.md)
- Intel Vulkan perf notes: [`perf/vulkan-review.md`](perf/vulkan-review.md)
- Config / `INFR_CM_8X8`: [`config.md`](config.md)
- llama.cpp SYCL backend docs (upstream)
- TileLang targets / SPIR-V roadmap (upstream issue #56)
