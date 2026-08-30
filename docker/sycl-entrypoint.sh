#!/bin/sh
# Phase-2 scaffold entrypoint — does NOT run inference.
# A future infr SYCL/oneAPI Backend will replace this stub.
set -eu
cat <<'EOF'
infr: SYCL Backend is not implemented yet.

This image (infr:sycl-intel) is a Phase-2 Docker scaffold for a future Intel
SYCL / oneAPI Backend. It ships Level Zero / compute-runtime plumbing only —
it does not load models or run inference.

Status and design: docs/intel-gpu.md
Until that Backend exists, use the Vulkan image (infr:vulkan) on Intel Arc
(ANV) or the CPU image (infr:cpu).
EOF
exit 1
