#!/usr/bin/env bash
# Build infr Docker images.
#
# Usage: ./scripts/docker-build.sh [cpu|vulkan|sycl|all] [extra docker build args...]
#
# Tags:
#   infr:cpu
#   infr:vulkan
#   infr:sycl-intel
#
# Examples:
#   ./scripts/docker-build.sh cpu
#   ./scripts/docker-build.sh vulkan --build-arg FOO=bar
#   ./scripts/docker-build.sh all --no-cache
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-all}"
if [[ $# -ge 1 ]]; then
  shift
fi
EXTRA=("$@")

build_one() {
  local name="$1"
  local file="$2"
  local tag="$3"
  echo "==> building ${tag} from ${file}"
  docker build -f "$file" -t "$tag" "${EXTRA[@]}" "$ROOT"
}

case "$TARGET" in
  cpu)
    build_one cpu docker/Dockerfile.cpu infr:cpu
    ;;
  vulkan)
    build_one vulkan docker/Dockerfile.vulkan infr:vulkan
    ;;
  sycl)
    build_one sycl docker/Dockerfile.sycl infr:sycl-intel
    ;;
  all)
    build_one cpu docker/Dockerfile.cpu infr:cpu
    build_one vulkan docker/Dockerfile.vulkan infr:vulkan
    build_one sycl docker/Dockerfile.sycl infr:sycl-intel
    ;;
  -h|--help|help)
    echo "Usage: $0 [cpu|vulkan|sycl|all] [extra docker build args...]" >&2
    exit 0
    ;;
  *)
    echo "unknown target: ${TARGET} (expected cpu|vulkan|sycl|all)" >&2
    echo "Usage: $0 [cpu|vulkan|sycl|all] [extra docker build args...]" >&2
    exit 1
    ;;
esac

echo "==> done"
