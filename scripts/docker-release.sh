#!/usr/bin/env bash
# Build all three infr Docker images, tag them with a release VERSION, save
# each to a gzip'd tarball under dist/, and write dist/SHA256SUMS.
#
# This is the "how we built this release" step referenced from RELEASE.md —
# it is a thin wrapper around scripts/docker-build.sh (build) + `docker save`
# (package), so a user who only wants to build+run locally can still just use
# scripts/docker-build.sh directly and skip this script entirely.
#
# Usage:
#   VERSION=0.1.0 ./scripts/docker-release.sh
#   ./scripts/docker-release.sh                 # VERSION defaults to 0.1.0
#   ./scripts/docker-release.sh cpu vulkan       # only build/save these targets
#   ./scripts/docker-release.sh --skip-sycl      # build cpu + vulkan only
#
# Output:
#   dist/infr-cpu-<VERSION>.tar.gz
#   dist/infr-vulkan-<VERSION>.tar.gz
#   dist/infr-sycl-intel-<VERSION>.tar.gz
#   dist/SHA256SUMS
#
# Load a saved image on another machine:
#   gunzip -c dist/infr-vulkan-0.1.0.tar.gz | docker load
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${VERSION:-0.1.0}"
DIST="$ROOT/dist"

# Targets: name -> (dockerfile, image tag, dist file stem)
ALL_TARGETS=(cpu vulkan sycl)
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --skip-sycl) SKIP_SYCL=1 ;;
    --skip-vulkan) SKIP_VULKAN=1 ;;
    --skip-cpu) SKIP_CPU=1 ;;
    cpu|vulkan|sycl) TARGETS+=("$arg") ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg (expected cpu|vulkan|sycl|--skip-cpu|--skip-vulkan|--skip-sycl)" >&2
      exit 1
      ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=("${ALL_TARGETS[@]}")
  [[ "${SKIP_CPU:-0}" == "1" ]] && TARGETS=("${TARGETS[@]/cpu/}")
  [[ "${SKIP_VULKAN:-0}" == "1" ]] && TARGETS=("${TARGETS[@]/vulkan/}")
  [[ "${SKIP_SYCL:-0}" == "1" ]] && TARGETS=("${TARGETS[@]/sycl/}")
fi

mkdir -p "$DIST"

image_tag() {
  case "$1" in
    cpu) echo "infr:cpu" ;;
    vulkan) echo "infr:vulkan" ;;
    sycl) echo "infr:sycl-intel" ;;
  esac
}

dist_stem() {
  case "$1" in
    cpu) echo "infr-cpu" ;;
    vulkan) echo "infr-vulkan" ;;
    sycl) echo "infr-sycl-intel" ;;
  esac
}

echo "==> infr Docker release — VERSION=${VERSION}"
echo "==> targets: ${TARGETS[*]}"

for t in "${TARGETS[@]}"; do
  [[ -z "$t" ]] && continue
  tag="$(image_tag "$t")"
  echo "==> [$t] building ${tag} (and ${tag}-${VERSION})"
  "$ROOT/scripts/docker-build.sh" "$t"
  docker tag "$tag" "${tag}-${VERSION}"
done

echo "==> saving images to ${DIST}/"
for t in "${TARGETS[@]}"; do
  [[ -z "$t" ]] && continue
  tag="$(image_tag "$t")"
  stem="$(dist_stem "$t")"
  out="${DIST}/${stem}-${VERSION}.tar.gz"
  echo "==> [$t] docker save ${tag}-${VERSION} | gzip -> ${out}"
  docker save "${tag}-${VERSION}" | gzip -1 > "$out"
done

echo "==> writing ${DIST}/SHA256SUMS"
(
  cd "$DIST"
  # shellcheck disable=SC2012
  sha256sum ./*"-${VERSION}.tar.gz" > SHA256SUMS.tmp
  mv SHA256SUMS.tmp SHA256SUMS
)

echo "==> done"
echo
echo "Artifacts:"
ls -lh "$DIST"/*"-${VERSION}.tar.gz" 2>/dev/null || true
echo
echo "Verify:"
echo "  cd dist && sha256sum -c SHA256SUMS"
echo
echo "Load elsewhere:"
echo "  gunzip -c dist/infr-vulkan-${VERSION}.tar.gz | docker load"
