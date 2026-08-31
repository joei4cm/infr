// C ABI for the infr SYCL backend shim — see `crates/infr-sycl/src/lib.rs` for the Rust side that
// declares these same signatures via `extern "C"`.
//
// Every function here is safe to call from Rust: no C++ exception ever crosses this boundary
// (each body catches everything internally and turns a failure into a nonzero return code).
#ifndef INFR_SYCL_SHIM_H
#define INFR_SYCL_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the device: a Level Zero GPU when one is found, else the SYCL CPU device, else (when
// built without a SYCL toolchain at all — see build.rs) a pure host fallback that still reports a
// name and still computes correct results, just with no real device underneath. Writes a
// NUL-terminated device name into `name_out` (truncated, never overflowed, to fit `name_len`).
// Returns 0 on success, nonzero if no device/queue could be created at all.
int infr_sycl_init(char *name_out, size_t name_len);

// Release whatever `infr_sycl_init` opened. Safe to call even if init failed or was never called.
void infr_sycl_shutdown(void);

// 1 if the selected device is a Level Zero GPU, 0 for the SYCL CPU device or the host fallback.
int infr_sycl_is_gpu(void);

// The device name `infr_sycl_init` selected — a pointer into a buffer owned by the shim, valid
// until the next `infr_sycl_init`/`infr_sycl_shutdown` call. Never NULL (a fixed placeholder
// string before the first successful `infr_sycl_init`).
const char *infr_sycl_device_name(void);

// Row-major f32 GEMM: `C[M,N] = A[M,K] * B[K,N]`. Prefers oneDNN when it was found at build time,
// falls back to a hand-rolled SYCL `parallel_for` when a real device is live but oneDNN isn't,
// falls back further to a plain host loop (no SYCL toolchain at all). Returns 0 on success.
int infr_sycl_gemm_f32(float *C, const float *A, const float *B, int M, int N, int K);

// Block until every queued SYCL operation (the GEMM above) has completed. A no-op on the host
// fallback, where every call is already synchronous.
void infr_sycl_sync(void);

#ifdef __cplusplus
}
#endif

#endif // INFR_SYCL_SHIM_H
