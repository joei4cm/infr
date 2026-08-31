// Implementation of the C ABI declared in `shim.h` — see that file's doc for the contract, and
// `crates/infr-sycl/build.rs` for how the two config macros below get defined:
//
//   INFR_SYCL_NO_SYCL   — no working SYCL compiler at build time (no `<sycl/sycl.hpp>`, no
//                          `-fsycl`). Every function below degrades to a pure host
//                          implementation: `infr_sycl_init` never opens a real device,
//                          `infr_sycl_gemm_f32` runs the triple-loop tier directly.
//   INFR_SYCL_NO_ONEDNN — SYCL compiles, but oneDNN (`<dnnl.hpp>`) was not found. GEMM skips the
//                          oneDNN tier and goes straight to the SYCL `parallel_for` tier.
//
// Neither macro is ever defined when the OTHER's toolchain is fully present (the intended
// `intel/deep-learning-essentials` target), so the three tiers in `infr_sycl_gemm_f32` — oneDNN,
// SYCL parallel_for, host loop — are each reachable on exactly the build that can support them.
#include "shim.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#ifndef INFR_SYCL_NO_SYCL
#include <sycl/sycl.hpp>
#if !defined(INFR_SYCL_NO_ONEDNN)
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#endif
#endif

namespace {

// Every piece of shim state is process-global and guarded by ONE mutex — this backend is
// constructed at most once per process in practice (`SyclBackend::new_with`), and the lock cost
// is irrelevant next to a device init or a GEMM.
std::mutex g_mu;
bool g_ready = false;
bool g_is_gpu = false;
std::string g_device_name = "(uninitialized)";

#ifndef INFR_SYCL_NO_SYCL
sycl::queue *g_queue = nullptr;
#if !defined(INFR_SYCL_NO_ONEDNN)
dnnl::engine *g_dnnl_engine = nullptr;
dnnl::stream *g_dnnl_stream = nullptr;
#endif
#endif

} // namespace

int infr_sycl_init(char *name_out, size_t name_len) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_ready) {
#ifndef INFR_SYCL_NO_SYCL
        try {
            // Level Zero GPU first (the whole point of this backend); fall back to the SYCL CPU
            // device rather than failing outright, so a machine with the oneAPI RUNTIME
            // installed but no Intel GPU still gets a working, if unaccelerated, SYCL path. (This
            // asks SYCL's generic GPU selector, which may also match a non-Level-Zero GPU backend
            // — `infr_sycl_is_gpu` reports "found A gpu", not specifically "found Level Zero";
            // good enough for the device banner this backend prints today.)
            try {
                g_queue = new sycl::queue(sycl::gpu_selector_v);
                g_is_gpu = true;
            } catch (const sycl::exception &) {
                g_queue = new sycl::queue(sycl::cpu_selector_v);
                g_is_gpu = false;
            }
            g_device_name = g_queue->get_device().get_info<sycl::info::device::name>();
        } catch (const std::exception &e) {
            g_device_name = std::string("sycl init failed: ") + e.what();
            if (name_out && name_len > 0) {
                std::snprintf(name_out, name_len, "%s", g_device_name.c_str());
            }
            return 1;
        } catch (...) {
            g_device_name = "sycl init failed: unknown error";
            if (name_out && name_len > 0) {
                std::snprintf(name_out, name_len, "%s", g_device_name.c_str());
            }
            return 1;
        }
#if !defined(INFR_SYCL_NO_ONEDNN)
        // oneDNN binding to the live SYCL device/context is best-effort: if it fails, GEMM simply
        // falls back to the SYCL parallel_for tier below — not fatal to `infr_sycl_init` itself.
        try {
            g_dnnl_engine = new dnnl::engine(
                dnnl::sycl_interop::make_engine(g_queue->get_device(), g_queue->get_context()));
            g_dnnl_stream =
                new dnnl::stream(dnnl::sycl_interop::make_stream(*g_dnnl_engine, *g_queue));
        } catch (...) {
            delete g_dnnl_engine;
            g_dnnl_engine = nullptr;
            delete g_dnnl_stream;
            g_dnnl_stream = nullptr;
        }
#endif
#else
        // No SYCL toolchain at build time (see the file doc) — a pure host device that is
        // honest about not being one, rather than pretending to a GPU that was never opened.
        g_device_name = "cpu fallback (built without a SYCL toolchain)";
        g_is_gpu = false;
#endif
        g_ready = true;
    }
    if (name_out && name_len > 0) {
        std::snprintf(name_out, name_len, "%s", g_device_name.c_str());
    }
    return 0;
}

void infr_sycl_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mu);
#ifndef INFR_SYCL_NO_SYCL
#if !defined(INFR_SYCL_NO_ONEDNN)
    delete g_dnnl_stream;
    g_dnnl_stream = nullptr;
    delete g_dnnl_engine;
    g_dnnl_engine = nullptr;
#endif
    delete g_queue;
    g_queue = nullptr;
#endif
    g_ready = false;
    g_is_gpu = false;
    g_device_name = "(uninitialized)";
}

int infr_sycl_is_gpu(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_is_gpu ? 1 : 0;
}

const char *infr_sycl_device_name(void) {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_device_name.c_str();
}

int infr_sycl_gemm_f32(float *C, const float *A, const float *B, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return 0; // a degenerate GEMM has nothing to write — that is success, not an error.
    }
    if (!C || !A || !B) {
        return 1;
    }

#if !defined(INFR_SYCL_NO_SYCL) && !defined(INFR_SYCL_NO_ONEDNN)
    // Tier 1: oneDNN's matmul primitive, on whatever engine `infr_sycl_init` bound (the SYCL
    // device when the sycl_interop bind above succeeded).
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_dnnl_engine && g_dnnl_stream) {
            try {
                using namespace dnnl;
                memory::dims a_dims = {M, K}, b_dims = {K, N}, c_dims = {M, N};
                memory::desc a_md(a_dims, memory::data_type::f32, memory::format_tag::ab);
                memory::desc b_md(b_dims, memory::data_type::f32, memory::format_tag::ab);
                memory::desc c_md(c_dims, memory::data_type::f32, memory::format_tag::ab);
                memory a_mem(a_md, *g_dnnl_engine, const_cast<float *>(A));
                memory b_mem(b_md, *g_dnnl_engine, const_cast<float *>(B));
                memory c_mem(c_md, *g_dnnl_engine, C);
                matmul::primitive_desc pd(*g_dnnl_engine, a_md, b_md, c_md);
                matmul prim(pd);
                prim.execute(*g_dnnl_stream, {
                                                 {DNNL_ARG_SRC, a_mem},
                                                 {DNNL_ARG_WEIGHTS, b_mem},
                                                 {DNNL_ARG_DST, c_mem},
                                             });
                g_dnnl_stream->wait();
                return 0;
            } catch (...) {
                // Fall through to the tiers below — a oneDNN failure on this call is not fatal.
            }
        }
    }
#endif

#ifndef INFR_SYCL_NO_SYCL
    // Tier 2: a hand-rolled SYCL `parallel_for` — no oneDNN, but still a real device dispatch.
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_queue) {
            try {
                sycl::queue &q = *g_queue;
                float *dA = sycl::malloc_device<float>(static_cast<size_t>(M) * K, q);
                float *dB = sycl::malloc_device<float>(static_cast<size_t>(K) * N, q);
                float *dC = sycl::malloc_device<float>(static_cast<size_t>(M) * N, q);
                if (dA && dB && dC) {
                    q.memcpy(dA, A, sizeof(float) * static_cast<size_t>(M) * K).wait();
                    q.memcpy(dB, B, sizeof(float) * static_cast<size_t>(K) * N).wait();
                    q.submit([&](sycl::handler &h) {
                         h.parallel_for(sycl::range<2>(static_cast<size_t>(M), static_cast<size_t>(N)),
                                        [=](sycl::id<2> idx) {
                                            int m = static_cast<int>(idx[0]);
                                            int n = static_cast<int>(idx[1]);
                                            float acc = 0.0f;
                                            for (int k = 0; k < K; ++k) {
                                                acc += dA[m * K + k] * dB[k * N + n];
                                            }
                                            dC[m * N + n] = acc;
                                        });
                     }).wait();
                    q.memcpy(C, dC, sizeof(float) * static_cast<size_t>(M) * N).wait();
                    sycl::free(dA, q);
                    sycl::free(dB, q);
                    sycl::free(dC, q);
                    return 0;
                }
                if (dA) sycl::free(dA, q);
                if (dB) sycl::free(dB, q);
                if (dC) sycl::free(dC, q);
            } catch (...) {
                // Fall through to the host loop below.
            }
        }
    }
#endif

    // Tier 3: a plain host loop — no SYCL toolchain at build time, or the device tiers above
    // failed. Correctness-first: this is what the shim's own `gemm_f32` unit test (and any future
    // Op::Linear caller) can always check itself against, even on a machine with none of the
    // toolchain.
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
    return 0;
}

void infr_sycl_sync(void) {
#ifndef INFR_SYCL_NO_SYCL
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_queue) {
        g_queue->wait();
    }
#endif
}
