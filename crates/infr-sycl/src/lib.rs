//! The Intel SYCL/oneAPI compute backend (`--dev sycl` / `INFR_DEV=sycl`).
//!
//! [`SyclBackend`] wraps [`infr_cpu::CpuBackend`] for every [`Backend`] method except identity
//! (`name`/`capabilities`): correctness rides the CPU reference interpreter end-to-end, while a
//! real SYCL device is initialized alongside it (via the C++ shim in `cxx/`) so `--dev sycl`
//! visibly reports a real Intel GPU (or, absent one, the SYCL CPU device / a host fallback — see
//! `build.rs`) and exposes a GEMM primitive ([`SyclBackend::gemm_f32`]) for accelerating
//! `Op::Linear` in a follow-up.
//!
//! ## Why forward everything to the CPU interpreter?
//! [`infr_core::backend::Backend::execute`] is one call per graph, and the CPU interpreter is a
//! monolithic per-op match (`infr_cpu`'s `execute`) — there is no seam to intercept just the
//! `Op::Linear` dispatches without either forking that interpreter or rewriting it to be
//! pluggable. Correctness-first MVP (this crate): every `Backend` method forwards straight to an
//! owned [`CpuBackend`], so a dense model run on `--dev sycl` is BYTE-IDENTICAL to `--dev cpu`
//! (same kernels, same weight binder) while still exercising a real SYCL device init/shutdown and
//! GEMM path (see `tests/gemm.rs`). Wiring the shim's oneDNN/SYCL GEMM into `Op::Linear` itself
//! is the natural next step — it would need `infr_cpu` to expose an override hook, or this crate
//! to grow its own thin `Op::Linear`-only interpreter layered in front of the CPU one.
//!
//! Building with the `sycl` feature requires a C++ toolchain; without a real SYCL compiler
//! (`icpx`, ideally — see `build.rs`) the shim degrades to a host CPU fallback that still reports
//! a device name and still computes correct GEMMs, just with no acceleration. Without the `sycl`
//! feature at all, this crate is not compiled (see `infr_llama`'s `sycl` feature, and
//! `infr-cli`'s clear "rebuild with --features sycl" runtime error when it was left off).
#![cfg(feature = "sycl")]

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan, ProgressScope};
use infr_core::config::Config as EngineConfig;
use infr_core::error::backend as be;
use infr_core::error::Result;
use infr_core::graph::Graph;
use infr_cpu::CpuBackend;
use std::os::raw::c_char;
use std::sync::Arc;

mod ffi {
    use std::os::raw::c_char;

    extern "C" {
        pub fn infr_sycl_init(name_out: *mut c_char, name_len: usize) -> i32;
        pub fn infr_sycl_shutdown();
        pub fn infr_sycl_is_gpu() -> i32;
        #[allow(dead_code)] // kept for parity with the C ABI; `SyclBackend` caches the name itself
        pub fn infr_sycl_device_name() -> *const c_char;
        pub fn infr_sycl_gemm_f32(
            c: *mut f32,
            a: *const f32,
            b: *const f32,
            m: i32,
            n: i32,
            k: i32,
        ) -> i32;
        pub fn infr_sycl_sync();
    }
}

/// The longest device name the shim's `infr_sycl_init` will report into a fixed stack buffer —
/// generous headroom over any real SYCL device-name string; a longer name is truncated by the
/// shim's own `snprintf`, never overflowed.
const DEVICE_NAME_BUF: usize = 256;

/// Intel SYCL/oneAPI compute backend. See the module doc for what's accelerated today (nothing —
/// [`Self::gemm_f32`] is exposed for tests/future `Op::Linear` use) vs. forwarded to the wrapped
/// [`CpuBackend`] (everything else, via the [`Backend`] impl below).
pub struct SyclBackend {
    inner: CpuBackend,
    device_name: String,
    is_gpu: bool,
}

impl SyclBackend {
    /// A backend on the shipped configuration, with the environment folded in — the same
    /// env-sourced bridge every other backend's `new()` is (see `CpuBackend::new`'s doc).
    /// Nothing in this repo calls this; prefer [`Self::new_with`] when a `Config` is on hand.
    pub fn new() -> Result<Self> {
        Self::new_with(Arc::new(EngineConfig::load_from_env()))
    }

    /// The real constructor: initializes the shim's SYCL queue/device (or its host fallback —
    /// see `cxx/shim.cpp`), logs what it found, and wraps a [`CpuBackend`] built on the SAME
    /// `cfg` for every other [`Backend`] method.
    pub fn new_with(cfg: Arc<EngineConfig>) -> Result<Self> {
        let mut name_buf = [0u8; DEVICE_NAME_BUF];
        // SAFETY: `name_buf` is a valid, correctly-sized, exclusively-borrowed buffer for the
        // duration of this call; the shim only ever writes a NUL-terminated string within it.
        let rc = unsafe { ffi::infr_sycl_init(name_buf.as_mut_ptr() as *mut c_char, name_buf.len()) };
        let device_name = read_c_string(&name_buf);
        if rc != 0 {
            return Err(be(format!(
                "sycl init: infr_sycl_init failed ({device_name}) — no SYCL/Level-Zero device \
                 and no usable host fallback"
            )));
        }
        // SAFETY: `infr_sycl_init` just returned success above, so the shim's global state is
        // initialized and this read is well-defined.
        let is_gpu = unsafe { ffi::infr_sycl_is_gpu() } != 0;
        tracing::info!(
            "[sycl backend] device: {device_name} ({})",
            if is_gpu {
                "Level Zero GPU"
            } else {
                "SYCL CPU device / host fallback"
            }
        );
        Ok(Self {
            inner: CpuBackend::new_with(cfg),
            device_name,
            is_gpu,
        })
    }

    /// The device name the shim's `infr_sycl_init` selected (Level Zero GPU, SYCL CPU device, or
    /// the pure host-loop fallback when built without a SYCL toolchain — see `build.rs`).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether the selected device is a Level Zero (or otherwise real) GPU (`true`) vs. the SYCL
    /// CPU device / host fallback (`false`).
    pub fn is_gpu(&self) -> bool {
        self.is_gpu
    }

    /// The wrapped CPU backend every other [`Backend`] method forwards to. The seam's weight
    /// binder (`infr_llama`'s `cpu_bind_with`) needs this concrete type, not the `dyn Backend`
    /// this type also implements — it calls `CpuBackend`-only methods (`map_weight`,
    /// `paged_weight`) that aren't part of the trait.
    pub fn inner(&self) -> &CpuBackend {
        &self.inner
    }

    /// Row-major f32 GEMM through the shim: `C[M,N] = A[M,K] * B[K,N]`. Public so
    /// `tests/gemm.rs` (and a future `Op::Linear` fast path) can exercise the real oneDNN/SYCL
    /// dispatch directly, independent of the CPU-forwarding `execute` below. Prefers oneDNN,
    /// falls back to a SYCL `parallel_for`, falls back further to a host loop — see the shim.
    pub fn gemm_f32(&self, c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<()> {
        assert_eq!(a.len(), m * k, "gemm_f32: A is not M*K elements");
        assert_eq!(b.len(), k * n, "gemm_f32: B is not K*N elements");
        assert_eq!(c.len(), m * n, "gemm_f32: C is not M*N elements");
        // SAFETY: the three slices above were just asserted to have exactly M*K / K*N / M*N
        // elements, matching the (M, N, K) the shim is told, so every pointer+len it derives
        // stays in bounds.
        let rc = unsafe {
            ffi::infr_sycl_gemm_f32(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m as i32, n as i32, k as i32)
        };
        if rc != 0 {
            return Err(be("sycl gemm_f32: the shim's GEMM failed"));
        }
        Ok(())
    }
}

impl Drop for SyclBackend {
    fn drop(&mut self) {
        // SAFETY: `infr_sycl_shutdown` is documented safe to call even if init failed or was
        // never called, and this backend never lets two instances race the shim's global state
        // (there is exactly one `SyclBackend` per initialized device in every caller today).
        unsafe { ffi::infr_sycl_shutdown() };
    }
}

/// Decode a NUL-terminated (or NUL-free, full-buffer) byte buffer the shim wrote into as UTF-8,
/// replacing anything invalid rather than failing — a device name is diagnostic text, never a
/// value the engine branches on.
fn read_c_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

impl Backend for SyclBackend {
    fn name(&self) -> &str {
        "sycl"
    }

    /// Same capabilities as the wrapped CPU backend (`combined_gu`/`embed_gather`/`gpu_sample`
    /// etc. all exactly what [`CpuBackend::capabilities`] returns) — every `execute` below
    /// forwards to `inner`, so the graph compiler must see precisely what the CPU interpreter can
    /// do, nothing more, nothing less. Only the `name` tag differs, so a capability dump can
    /// still tell the two backends apart.
    fn capabilities(&self) -> Capabilities {
        let mut caps = self.inner.capabilities();
        caps.name = "sycl".to_string();
        caps
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.inner.alloc(bytes, usage)
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.inner.alloc_uninit(bytes, usage)
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        self.inner.upload(dst, src)
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        self.inner.download(src, dst)
    }

    fn kv_overflow_report(&self) {
        self.inner.kv_overflow_report()
    }

    fn device_alloc_room(&self) -> Option<u64> {
        self.inner.device_alloc_room()
    }

    fn activation_peak(&self) -> Option<u64> {
        self.inner.activation_peak()
    }

    fn weight_progress(&self, total_bytes: Option<u64>) -> Box<dyn ProgressScope> {
        self.inner.weight_progress(total_bytes)
    }

    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        self.inner.copy_buffer(src, dst, bytes)
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        self.inner.compile(graph)
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        self.inner.execute(plan, bindings)
    }

    fn execute_chain(&self, plan: &dyn Plan, bindings: &Bindings, n: usize) -> Result<Option<Vec<u32>>> {
        self.inner.execute_chain(plan, bindings, n)
    }

    fn max_decode_chain(&self) -> usize {
        self.inner.max_decode_chain()
    }

    /// Syncs the wrapped CPU backend (a no-op — the interpreter is synchronous) AND the shim's
    /// own SYCL queue, so a caller that treats `sync` as "every op I issued has completed" gets
    /// that guarantee for the real device too, even though nothing is dispatched to it yet.
    fn sync(&self) -> Result<()> {
        self.inner.sync()?;
        // SAFETY: `infr_sycl_sync` is documented safe to call at any time (a no-op before init /
        // on the host fallback).
        unsafe { ffi::infr_sycl_sync() };
        Ok(())
    }

    fn moe_paged(&self) -> bool {
        self.inner.moe_paged()
    }

    fn dense_paged(&self) -> bool {
        self.inner.dense_paged()
    }
}
