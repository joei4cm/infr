//! Compiles `cxx/shim.cpp` into a static lib linked into `infr-sycl` — ONLY when the `sycl`
//! feature is on (see `src/lib.rs`'s `#![cfg(feature = "sycl")]`). Build scripts run
//! unconditionally regardless of features, so the early return below is what actually keeps a
//! default (non-sycl) workspace build from ever invoking a C++ compiler or requiring the Intel
//! oneAPI toolchain.
//!
//! Toolchain preference: `icpx` (the Intel oneAPI DPC++ compiler — the ONLY one that can build a
//! real SYCL kernel) > `clang++` (some upstream intel/llvm builds install under this name) > a
//! plain `c++`. Two independent probes then decide what the shim actually gets built with:
//!
//! - Does `cxx` compile a `<sycl/sycl.hpp>` translation unit with `-fsycl`? If not (no SYCL
//!   headers/runtime — the common case outside the `intel/deep-learning-essentials` image this
//!   crate targets), the shim is built with `INFR_SYCL_NO_SYCL` — a pure host build with no
//!   `-fsycl` flag at all, so it compiles with an ordinary `clang++`/`c++`/`g++`.
//! - Does `cxx` compile a `<dnnl.hpp>` translation unit (oneDNN's C++ API)? If not, the shim is
//!   built with `INFR_SYCL_NO_ONEDNN` and its GEMM falls back to a hand-written SYCL
//!   `parallel_for` (or, absent SYCL too, a plain host loop).
//!
//! Either way the crate LINKS and [`infr_sycl::SyclBackend::gemm_f32`] still returns correct
//! numbers — see `cxx/shim.cpp`'s three-tier GEMM. Only a machine with the real toolchain
//! (`icpx` + oneDNN, e.g. the `intel/deep-learning-essentials` Docker image) gets the
//! accelerated path.

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cxx/shim.cpp");
    println!("cargo:rerun-if-changed=cxx/shim.h");
    println!("cargo:rerun-if-env-changed=INFR_SYCL_CXX");

    // The crate's own feature gate (mirrors `#![cfg(feature = "sycl")]` in `src/lib.rs`) — see
    // the module doc for why this has to be a build-script-level check too.
    if env::var_os("CARGO_FEATURE_SYCL").is_none() {
        return;
    }

    let cxx = pick_compiler();
    let have_sycl = probe_compiles(&cxx, "#include <sycl/sycl.hpp>\nint main() { return 0; }\n", &["-fsycl"]);
    let have_onednn = probe_compiles(&cxx, "#include <dnnl.hpp>\nint main() { return 0; }\n", &[]);

    if !have_sycl {
        println!(
            "cargo:warning=infr-sycl: `{cxx}` could not compile a <sycl/sycl.hpp> translation \
             unit with -fsycl — building the HOST CPU fallback (no real Level Zero/SYCL device). \
             Install the Intel oneAPI DPC++/C++ Compiler (`icpx`, e.g. via the \
             `intel/deep-learning-essentials` image) for GPU acceleration."
        );
    } else if !have_onednn {
        println!(
            "cargo:warning=infr-sycl: SYCL compiles via `{cxx}` but oneDNN (<dnnl.hpp>) was not \
             found — Op::Linear-style GEMM will fall back to a hand-written SYCL parallel_for \
             instead of oneDNN. Install oneDNN (ships in `intel/deep-learning-essentials`) for \
             the faster path."
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .compiler(&cxx)
        .file("cxx/shim.cpp")
        .include("cxx")
        .std("c++17")
        .warnings(false); // the shim's own tiers legitimately leave some branches unused per-config
    if have_sycl {
        build.flag("-fsycl");
    } else {
        build.define("INFR_SYCL_NO_SYCL", None);
    }
    if !have_onednn {
        build.define("INFR_SYCL_NO_ONEDNN", None);
    }
    build.compile("infr_sycl_shim");

    // `cc::Build::compile` already emitted the `rustc-link-lib=static=...` + search-path
    // directives for the archive itself; only the shim's OWN runtime deps are ours to add.
    if have_sycl {
        println!("cargo:rustc-link-lib=sycl");
        // icpx-compiled objects reference Intel's optimized libc helpers
        // (`_intel_fast_memcpy`, …). rustc links via the host `cc`/`lld`, which
        // does NOT pull icpx's driver libs automatically — name them explicitly
        // or the final `infr` link fails with an undefined-symbol error inside
        // the `intel/deep-learning-essentials` image.
        if cxx.contains("icpx") || env::var("INFR_SYCL_CXX").is_ok_and(|c| c.contains("icpx")) {
            for lib in ["irc", "imf", "svml", "irng", "intlc"] {
                println!("cargo:rustc-link-lib={lib}");
            }
            // Also ask the linker to keep SYCL device code linked the way icpx
            // expects (harmless no-op for some toolchains; required for others).
            println!("cargo:rustc-link-arg=-fsycl");
        }
    }
    if have_onednn {
        println!("cargo:rustc-link-lib=dnnl");
    }
}

/// `icpx` > `clang++` > `c++` (see the module doc). `INFR_SYCL_CXX` overrides the search
/// entirely — the escape hatch for a oneAPI install under a nonstandard name/path.
///
/// Each candidate must compile a TRIVIAL translation unit (`<cstdio>` + `main`), not merely
/// answer `--version` — a `clang++`/`c++` whose gcc-toolchain autodetection points at a
/// half-installed libstdc++ (mismatched `g++`/`libstdc++-dev` package versions, seen on some
/// distros) runs but fails on `#include <cstdio>`, and that failure needs to fall through to the
/// next candidate here rather than surface as a confusing shim-compile error later.
fn pick_compiler() -> String {
    if let Ok(c) = env::var("INFR_SYCL_CXX") {
        return c;
    }
    for candidate in ["icpx", "clang++", "c++"] {
        if probe_compiles(candidate, "#include <cstdio>\nint main() { return 0; }\n", &[]) {
            return candidate.to_string();
        }
    }
    panic!(
        "infr-sycl: feature `sycl` is enabled but no working C++ compiler was found on PATH \
         (looked for `icpx`, `clang++`, `c++`, and none could even compile a trivial \
         translation unit). Install a working C++ toolchain — ideally the Intel oneAPI DPC++/C++ \
         Compiler (`icpx`) for real SYCL acceleration — or build without `--features sycl`."
    );
}

/// Test-compile `source` with `cxx` (plus `extra_flags`) into a throwaway object file in
/// `OUT_DIR`, returning whether it succeeded. The only reliable way to tell a real SYCL/oneDNN
/// toolchain from a compiler that merely tolerates an unrecognized flag or a missing header.
fn probe_compiles(cxx: &str, source: &str, extra_flags: &[&str]) -> bool {
    let dir = env::var("OUT_DIR").expect("OUT_DIR is set by cargo for every build script");
    let src_path = Path::new(&dir).join("infr_sycl_probe.cpp");
    let obj_path = Path::new(&dir).join("infr_sycl_probe.o");
    if std::fs::write(&src_path, source).is_err() {
        return false;
    }
    let ok = Command::new(cxx)
        .arg("-std=c++17")
        .arg("-c")
        .args(extra_flags)
        .arg(&src_path)
        .arg("-o")
        .arg(&obj_path)
        .output()
        .is_ok_and(|o| o.status.success());
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&obj_path);
    ok
}
