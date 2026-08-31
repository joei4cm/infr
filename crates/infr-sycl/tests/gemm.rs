//! Exercises the shim's GEMM directly through [`SyclBackend::gemm_f32`] — independent of the
//! CPU-forwarding `execute` path (see `src/lib.rs`'s module doc). Only compiled with `--features
//! sycl` (the whole `infr-sycl` crate is `#![cfg(feature = "sycl")]`); this file mirrors that gate
//! so `cargo test -p infr-sycl` without the feature is a clean no-op instead of a link error.
#![cfg(feature = "sycl")]

use infr_sycl::SyclBackend;

/// `C[M,N] = A[M,K] * B[K,N]`, checked against a plain host computation. Passes on every tier the
/// shim can land on (oneDNN, SYCL parallel_for, or the pure host loop) since all three implement
/// the same row-major convention.
#[test]
fn gemm_f32_matches_naive_reference() {
    let be = SyclBackend::new().expect("sycl backend init (falls back to a host device when no toolchain is present)");

    // 2x3 * 3x2 = 2x2, deliberately non-symmetric so a transposed or misindexed kernel would
    // produce a visibly wrong answer rather than an accidentally-correct one.
    let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b: [f32; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3,2]
    let mut c = [0.0f32; 4]; // [2,2]
    be.gemm_f32(&mut c, &a, &b, 2, 2, 3).unwrap();

    let expect = naive_gemm(&a, &b, 2, 2, 3);
    for (got, want) in c.iter().zip(expect.iter()) {
        assert!((got - want).abs() < 1e-4, "got {c:?} want {expect:?}");
    }
}

/// A larger, non-square shape (the kind an `Op::Linear` projection actually has) — catches an
/// off-by-one in the row/col strides that the tiny 2x2 case above could hide.
#[test]
fn gemm_f32_larger_non_square_shape() {
    let be = SyclBackend::new().unwrap();
    let (m, n, k) = (5, 7, 11);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
    let mut c = vec![0.0f32; m * n];
    be.gemm_f32(&mut c, &a, &b, m, n, k).unwrap();

    let expect = naive_gemm(&a, &b, m, n, k);
    for (i, (got, want)) in c.iter().zip(expect.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "mismatch at {i}: got {got} want {want}"
        );
    }
}

#[test]
fn device_reports_a_name() {
    let be = SyclBackend::new().unwrap();
    assert!(!be.device_name().is_empty());
}

fn naive_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}
