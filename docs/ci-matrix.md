# CI platform matrix

A plan to lint and test on all three supported platforms. Today CI is one
platform wide, and the gap is not theoretical: **five real clippy failures are
sitting in platform-gated code right now**, found by running the cross-target
lint locally while writing this.

## What CI does today

Seven jobs, all `ubuntu-26.04` except one, with durations measured from recent
`main` runs:

| job           | runner       | what it runs                                               | time      |
| ------------- | ------------ | ---------------------------------------------------------- | --------- |
| `fmt`         | ubuntu-26.04 | `cargo fmt --check`                                        | ~15s      |
| `clippy`      | ubuntu-26.04 | `--workspace --all-targets --locked -D warnings`           | ~2m26s    |
| `test`        | ubuntu-26.04 | `nextest run --workspace` + `cargo test --doc`             | ~3m38s    |
| `cpu-goldens` | ubuntu-26.04 | `infr-llama` CPU goldens, real models (~1.7 GB download)   | 24–36 min |
| `build`       | ubuntu-26.04 | `cargo build --release`                                    | ~4m       |
| `metal-check` | ubuntu-26.04 | `cargo check -p infr-metal --target aarch64-apple-darwin`  | ~45s      |
| `test-macos`  | macos-15     | `cargo build --workspace`, then `cargo test -p infr-metal` | ~3m14s    |

Two things that table hides:

- **No Windows runner and no Windows target anywhere.** PR #91 added native
  Windows support and nothing in CI compiles it.
- **`test-macos` builds the workspace but only tests `-p infr-metal`.** The
  crates PR #91 changed — `infr-hub`, `infr-core` — have never had their suites
  run on macOS.

## The gap, demonstrated

Platform-gated code in the tree, counted directly:

| gate                    | sites |
| ----------------------- | ----- |
| `target_os = "macos"`   | 68    |
| `target_os = "windows"` | 23    |
| `cfg(windows)`          | 5     |
| `cfg(unix)`             | 11    |
| `target_os = "linux"`   | 7     |
| `cfg(not(unix))`        | 3     |

Windows-gated code lives in seven files across three crates — `infr-hub`
(`download.rs` 15 sites, `ranged.rs` 2, `pull.rs` 2), `infr-core` (`hostmem.rs`
3, `blockio.rs` 2) and `infr-vulkan` (`p2p.rs` 2, `tp_sem.rs` 2). Note
`infr-vulkan` is wider than B66 recorded, which named only the hub and core
files.

Running the lints that CI does not run turns up **five errors**:

**Windows** —
`cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings`:

```
error: field assignment outside of initializer for an instance created with Default::default()
  --> crates/infr-core/src/hostmem.rs:56:5
```

**macOS** —
`cargo clippy -p infr-metal --all-targets --target aarch64-apple-darwin -- -D warnings`:

```
error: using `chunks_exact` with a constant chunk size
    --> crates/infr-metal/src/exec.rs:2355:22
    --> crates/infr-metal/src/exec.rs:2364:22
error: manual implementation of `Option::map`
    --> crates/infr-metal/src/exec.rs:5545:27
    --> crates/infr-metal/src/exec.rs:5551:27
```

The macOS four are particularly worth noting: `metal-check` already
cross-compiles this exact crate on every commit, but it runs `cargo check`, not
`cargo clippy`. Changing one word in that job catches all four.

These paths are a snapshot. [`infr-plat.md`](infr-plat.md) proposes moving
`hostmem.rs` into a new crate, and lists fixing this exact lint as a
prerequisite that must land **before** the file moves. If that migration has
already run, the Windows failure above is fixed and lives at
`crates/infr-plat/`; verify before re-fixing it.

## What can cross-compile and what needs a real runner

This is the part that decides the cost, and it is asymmetric. Measured locally:

**Windows cross-compiles completely.**
`cargo check --workspace --all-targets --locked --target x86_64-pc-windows-gnu`
finishes clean in **1m17s** on a Linux host, no extra tooling. So does clippy
(modulo the error above). Use the `gnu` target, not `msvc` — `msvc` wants
`lib.exe`, which a Linux runner does not have.

**macOS does not.** Two C/C++ dependencies need an Apple toolchain to build for
`aarch64-apple-darwin`:

- `ring` (via `rustls` → `reqwest` → `infr-hub`) — so `infr-hub` and everything
  above it cannot cross-build.
- `esaxx-rs` (via `tokenizers` → `infr-chat`/`infr-llama`) — so `infr-llama`,
  `infr-chat`, `infr-engine`, `infr-server`, `infr-cli` cannot either.

What _is_ cross-lintable for Apple is `infr-core`, `infr-gguf`, `infr-vulkan`
and `infr-metal` — which is why `metal-check` is scoped to one crate. Since 68
of the platform-gated sites are macOS ones and `infr-metal` is
`#![cfg(target_os = "macos")]` in its entirety, that subset is most of the
value; the rest needs the `macos-15` runner it already has.

### A hazard to fix first

`.cargo/config.toml` sets, unconditionally:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

That applies to cross-target builds too, where it is meaningless and noisy —
cross-compiling to `aarch64-apple-darwin` on this x86 box emits a wall of
`'+avx512dq' is not a recognized feature for this target` and
`'znver5' is not a recognized processor for this target`. It should be scoped
per-target (`[target.x86_64-unknown-linux-gnu] rustflags = [...]`) so
cross-target jobs do not inherit it. Left as-is, every cross job in the matrix
starts with noise that hides real diagnostics.

## Proposed matrix

Three changes, ordered cheapest-first. They are independent — do any subset.

### 1. Cross-lint from the existing Linux runner (minutes, no new runners)

Add Windows to the existing `clippy` job as extra targets, and upgrade
`metal-check` from `check` to `clippy`:

- `cargo clippy --workspace --all-targets --locked --target x86_64-pc-windows-gnu -- -D warnings`
- `cargo clippy -p infr-metal -p infr-core -p infr-gguf -p infr-vulkan --all-targets --locked --target aarch64-apple-darwin -- -D warnings`

Cost: roughly +1.5 min on a job that already runs, no new runner minutes. This
alone closes all five known failures and is the highest value-per-minute change
in this document. It also gives Windows a compile gate for the first time — B66
records that `main` before PR #91 had 11 compile errors for
`x86_64-pc-windows-gnu`, and nothing would have caught that.

### 2. Fan `test` out over three runners

`runs-on: ${{ matrix.os }}` over `[ubuntu-26.04, macos-15, windows-2025]` for
the `test` job (`nextest run --workspace`). At ~3m38s on Linux this is the
affordable one — unlike `cpu-goldens`, which must stay single-platform (see
below).

Two things this needs:

- **A Windows-shaped lock test.** `file_lock_is_exclusive`
  (`crates/infr-hub/src/download.rs:766`) is gated
  `#[cfg(not(target_os = "windows"))]` at `:764`, so the `LockFileEx` path added
  by PR #91 has no test on any platform. It is a two-process exclusion check, so
  it needs a real Windows runner — a cross-check cannot substitute.
- **A decision about `test-macos`.** Once `test` runs on `macos-15`, the
  existing `test-macos` job overlaps it. Keep `test-macos` for the
  `--include-ignored` Metal parity suite (which needs a real Metal device and
  prints the pipeline-cache measurement nothing else records), and let the
  matrix job cover the ordinary workspace suite.

### 3. Leave `cpu-goldens` on one platform — and make it matter less

Do **not** fan the goldens job out. Beyond the 24–36 minute cost per platform,
it would not work: commit `273f8d4` records that `cpu_golden_qwen35`'s
exact-token FNV hash was not reproducible between the GitHub x86 runner and the
dev box — same OS, same ISA family, different microarchitecture, coherent
output, different tokens. Fanning that assertion across three platforms
multiplies a known-flaky check.

The real fix is [`synthetic-models.md`](synthetic-models.md): tolerance-scored
logit goldens on synthetic fixtures are portable by construction, need no
downloads, and never self-skip. **That plan is the enabler for a three-platform
test matrix** — until it lands, the only model-level coverage that can safely
run on all three platforms is what does not depend on a real GGUF.

## Sequencing

1. Scope `target-cpu=native` per-target in `.cargo/config.toml`.
2. Fix the five lint errors.
3. Add the two cross-lint invocations (change 1). These three together are one
   small PR and should not wait on anything.
4. Add a Windows-shaped `file_lock_is_exclusive` equivalent.
5. Fan `test` out to three runners (change 2), reconciling `test-macos`.
6. Revisit the goldens job only after synthetic fixtures land.

## Open questions

- **Runner cost.** Windows and macOS runners bill at higher multipliers than
  Linux. Change 1 costs nothing extra; change 2 is the one that needs a budget
  decision, and it is the user's call, not a technical one.
- **`msvc` vs `gnu`.** The cross-lint uses `gnu` because it works on a Linux
  host. If a real `windows-2025` runner lands in change 2, it will build `msvc`
  by default — which is a different ABI and can surface different problems.
  Running both is possible; whether it is worth it is unknown until the first
  `msvc` run happens.
- **`available_bytes` asymmetry** (carried from B66, still unaddressed): the
  Linux arm picks `MemAvailable` and clamps by the cgroup limit; the Windows arm
  returns `ullAvailPhys` with no container clamp. A matrix that runs tests on
  Windows may surface this as a sizing difference rather than a compile error.
