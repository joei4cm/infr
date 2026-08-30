# `infr-plat` — the platform seam

A plan to collect OS-specific code into one reviewable place, and to key the
Metal backend on something that says what it means instead of on the host OS.

## Two problems wearing the same syntax

Every `cfg` in `crates/*/src` and `crates/*/tests`:

| gate                    | sites |
| ----------------------- | ----- |
| `target_os = "macos"`   | 68    |
| `target_os = "windows"` | 23    |
| `cfg(unix)`             | 11    |
| `target_os = "linux"`   | 7     |
| `cfg(windows)`          | 5     |
| `cfg(not(unix))`        | 3     |

The populations do not overlap. Counts exclude `build.rs` and `Cargo.toml`, and
that exclusion hides nothing: no `build.rs` in the tree contains a platform
`cfg`, and the workspace has exactly two `[target.'cfg(...)']` tables —
`infr-core/Cargo.toml:21` (windows) and `infr-metal/Cargo.toml:18` (macos) —
both accounted for below. One of the 68 is a false positive: a `//` comment in
`infr-core/tests/no_bare_print.rs:138` quoting the string while gating nothing.
So 67 are real.

**Problem A — OS plumbing.** One capability, same intent, different spelling per
platform. This is what a platform crate is for.

**Problem B — backend availability.** `#[cfg(target_os = "macos")]` used to mean
"the Metal backend exists here". **This is where the mass is**: 43 of the 68 are
in `infr-llama` (37 src / 6 tests) and 12 more in `infr-cli` — conditional
re-exports in `chat/mod.rs`, a `metal_chat_model` constructor, and errors like
`chat/cpu.rs:52`'s _"the Metal backend is only available on macOS"_. Moving
these into a platform crate would relocate them, not remove them: they are not a
syscall spelling difference, they are a capability that is absent. **Problem B
is a feature-gating problem, not an `infr-plat` problem** — see below.

**Problem C — the GPU interop handles**, which turn out not to be what they look
like; also below.

## Scope: what moves

The third column is the predicate as written, because it is not uniform — two
rows key on `cfg(unix)`/`cfg(not(unix))` rather than on Windows, which is
equivalent today only because no third target ships.

| capability                   | today                               | predicate        | first arm                                | other arm                                    |
| ---------------------------- | ----------------------------------- | ---------------- | ---------------------------------------- | -------------------------------------------- |
| available physical memory    | `infr-core/src/hostmem.rs` (3)      | `target_os`      | **linux**: `MemAvailable` + cgroup clamp | windows: `ullAvailPhys`, no clamp            |
| positional read              | `infr-core/src/blockio.rs` (2)      | `unix`/`windows` | `FileExt::read_at`                       | `seek_read`; third arm is a `compile_error!` |
| positional write             | `infr-hub/src/ranged.rs` (2)        | `target_os`      | `write_all_at` (std loops internally)    | hand-rolled `seek_write` retry loop          |
| content-addressed link       | `infr-hub/src/pull.rs` (2)          | `target_os`      | `symlink`, no fallback                   | `symlink_file`, falling back to `hard_link`  |
| advisory exclusive file lock | `infr-hub/src/download.rs` (15)     | `target_os`      | `flock(LOCK_EX)`                         | `LockFileEx`/`UnlockFileEx`, hand-rolled FFI |
| process liveness probe       | `infr-core/src/kernel_cache.rs` (2) | `unix`           | `kill(pid, 0)`                           | stub returning `true`, deliberately          |
| signal handlers              | `infr-cli/src/main.rs` (3)          | `unix`           | `sigaction` + async-signal-safe handler  | no-op stub                                   |

Excluding the `p2p.rs`/`tp_sem.rs` stub (see Problem C), Windows-gated code is
**24 sites across five files in two crates**: `infr-hub` (`download.rs` 15,
`ranged.rs` 2, `pull.rs` 2) and `infr-core` (`hostmem.rs` 3, `blockio.rs` 2).

Three details are easy to get backwards and are therefore spelled out:

- **`available_bytes` has no Unix arm — it has a _Linux_ arm.** The third branch
  is `#[cfg(not(any(target_os = "linux", windows)))] { None }`
  (`hostmem.rs:44-47`), so **macOS and every BSD get no memory probe at all**.
  Anything that sizes an arena from this figure is running unbudgeted on macOS
  today. That is a gap the move should surface, not preserve silently.
- **The link step is a symlink, not a hard link.** `pull.rs::link_blob` calls
  `std::os::unix::fs::symlink` unconditionally on unix; the hard link is only a
  Windows fallback for when `symlink_file` fails without
  `SeCreateSymbolicLinkPrivilege`. No copy fallback exists anywhere.
- **`blockio.rs`'s third arm is a compile-time `compile_error!`**, not a runtime
  "unsupported".

### The category a `cfg` grep does not find

Platform-specific code does not always carry a `cfg`. Four sites in production
code hand-roll XDG/`HOME` conventions with no Windows path, and none appear in
the table above:

- **`infr-core/src/config/file.rs:76-80`** — the global config path is
  `XDG_CONFIG_HOME`, else `HOME/.config`, else `None`. On Windows neither
  variable is normally set, so **`infr` can never find a global config file
  there**, silently. A live defect.
- **`infr-core/src/kernel_cache.rs:127-131`** — same shape for the cache
  directory; `KernelCache::open` falls back to `temp_dir()` (`:185`), so it
  degrades to losing cross-reboot persistence rather than failing.
- **`infr-cli/src/main.rs:3343`** (`fork_diffusion_cli_path`) —
  `env::var("HOME").unwrap_or_default()`, which on Windows silently builds a
  bogus **relative** path rather than returning `None`. It also hardcodes a
  personal directory, which is a separate problem.
- By contrast **`infr-hub/src/store.rs:46`** does it correctly with
  `dirs::cache_dir()`, and `dirs` is already a workspace dependency
  (`Cargo.toml:99`).

So a third capability belongs in the crate — **config/cache/data directory
resolution** — and the fix is to use `dirs` uniformly rather than to write more
`cfg` arms. Note this item is **new API closing a bug, not a relocation**: it is
the one part of this work that changes behaviour on Windows, from "silently
cannot find config" to "works".

### Problem C is a stub, not a second platform

It is tempting to read `p2p.rs`/`tp_sem.rs` as aliasing a real Win32 `HANDLE`
against a `RawFd`. They do not. The actual Windows arm is `p2p.rs:38-39`:

```rust
#[cfg(target_os = "windows")]
type RawFd = std::os::raw::c_int;
```

There is no `HANDLE`, and no `VK_KHR_external_memory_win32` anywhere in the tree
— every call site is POSIX-fd-only (`vkGetMemoryFdKHR`, `libc::dup`). The
Windows alias exists only to keep `P2pExport`'s field type-checking; the path is
functionally Unix-only.

**So there is nothing to move.** Hoisting a `c_int` masquerading as a Windows
handle into a shared crate would launder a stub into an abstraction. Either
`infr-plat` exposes a Unix-only `GpuHandle` and the Windows arm is deleted in
favour of a real refusal, or this is left alone until someone implements
`external_memory_win32` — which is feature work, not a move. **Recommendation:
leave `p2p.rs`/`tp_sem.rs` out of scope** and record the stub as a known gap.
That costs the "no `cfg` outside `infr-plat`" rule one exception, a fair price
for not pretending a stub is a port.

### What does not move

`infr-metal` stays where it is. It is `#![cfg(target_os = "macos")]` at
`src/lib.rs:4` with Apple-only deps target-gated in its own `Cargo.toml`, so off
macOS it already compiles to an empty lib with no deps. It is a **backend**; the
hardware abstraction in this tree is the `Backend` trait. That is why this crate
is not called `infr-hal`.

## Crate, or module in `infr-core`?

A tempting justification for a separate crate is that `infr-plat` must be a leaf
to avoid a dependency cycle. **That does not hold.** `infr-hub`, `infr-cli` and
`infr-vulkan` all already depend on `infr-core`, and `infr-core` depends only on
`infr-prof`/`infr-prof-rt` — so a `pub mod platform` inside `infr-core` would be
reachable by every consumer named, at zero new workspace cost.

The honest justifications for a separate crate are narrower:

- **An absolute enforcement rule is marginally cheaper to enforce.** "No
  `cfg(target_os)` outside `infr-plat`" is a one-line check; "outside
  `infr-core/src/platform/`" is the same check with a path prefix — nearly as
  good.
- **`infr-core` is already the workspace's dumping ground**, and CLAUDE.md's
  rule is that a module which keeps growing is a defect. Splitting out is the
  direction of travel.

**Against it, stated plainly: this is a heavy structure for the payload.** About
120 lines of code move. A ninth crate is a permanent commitment — another
manifest, another CI matrix entry, another thing every future crate reasons
about. The minimal version of this whole plan is: `pub mod platform` in
`infr-core`, the same grep-based enforcement scoped to a path prefix, no
`infr-build`. That is the lower-cost option and it delivers most of the benefit.

This document assumes the crate, since that is the decision taken. The module
variant remains a legitimate fallback and every other section works unchanged
under it.

### Manifest

`windows` is **not** a workspace dependency — it is declared directly in
`infr-core/Cargo.toml:22` as
`windows = { version = "0.58", features = ["Win32_System_SystemInformation"] }`.
And `libc` is a plain unconditional dependency of `infr-core`, not target-gated,
so target-gating it here is new rather than a copied pattern. `infr-plat` needs
the union of Win32 feature sets (all four verified present in `windows` 0.58),
and follows the workspace-inherited `version.workspace` / `edition.workspace`
convention every other crate uses:

```toml
[dependencies]
dirs.workspace = true          # config/cache/data dirs — the non-cfg category above

[target.'cfg(unix)'.dependencies]
libc.workspace = true

[target.'cfg(windows)'.dependencies]
windows = { version = "0.58", features = [
    "Win32_Foundation",              # HANDLE, BOOL
    "Win32_Storage_FileSystem",      # LockFileEx / UnlockFileEx
    "Win32_System_IO",               # OVERLAPPED
    "Win32_System_SystemInformation" # GlobalMemoryStatusEx
] }
```

Add `"crates/infr-plat"` to `members` and
`infr-plat = { path = "crates/infr-plat" }` to `[workspace.dependencies]`.

**A bonus this unlocks:** `infr-hub` today hand-rolls its Win32 FFI —
`download.rs:505` opens `unsafe extern "system" { fn LockFileEx(...) }` with a
hand-written `OVERLAPPED`, while `hostmem.rs` uses the `windows` crate properly.
B66 recorded the open question as "whether a second dependency edge on
`infr-hub` is wanted". After this move there is no such edge, because `infr-hub`
stops doing FFI at all — the lock lives in the crate that already needs
`windows`. **B66's hand-rolled-FFI residual is resolved as a side effect.**

### The seam

```rust
pub fn available_bytes() -> Option<u64>;
pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize>;
pub fn write_all_at(f: &File, buf: &[u8], off: u64) -> io::Result<()>;

/// `target` is a RELATIVE path (`../../blobs/<hex>`), deliberately, so the hub
/// directory stays movable and byte-identical to what `huggingface_hub` and
/// llama.cpp write. Fails if `link` already exists — the caller unlinks first
/// (`pull.rs:160` does `let _ = fs::remove_file(&link)`), so idempotency is the
/// caller's job and must stay there.
pub fn link_blob(target: &str, link: &Path) -> io::Result<()>;

/// Blocks until any other holder releases; released on drop AND on process
/// death. Advisory-strength only — see the trap below.
pub struct FileLock { /* .. */ }
impl FileLock {
    pub fn acquire(path: &Path) -> io::Result<FileLock>;
}

pub fn pid_alive(pid: i32) -> bool;

/// Idempotent; installing twice replaces the previous handler. Returns the
/// installation error rather than panicking, since a CLI can run without it.
pub fn install_signal_handlers(on_signal: fn(i32) -> bool) -> io::Result<()>;

pub fn config_dir() -> Option<PathBuf>;
pub fn cache_dir() -> Option<PathBuf>;
```

**`install_signal_handlers` takes a callback, and that is not cosmetic.** The
existing handler at `infr-cli/src/main.rs:519` is:

```rust
extern "C" fn on_signal(signo: libc::c_int) {
    if infr_core::shutdown::request_shutdown(signo) { return; }
    // ... second signal: async-signal-safe write + _exit
}
```

It calls into `infr_core::shutdown`. Moving it verbatim would make `infr-plat`
depend on `infr-core` and destroy the leaf property. Passing the latch in as a
`fn(i32) -> bool` — which matches `request_shutdown`'s real signature at
`shutdown.rs:55` exactly — keeps `infr-plat` a leaf and leaves `infr-cli` to
wire it. Note the `128 + signo` exit-code helper carries **no** `cfg` and is not
part of this capability.

### The trap: do not unify a signature over a real semantic difference

An identical type over different guarantees is the "not supported here arrives
as passed" shape. Three cases:

- **`available_bytes`.** Linux picks `MemAvailable` and clamps by the cgroup
  limit, because sizing an arena from the unclamped figure is an OOM kill in a
  container. Windows returns `ullAvailPhys` with no clamp. macOS returns
  **nothing**. A bare `-> Option<u64>` hides all three behaviours behind one
  type. Either return the clamp provenance, or document it where an arena is
  sized. (B66 carries the Linux/Windows half as an open item; settle it here, in
  one place.)
- **`FileLock`.** `flock` on unix is **advisory** — a process that never calls
  it reads and writes freely. `LockFileEx` on Windows, as used here
  (`LOCKFILE_EXCLUSIVE_LOCK`, no `FAIL_IMMEDIATELY`), is **mandatory** for the
  locked range, enforced against every handle. Both block and both release on
  process death, so those are not the difference. The API must document that
  what it promises is advisory-strength, or a caller will rely on the stronger
  guarantee and be wrong on the platform we develop on.
- **`pid_alive`.** The non-unix arm returns `true` on purpose — "lose the
  tripwire, never misbehave", with a comment saying so. That reasoning must
  survive the move rather than being flattened into an unexplained stub.

`write_all_at` versus the `seek_write` retry loop is **not** in this list: std's
unix `write_all_at` already loops internally on short writes and `EINTR`, and
Windows has no such wrapper, so the hand-rolled loop is the same operation.
Classifying it correctly matters as much as flagging the others.

## Problem B: gating the Metal backend

A `#![cfg(feature = "metal")]` on `infr-metal` plus a build script emitting
`cfg(metal)` in consumers looks reasonable and does not work. Two reasons, each
verified by experiment:

1. **A build script cannot set a Cargo feature.** Feature resolution happens
   before any build script runs. `cargo:rustc-cfg=metal` injects `--cfg metal`,
   never `--cfg feature="metal"`. So `cfg(feature = "metal")` and `cfg(metal)`
   are two disconnected predicates.
2. **Cargo _can_ enable a feature per target.** With a plain dependency plus
   `[target.'cfg(target_os = "windows")'.dependencies] a = { features = ["heavy"] }`,
   `cargo tree -e features` shows `a feature "heavy"` for the Windows target and
   only `default` for the host.

What Cargo cannot do is let a crate enable **its own** feature per target. So
consumer code cannot write `#[cfg(feature = "metal")]` and keep today's "on by
default on macOS" behaviour. That forces a choice:

**Option A — one build-script `cfg(metal)`, used uniformly (recommended).** No
Cargo feature at all. Every crate that gates on Metal — `infr-metal` included —
emits `cfg(metal)` from its `build.rs` when `CARGO_CFG_TARGET_OS == "macos"`
unless an opt-out env var is set. `infr-metal`'s `metal`/`objc` deps stay
target-gated exactly as they are today. This matches existing precedent:
`crates/{infr-core,infr-cpu,infr-gguf,infr-llama}/build.rs` are four
byte-identical copies of this shape for `INFR_PROFILE` → `cfg(infr_profile)`,
single-colon `cargo:` syntax included.

```rust
println!("cargo:rerun-if-env-changed=INFR_NO_METAL");
println!("cargo:rustc-check-cfg=cfg(metal)");
if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    && !std::env::var("INFR_NO_METAL").is_ok_and(|v| !v.is_empty() && v != "0")
{
    println!("cargo:rustc-cfg=metal");
}
```

Cost: `infr-metal` and `infr-cli` each need a `build.rs` they do not have today,
against a tree that already carries four identical copies of one build script.

**Option B — a real Cargo feature end-to-end.** `infr-metal` declares
`metal = ["dep:metal", "dep:objc"]` with those deps `optional` **and**
target-gated; consumers declare `metal = ["infr-metal/metal"]` plus a
target-conditional dependency to switch it on for macOS. Discoverable via
`--features`, no build script. The cost is that consumer code still cannot key
on its own feature per target, so `infr-llama`/`infr-cli` keep
`#[cfg(target_os = "macos")]` and only `infr-metal` itself becomes feature-gated
— leaving **55 of the 68 gates exactly as they are**, i.e. it does not solve
Problem B.

**Either way, guard against the confusing failure.** Today selecting Metal off
macOS fails cleanly at runtime — `parse_dev_spec` accepts `metal` on any OS and
`chat/cpu.rs:50-53` returns "the Metal backend is only available on macOS". If a
feature can be forced on for a non-Apple target, the build instead dies inside
unresolved `metal`/`objc` imports. Gate the crate body on
`all(feature = "metal", target_os = "macos")`, or add an explicit
`compile_error!`, so misuse fails at the crate boundary with a sentence.

**What this buys, precisely.** On macOS both arms become compilable — today the
"Metal unavailable" path cannot be built on a Mac and the "available" path
cannot be built anywhere else, so each is compiled by exactly one runner and
neither by both. **What it does not buy:** Linux still cannot compile the Metal
code, because `metal`/`objc` link the Objective-C runtime. This names intent and
doubles the arms macOS CI can check; it does not make Metal cross-buildable.

## Prerequisites — do these first, separately

Neither is part of the refactor, and neither should wait for it:

- **Fix `infr-core/src/hostmem.rs:56`** (`clippy::field_reassign_with_default`,
  which `ci-matrix.md`'s cross-lint already fails on). A verbatim move would
  carry the defect to the new path and the lint would be red the moment it is
  enabled. **This ordering is load-bearing.**
- **Fix `config/file.rs`'s Windows config-path hole** with `dirs::config_dir()`.
  It is a live defect; the refactor is not. When the directory helpers later
  land in `infr-plat`, they absorb the already-fixed code rather than re-fixing
  it.

Also confirm **`ci-matrix.md`'s change 1 (the Windows cross-lint) is live**
before starting, or the enforcement below has nothing to enforce with.

## Migration staging

Move first, change behaviour second, so a reviewer can see a move was only a
move.

1. Create `infr-plat` empty, wire it into the workspace.
2. **One capability per commit, verbatim**, callers updated in the same commit.
   Order by isolation: `pid_alive`, `read_at`/`write_all_at`, `link_blob`, the
   directory helpers, `available_bytes`.
3. **`FileLock` (15 call sites).** Not verbatim — the hand-rolled FFI is
   replaced with the `windows` crate. Its own reviewed change.
4. **Signal handlers.** Not verbatim — the callback parameter is a signature
   change.
5. Settle the `available_bytes` asymmetry. **This step needs a design decision
   before it can start**, not during: pick either "return clamp provenance in
   the type" or "document per-platform at the arena-sizing call site", and say
   which in the commit message. macOS returning `None` is part of that decision.
6. Metal gating (Problem B) — independent, can run in parallel with all of the
   above.
7. Land the enforcement test last, when the tree is already clean.

A "verbatim" commit is checkable, so check it: its diff should contain only path
and import changes plus the new module's contents: no altered conditions, no
changed error text, no reordered operations. If a reviewer cannot confirm that
by reading the diff, it was not a verbatim move and should be split.

## Verification

Moving code adds no tests. The value is that the platform surface becomes small
enough to lint and review — the lint is what catches drift, so it is the
acceptance criterion.

- **The cross-target lints from [`ci-matrix.md`](ci-matrix.md) gate this work.**
  After the prerequisites and the move,
  `cargo clippy --workspace --target x86_64-pc-windows-gnu -- -D warnings` must
  be green and stay green.
- **An enforcement test** asserting no `cfg(windows|unix|target_os = ...)`
  outside `infr-plat`, `infr-metal`, and the recorded `p2p.rs`/`tp_sem.rs`
  exception. Write it so it can fail: add a stray gate, watch it go red, remove
  it. A grep-based test whose path glob matches nothing is the classic vacuous
  check — and `infr-core/tests/no_bare_print.rs` already carries a comment about
  exactly that miscounting trap, so the precedent for getting this wrong is
  in-tree.
- **Testing the crate itself.** `infr-plat` is the one crate whose entire
  purpose is per-platform behaviour, so unit tests on the dev box exercise one
  arm of everything it owns. It needs the three-platform `test` matrix from
  `ci-matrix.md` to mean anything, and `file_lock_is_exclusive`
  (`download.rs:766`, today `#[cfg(not(target_os = "windows"))]`) moves here and
  needs a Windows-shaped equivalent. **`infr-plat` without the CI matrix is a
  tidier place to keep untested code.**
- **State an `unsafe` policy.** This crate concentrates nearly all the tree's
  FFI — `libc::flock`, `libc::kill`, `sigaction`, `LockFileEx`/`UnlockFileEx`.
  Some moved code already carries SAFETY comments (`kernel_cache.rs:139`,
  `main.rs:525`); requiring that every `unsafe` block arrives with one, and that
  none regress in the move, is a cheap acceptance criterion.

## Open questions

- **Crate or module?** The decision is the crate; the module remains a
  defensible fallback, and is the cheaper option for a payload this size.
- **`infr-plat`'s error type.** As a leaf it cannot use `infr-core`'s. Plain
  `std::io::Result` for the IO surface, `Option` for the queries, matching
  today's signatures.
- **`infr-build` for shared build scripts.** The duplication is real today (four
  byte-identical copies, and Option A would add more). Whether to fold them into
  a build-dependency crate is a follow-up, deliberately **not** part of this
  work — a second new crate to deduplicate a five-line script is exactly the
  scope creep this plan should not carry.
