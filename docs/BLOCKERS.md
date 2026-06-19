# Blockers: GPU (cuTile Rust) path

This document records what could **not** be run on the development machine, why,
and exactly what is needed to resolve it. The CPU reference path (the bulk of the
port) builds, runs, and is tested here with no GPU; only the cuTile Rust GPU
backend is blocked.

## Summary

| Item | Status | Where |
|---|---|---|
| CPU reference kernels (KL, CE) | ✅ built + tested | `src/reference/` |
| Chunk resolver / dispatcher logic | ✅ built + tested | `src/resolver.rs` |
| INT8 grad-W quantization | ✅ built + tested | `src/quant.rs` |
| End-to-end `distillation_loss` (3 teacher modes) | ✅ built + tested (finite-difference gradient checks) | `src/api.rs`, `tests/end_to_end.rs` |
| cuTile Rust GPU kernels (KL, CE) | ⛔ written, **not compiled/run** | `src/cuda/` |
| GPU host orchestration | ⛔ written, **not compiled/run** | `src/cuda/launch.rs` |

## Blocker 1 (fundamental): cuTile cannot build without a CUDA toolkit

`cutile` depends transitively on `cuda-bindings`, whose `build.rs` runs `bindgen`
against the CUDA headers and **hard-fails** when no CUDA 13.2+ toolkit is found.
From `cuda-bindings/build.rs`:

```rust
fn run() -> Result<(), Box<dyn Error>> {
    ...
    let cuda_toolkit = resolve_cuda_toolkit()?;   // errors if no toolkit
    ...
}

fn find_default_cuda_toolkit() -> Result<PathBuf, Box<dyn Error>> {
    ...
    Err(format!(
        "{CUDA_TOOLKIT_PATH_ENV} is not set, and no CUDA 13.2+ toolkit was found ...",
    ).into())
}
```

and `main()` does `if let Err(error) = run() { eprintln!("{error}"); exit(1); }`.

This development machine is **macOS (darwin) with no CUDA** (CUDA is not available
for macOS at all). Therefore any crate that enables the `cuda` feature — which
pulls in `cutile` → `cuda-bindings` — cannot compile here. This is intrinsic to
cuTile Rust's requirements (NVIDIA GPU `sm_80+`, CUDA 13.3, Linux), not a defect
in this port.

## Blocker 2 (this environment): cargo cannot fetch the cuda-only deps

Independently, this sandboxed environment cannot populate the cargo registry
cache for the additional crates the `cuda` feature pulls in:

```
error: failed to open `~/.cargo/registry/cache/.../linkme-impl-0.3.36.crate`
Caused by: Operation not permitted (os error 1)
```

So `cargo check --features cuda` fails during dependency extraction, before it
even reaches Blocker 1. The default (CPU) build is unaffected because its
dependency set is empty and already resolved.

## What this means

The GPU kernels in `src/cuda/` are written against the documented cuTile DSL and
the upstream example patterns (`softmax.rs`, `rms_norm.rs`, `argmax.rs`,
`gemm.rs`), but they have **not been compiled or executed**. Their math is
identical to the CPU reference in `src/reference/`, which **is** fully validated
(unit tests + finite-difference gradient checks across all three teacher modes).

## Resolution (GPU bring-up checklist)

On a machine that meets cuTile Rust's requirements:

1. **Hardware**: NVIDIA GPU with compute capability `sm_80` or higher.
2. **Software**: Linux (Ubuntu 24.04 tested by upstream), CUDA 13.3 (13.2+
   supported), Rust 1.89+. Set `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3` (or rely
   on default discovery), and ensure normal `~/.cargo` permissions.
3. **Build**: `cargo build --features cuda`.
4. **Run the demo / tests**: `cargo run --features cuda` and `cargo test --features cuda`.

## What specifically needs on-hardware validation

The math is settled (it matches the validated CPU reference); these are the
cuTile-API/runtime details to confirm and tune once it compiles:

1. **Grid mapping.** The kernels declare per-row mutable outputs (`g: [1, V]`,
   `k`/`l: [1]`) and the host partitions them `[1, V]` / `[1]` so the launch grid
   is `(rows, 1, 1)` — the `rms_norm.rs` idiom. Confirm `pid.0` indexes the row
   and `g_part.store(tile, [0, j])` writes the correct row.
2. **`partition_mut` + per-tile store** within a block (used for `g`) — confirm
   the mutable aliasing/token discipline is satisfied (the `unsafe` blocks).
3. **Multiple mutable outputs** (`g` and `k`/`l`) in one entry — attested by
   `argmax.rs`; confirm both partitions agree on the grid.
4. **`V % BLOCK == 0`.** The kernels tile the vocabulary and assume divisibility.
   Pick `BLOCK` to divide `V` (or pad `V`); bound it by `max_fused_size`.
5. **dtype.** The kernels compute in `f32`. Upstream ORDA stores logits/grads in
   fp16 (bf16 on HIP) with fp32 reductions; switch the tensor element type to
   `f16` and `convert_tile` at load/store (as `argmax.rs` does) for parity.
6. **In-place gradient.** To recover ORDA's exact HBM-traffic optimization, write
   the CE gradient back into the logits buffer via `partition_mut` instead of a
   separate `g` (see `src/cuda/mod.rs`). Functionally equivalent; do after #1–#2
   are confirmed.
7. **GEMM.** Wire the projection and backward GEMMs (`mma`) from
   `cutile-examples/gemm.rs` / `cutile-kernels` into `src/cuda/launch.rs`.

## Suggested GPU correctness gate

Add a `#[cfg(feature = "cuda")]` test that, for random inputs, compares the GPU
kernel outputs against `crate::reference` (the CPU oracle) within fp16/fp32
tolerance — the same role `kl_python_ref.py` plays for the upstream Triton
kernel. Because the CPU reference is already finite-difference-validated, passing
this gate validates the GPU kernels end to end.
