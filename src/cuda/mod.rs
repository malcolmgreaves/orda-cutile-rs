//! cuTile Rust GPU backend (feature `cuda`).
//!
//! # Status: written, not yet compiled or run here
//!
//! This module contains the real GPU deliverable — the ORDA forward-KL and fused
//! cross-entropy kernels expressed in the [cuTile Rust] DSL, mirroring the
//! upstream Triton kernels `_kl_from_logits_chunk_kernel` and
//! `_exact_ce_fwdbwd_kernel_merged`.
//!
//! It is gated behind `--features cuda` because the cuTile stack **cannot be
//! compiled without a CUDA toolkit** (its `cuda-bindings` build script requires
//! one). The development machine for this port has no NVIDIA GPU and no CUDA
//! toolkit, so this code has been written carefully against the documented DSL
//! and the upstream examples (`softmax.rs`, `rms_norm.rs`, `argmax.rs`,
//! `gemm.rs`) but has **not been compiled or executed**. See `docs/BLOCKERS.md`
//! for the exact hardware bring-up and validation steps.
//!
//! The kernel *math* is identical to the CPU reference in [`crate::reference`],
//! which is fully validated (including finite-difference gradient checks). On
//! hardware, the GPU outputs should match that reference within fp16/fp32
//! tolerance.
//!
//! ## Faithfulness note: separate gradient buffer vs in-place
//!
//! Upstream ORDA overwrites the shared `logits_chunk` *in place* with the CE
//! gradient to save HBM traffic. To stay strictly within cuTile's verified
//! read-immutable / write-mutable example patterns, these kernels write the
//! gradient to a *separate* output tensor. That is functionally identical; the
//! in-place optimization can be layered on later with `partition_mut` once
//! validated on hardware. The host orchestration in [`launch`] still preserves
//! the rest of the ORDA design (chunking, single concatenated projection GEMM,
//! KL reading clean logits before CE, KL-gradient merge into the student slice).
//!
//! ## Vocab tiling
//!
//! The kernels process one row per tile-block and loop over the vocabulary in
//! `BLOCK`-wide tiles (the `rms_norm.rs` pattern), so they scale to the large
//! vocabularies (e.g. 128k) ORDA targets. They assume `V % BLOCK == 0`; the host
//! picks `BLOCK` to divide `V` (or pads), matching how `max_fused_size` bounds
//! the Triton block width.
//!
//! [cuTile Rust]: https://github.com/NVlabs/cutile-rs

pub mod ce_kernel;
pub mod kl_kernel;
pub mod launch;

/// Whether a usable CUDA device is present. On hardware this should query the
/// driver (e.g. `cuda_core::Device::new(0)`); here it conservatively reports
/// `true` only if a device can be created.
pub fn is_available() -> bool {
    cuda_core::Device::new(0).is_ok()
}
