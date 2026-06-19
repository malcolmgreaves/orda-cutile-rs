//! # orda-cutile-rs
//!
//! A Rust port of the **ORDA** fused Cross-Entropy + forward-KL knowledge-
//! distillation loss kernel ([upstream, Triton/PyTorch][orda]), targeting
//! [cuTile Rust][cutile] for GPU execution.
//!
//! ## What's here
//!
//! * A faithful, dependency-free **CPU reference** of the whole pipeline — KL and
//!   CE kernels, chunk resolver, three teacher modes, GEMM backward, and INT8
//!   weight-gradient quantization — runnable and tested anywhere.
//! * The real **cuTile Rust GPU kernels** in [`cuda`], behind the `cuda` feature.
//!
//! ## GPU support is feature-gated
//!
//! The cuTile stack cannot even *compile* without a CUDA toolkit (its
//! `cuda-bindings` build script requires one), so the GPU path lives behind
//! `--features cuda` and is documented as a hardware blocker in
//! `docs/BLOCKERS.md`. Everything else builds and runs without a GPU.
//!
//! ## Quick start (CPU)
//!
//! ```
//! use orda_cutile_rs::{distillation_loss, LossOptions, Mat, Teacher};
//!
//! let bt = 4; let h = 8; let v = 16;
//! let student = Mat::from_vec(bt, h, vec![0.01; bt * h]);
//! let weight  = Mat::from_vec(v, h, vec![0.02; v * h]);
//! let teacher = Teacher::Tied { hidden: Mat::from_vec(bt, h, vec![0.015; bt * h]) };
//! let labels  = vec![1i64, 2, 3, 4];
//!
//! let opts = LossOptions { kd_weight: 0.4, temperature: 1.5, ..Default::default() };
//! let out = distillation_loss(&student, &weight, &labels, &teacher, &opts).unwrap();
//! assert!(out.loss.is_finite());
//! ```
//!
//! [orda]: https://github.com/hiwuhgds-pixel/ORDA-Knowledge-Distillation-Kernel
//! [cutile]: https://github.com/NVlabs/cutile-rs

// Numeric kernels read clearest with explicit indexed loops and carry many
// parameters (matching the upstream Triton signatures).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod api;
pub mod config;
pub mod mat;
pub mod quant;
pub mod reference;
pub mod resolver;
pub mod teacher;

#[cfg(feature = "cuda")]
pub mod cuda;

// ── Curated public surface ──────────────────────────────────────────────────
pub use api::{
    distillation_loss, forward_loss_f64, loss_only_f64, teacher_logits, DistillationLossOutput,
    LossOptions,
};
pub use config::{Backend, KernelConfig, Profile, Reduction};
pub use mat::Mat;
pub use resolver::{ChunkPlan, ChunkSize};
pub use teacher::{Teacher, TeacherMode};

/// Whether a usable GPU (cuTile) backend is compiled in and a device is present.
///
/// Mirrors upstream `is_available()`. In a build without the `cuda` feature this
/// is always `false`, and `Backend::Auto` resolves to the CPU reference.
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
