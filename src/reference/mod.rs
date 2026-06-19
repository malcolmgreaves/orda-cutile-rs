//! CPU reference implementations of the ORDA kernels.
//!
//! These mirror the upstream Triton kernels (`ops/kernels.py`, `ops/kl_kernel.py`)
//! and the Python reference (`reference/kl_python_ref.py`). They run anywhere (no
//! GPU required) and act as the correctness oracle for the cuTile Rust GPU path.

pub mod cross_entropy;
pub mod kl;

pub use cross_entropy::{ce_fwdbwd_row, ce_loss_row};
pub use kl::{forward_kl_row, kl_chunk, kl_reference_chunk, softmax_logsoftmax};
