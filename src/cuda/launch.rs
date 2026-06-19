//! Host orchestration for the GPU path.
//!
//! The ORDA outer loop — chunk planning ([`crate::resolver`]), the three teacher
//! modes, the single concatenated projection GEMM, the KL-gradient merge into the
//! student slice, and the backward GEMM — is *identical in structure* to the
//! validated CPU path in [`crate::api::distillation_loss`]. The only GPU-specific
//! piece is launching the two kernels over a prepared `logits_chunk`, which is
//! what [`kl_ce_on_logits_chunk`] does. A full GPU `distillation_loss` is the CPU
//! orchestration with:
//!
//! 1. the projection / backward GEMMs run via cuTile `mma` kernels
//!    (see `cutile-examples/gemm.rs` / `cutile-kernels`), and
//! 2. the inner KL+CE step replaced by [`kl_ce_on_logits_chunk`].
//!
//! NOT YET COMPILED OR RUN — this dev box has no CUDA toolkit, so the cuTile
//! stack cannot build here. See `docs/BLOCKERS.md` for the bring-up + validation
//! checklist. The kernel math is identical to [`crate::reference`], which is
//! fully tested on CPU (including finite-difference gradient checks).

#![allow(clippy::too_many_arguments)]

use cutile::prelude::*;

use super::ce_kernel::exact_ce_fwdbwd;
use super::kl_kernel::kl_from_logits_chunk;

/// Outputs of one fused KL+CE chunk step (host-side copies).
pub struct KlCeChunkOutput {
    /// Student KL gradient, row-major `[n_rows, V]`.
    pub grad_kl_student: Vec<f32>,
    /// Per-row KL loss (`× T²`), `[n_rows]`.
    pub kl_per_row: Vec<f32>,
    /// CE gradient, row-major `[grid_rows, V]` (`grid_rows = n_rows` or `2*n_rows`).
    pub grad_ce: Vec<f32>,
    /// Per-row CE NLL, `[grid_rows]`.
    pub ce_loss: Vec<f32>,
}

/// Run the KL kernel then the CE kernel over a prepared `logits_chunk`.
///
/// This mirrors the inner body of the ORDA chunk loop. The KL kernel reads the
/// clean logits first (producing `grad_kl_student` and `kl_per_row`); the CE
/// kernel then produces the CE gradient and loss. The caller merges
/// `grad_kl_student` into the student rows of `grad_ce` and proceeds to the
/// backward GEMM — exactly as the CPU path does.
///
/// * `logits_chunk` — row-major `[2*n_rows, V]` (student rows first).
/// * `targets` — `[n_rows]` class ids.
/// * `block` — vocab tile width; must divide `V`.
/// * `student_only` — when `true`, the CE grid is `n_rows` (teacher CE skipped).
pub fn kl_ce_on_logits_chunk(
    logits_chunk: &[f32],
    targets: &[i32],
    n_rows: usize,
    v: usize,
    block: usize,
    inv_t: f32,
    kl_grad_scale: f32,
    t_sq: f32,
    student_ce_weight: f32,
    teacher_ce_weight: f32,
    label_smoothing: f32,
    ignore_index: i32,
    student_only: bool,
    compute_kl: bool,
) -> Result<KlCeChunkOutput, Error> {
    assert_eq!(v % block, 0, "V must be divisible by BLOCK");
    assert_eq!(
        logits_chunk.len(),
        2 * n_rows * v,
        "logits_chunk must be [2*n_rows, V]"
    );
    assert_eq!(targets.len(), n_rows, "targets must be [n_rows]");

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let grid_rows = if student_only { n_rows } else { 2 * n_rows };

    // Upload inputs. The CE kernel only needs the rows it scores (grid_rows); the
    // KL kernel always needs both halves, so upload the full [2*n_rows, V] buffer.
    let x_full: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(logits_chunk.to_vec()))
        .sync_on(&stream)?
        .reshape(&[2 * n_rows, v])
        .unwrap()
        .into();
    let y_dev: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(targets.to_vec()))
        .sync_on(&stream)?
        .into();

    let gen2 = vec![v.to_string(), block.to_string()];

    // ── KL kernel (reads clean logits) ───────────────────────────────────────
    let (grad_kl_student, kl_per_row) = if compute_kl {
        let g_kl = api::zeros::<f32>(&[n_rows, v])
            .sync_on(&stream)?
            .partition([1, v]);
        let k = api::zeros::<f32>(&[n_rows])
            .sync_on(&stream)?
            .partition([1]);
        // `.sync_on()` returns every kernel argument in order (see
        // `rms_norm.rs`), so the tuple arity matches the parameter count.
        let (g_kl, k, _x, _y, _n, _it, _gs, _ts, _ii) = kl_from_logits_chunk(
            g_kl,
            k,
            x_full.clone(),
            y_dev.clone(),
            n_rows as i32,
            inv_t,
            kl_grad_scale,
            t_sq,
            ignore_index,
        )
        .generics(gen2.clone())
        .sync_on(&stream)?;
        let g_host = g_kl.unpartition().to_host_vec().sync_on(&stream)?;
        let k_host = k.unpartition().to_host_vec().sync_on(&stream)?;
        (g_host, k_host)
    } else {
        (vec![0.0f32; n_rows * v], vec![0.0f32; n_rows])
    };

    // ── CE kernel ─────────────────────────────────────────────────────────────
    let g_ce = api::zeros::<f32>(&[grid_rows, v])
        .sync_on(&stream)?
        .partition([1, v]);
    let l = api::zeros::<f32>(&[grid_rows])
        .sync_on(&stream)?
        .partition([1]);
    let student_only_flag = if student_only { 1 } else { 0 };
    let gen3 = vec![
        v.to_string(),
        block.to_string(),
        student_only_flag.to_string(),
    ];
    let (g_ce, l, _x, _y, _n, _ss, _ts2, _ls, _ii) = exact_ce_fwdbwd(
        g_ce,
        l,
        x_full.clone(),
        y_dev.clone(),
        n_rows as i32,
        student_ce_weight,
        teacher_ce_weight,
        label_smoothing,
        ignore_index,
    )
    .generics(gen3)
    .sync_on(&stream)?;
    let grad_ce = g_ce.unpartition().to_host_vec().sync_on(&stream)?;
    let ce_loss = l.unpartition().to_host_vec().sync_on(&stream)?;

    Ok(KlCeChunkOutput {
        grad_kl_student,
        kl_per_row,
        grad_ce,
        ce_loss,
    })
}
