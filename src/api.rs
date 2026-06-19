//! Public API: [`distillation_loss`] and friends.
//!
//! This is the Rust analogue of upstream `api.py::distillation_loss` plus the
//! `DistillCEFunction` forward/backward. It runs the full ORDA flow on the CPU
//! reference kernels: chunk over `BT`, build the shared `logits_chunk`, run KL
//! (reading clean logits), run CE (overwriting logits with gradients in place),
//! merge the KL student gradient, and accumulate the GEMM-backward gradients.
//!
//! The mean normalization `1/n_non_ignore` is deferred and applied once at the
//! end (in fp32), exactly as upstream defers it out of the fp16 kernel. Because
//! of that, the chunked result is independent of the chunk size — a property the
//! tests assert directly.

use crate::config::{Backend, KernelConfig, Reduction};
use crate::mat::{self, Mat};
use crate::quant;
use crate::reference::cross_entropy::ce_fwdbwd_row;
use crate::reference::kl::kl_chunk;
use crate::resolver::{resolve_chunk_size, ChunkPlan, ChunkSize};
use crate::teacher::{Teacher, TeacherMode};

/// Options for [`distillation_loss`]. Use [`LossOptions::default`] and override
/// fields, matching the upstream keyword arguments.
#[derive(Clone, Debug)]
pub struct LossOptions {
    /// `lambda_student`: weight on the student CE term. Default `1.0`.
    pub student_ce_weight: f32,
    /// Weight on the teacher CE term. `None` resolves to the teacher-mode default
    /// (`1.0` for `Tied`, `0.0` for `Separate`/`Precomputed`).
    pub teacher_ce_weight: Option<f32>,
    /// `kd_weight`: weight on the forward-KL term. Default `0.0` (no distillation).
    pub kd_weight: f32,
    /// Softmax temperature `T` for KL. Default `1.0`.
    pub temperature: f32,
    /// Target id to ignore in CE/KL. Default `-100`.
    pub ignore_index: i64,
    /// Label smoothing `α` in `[0, 1]`. Default `0.0`.
    pub label_smoothing: f32,
    /// Loss reduction. Default `Mean`.
    pub reduction: Reduction,
    /// Chunking strategy. Default `Auto`.
    pub chunk_size: ChunkSize,
    /// Kernel tuning options.
    pub config: KernelConfig,
    /// Backend selection. Default `Auto` (resolves to CPU in this build).
    pub backend: Backend,
}

impl Default for LossOptions {
    fn default() -> Self {
        LossOptions {
            student_ce_weight: 1.0,
            teacher_ce_weight: None,
            kd_weight: 0.0,
            temperature: 1.0,
            ignore_index: -100,
            label_smoothing: 0.0,
            reduction: Reduction::Mean,
            chunk_size: ChunkSize::Auto,
            config: KernelConfig::default(),
            backend: Backend::Auto,
        }
    }
}

/// Structured loss output. `loss` is the backward-carrying objective; the three
/// component scalars are weight-free reported values (matching upstream, where
/// only `loss` carries gradients and the components are detached). Gradients are
/// returned explicitly (this CPU path always computes them).
#[derive(Clone, Debug)]
pub struct DistillationLossOutput {
    /// Total objective: `w_s·meanNLL_s + w_t·meanNLL_t + kd·meanKL·T²`.
    pub loss: f32,
    /// Reported student CE (weight-free mean/sum NLL).
    pub student_ce: f32,
    /// Reported teacher CE (weight-free; `0` when no teacher CE term).
    pub teacher_ce: f32,
    /// Reported KL (weight-free, includes the `T²` scaling).
    pub kl: f32,
    /// `d(loss)/d(student_hidden)`, shape `[BT, H]`.
    pub grad_student_hidden: Mat,
    /// `d(loss)/d(weight)`, shape `[V, H]`.
    pub grad_weight: Mat,
    /// `d(loss)/d(teacher_hidden)` — `Some` only for `Tied`/`Separate` with a
    /// teacher CE term (KL is teacher-detached).
    pub grad_teacher_hidden: Option<Mat>,
    /// `d(loss)/d(teacher_weight)` — `Some` only for `Separate` with a teacher
    /// CE term.
    pub grad_teacher_weight: Option<Mat>,
    /// The chunk plan actually used (informational).
    pub chunk_plan: ChunkPlan,
}

/// Copy `count` rows of `m` starting at `start` into a fresh matrix.
fn sub_rows(m: &Mat, start: usize, count: usize) -> Mat {
    let cols = m.cols;
    let mut out = Mat::zeros(count, cols);
    out.data
        .copy_from_slice(&m.data[start * cols..(start + count) * cols]);
    out
}

/// Validate shapes and arguments. Returns the resolved `(V, H, BT)`.
fn validate(
    student_hidden: &Mat,
    weight: &Mat,
    labels: &[i64],
    teacher: &Teacher,
    opts: &LossOptions,
) -> Result<(usize, usize, usize), String> {
    opts.config.validate()?;
    let (bt, h) = student_hidden.shape();
    if bt == 0 {
        return Err("student_hidden has 0 rows (BT must be > 0)".into());
    }
    if weight.cols != h {
        return Err(format!(
            "weight must have shape (V, {h}), got ({}, {})",
            weight.rows, weight.cols
        ));
    }
    let v = weight.rows;
    if labels.len() != bt {
        return Err(format!(
            "labels must have length BT={bt}, got {}",
            labels.len()
        ));
    }
    for &t in labels {
        if t != opts.ignore_index && (t < 0 || t as usize >= v) {
            return Err(format!(
                "label {t} out of range [0, {v}) and != ignore_index {}",
                opts.ignore_index
            ));
        }
    }
    if !(0.0..=1.0).contains(&opts.label_smoothing) {
        return Err(format!(
            "label_smoothing must be in [0,1], got {}",
            opts.label_smoothing
        ));
    }
    if opts.kd_weight < 0.0 {
        return Err(format!("kd_weight must be >= 0, got {}", opts.kd_weight));
    }
    if opts.temperature <= 0.0 {
        return Err(format!("temperature must be > 0, got {}", opts.temperature));
    }

    match teacher {
        Teacher::Tied { hidden } => {
            if hidden.shape() != (bt, h) {
                return Err(format!(
                    "Tied teacher hidden must be ({bt}, {h}), got {:?}",
                    hidden.shape()
                ));
            }
        }
        Teacher::Separate { hidden, weight: wt } => {
            if hidden.rows != bt {
                return Err(format!("Separate teacher hidden must have BT={bt} rows"));
            }
            let ht = hidden.cols;
            if wt.shape() != (v, ht) {
                return Err(format!(
                    "Separate teacher weight must be ({v}, {ht}), got {:?}",
                    wt.shape()
                ));
            }
        }
        Teacher::Precomputed { logits } => {
            if logits.shape() != (bt, v) {
                return Err(format!(
                    "Precomputed teacher logits must be ({bt}, {v}), got {:?}",
                    logits.shape()
                ));
            }
        }
    }
    Ok((v, h, bt))
}

/// Compute the ORDA fused CE + forward-KL distillation loss and its gradients.
///
/// On this build (no `cuda` feature) `Backend::Auto` and `Backend::Cpu` use the
/// reference kernels; `Backend::Cuda` returns an error pointing at
/// `docs/BLOCKERS.md`.
pub fn distillation_loss(
    student_hidden: &Mat,
    weight: &Mat,
    labels: &[i64],
    teacher: &Teacher,
    opts: &LossOptions,
) -> Result<DistillationLossOutput, String> {
    let (v, h, bt) = validate(student_hidden, weight, labels, teacher, opts)?;

    match opts.backend {
        Backend::Cuda => {
            return Err(
                "Backend::Cuda requires building with --features cuda on an NVIDIA \
                 sm_80+ GPU with CUDA 13.3. This build has no GPU backend. See \
                 docs/BLOCKERS.md."
                    .into(),
            )
        }
        Backend::Auto | Backend::Cpu => {}
    }

    let mode = teacher.mode();
    let w_s = opts.student_ce_weight;
    let w_t = opts
        .teacher_ce_weight
        .unwrap_or_else(|| teacher.default_teacher_ce_weight());
    if w_t < 0.0 {
        return Err(format!("teacher_ce_weight must be >= 0, got {w_t}"));
    }
    let kd = opts.kd_weight;
    let temperature = opts.temperature;
    let ls = opts.label_smoothing;
    let ignore = opts.ignore_index;

    let student_only = w_t == 0.0;
    let need_teacher = kd > 0.0 || w_t > 0.0;

    let n_non_ignore = labels.iter().filter(|&&t| t != ignore).count();
    let denom = match opts.reduction {
        Reduction::Mean => n_non_ignore.max(1),
        Reduction::Sum => 1,
    };
    let mean_scale = 1.0f32 / denom as f32;

    let plan = resolve_chunk_size(bt, opts.chunk_size, Some(v), opts.config.max_chunks);

    // Teacher hidden dimension (= student H for Tied; the teacher's own H for
    // Separate; irrelevant for Precomputed).
    let ht_dim = match teacher {
        Teacher::Separate { hidden, .. } => hidden.cols,
        _ => h,
    };

    // Output accumulators.
    let mut grad_h_s = Mat::zeros(bt, h);
    let mut grad_w = Mat::zeros(v, weight.cols);
    let mut grad_h_t: Option<Mat> =
        if w_t > 0.0 && matches!(mode, TeacherMode::Tied | TeacherMode::Separate) {
            Some(Mat::zeros(bt, ht_dim))
        } else {
            None
        };
    let mut grad_w_t: Option<Mat> = if w_t > 0.0 && mode == TeacherMode::Separate {
        let ht = match teacher {
            Teacher::Separate { weight, .. } => weight.cols,
            _ => unreachable!(),
        };
        Some(Mat::zeros(v, ht))
    } else {
        None
    };

    let mut sum_nll_s = 0.0f64;
    let mut sum_nll_t = 0.0f64;
    let mut kl_accum = 0.0f64;

    for c in 0..plan.num_chunks {
        let start = c * plan.chunk_size;
        if start >= bt {
            break;
        }
        let end = (start + plan.chunk_size).min(bt);
        let n = end - start;

        let h_s_chunk = sub_rows(student_hidden, start, n);
        let logits_s = mat::matmul_nt(&h_s_chunk, weight); // [n, V]

        // Assemble logits_chunk = [student rows; teacher rows] (teacher optional).
        let total_rows = if need_teacher { 2 * n } else { n };
        let mut logits_chunk = Mat::zeros(total_rows, v);
        logits_chunk.data[..n * v].copy_from_slice(&logits_s.data);

        // Teacher hidden chunk + teacher logits.
        let mut h_t_chunk: Option<Mat> = None;
        if need_teacher {
            let logits_t = match teacher {
                Teacher::Tied { hidden } => {
                    let htc = sub_rows(hidden, start, n);
                    let lt = mat::matmul_nt(&htc, weight);
                    h_t_chunk = Some(htc);
                    lt
                }
                Teacher::Separate { hidden, weight: wt } => {
                    let htc = sub_rows(hidden, start, n);
                    let lt = mat::matmul_nt(&htc, wt);
                    h_t_chunk = Some(htc);
                    lt
                }
                Teacher::Precomputed { logits } => sub_rows(logits, start, n),
            };
            logits_chunk.data[n * v..2 * n * v].copy_from_slice(&logits_t.data);
        }

        // ── KL (reads clean logits, before CE overwrites) ────────────────────
        let mut grad_kl: Option<Mat> = None;
        if kd > 0.0 {
            let (kl_sum_chunk, gkl) = kl_chunk(
                &logits_chunk,
                &labels[start..end],
                n,
                kd,
                temperature,
                ignore,
                1.0, // deferred mean; applied once at the end
                true,
            );
            kl_accum += kl_sum_chunk as f64;
            grad_kl = gkl;
        }

        // ── CE student rows (overwrite logits_chunk with grad in place) ──────
        for r in 0..n {
            let (nll, g) = ce_fwdbwd_row(logits_chunk.row(r), labels[start + r], ignore, ls, w_s);
            sum_nll_s += nll as f64;
            logits_chunk.row_mut(r).copy_from_slice(&g);
        }

        // ── CE teacher rows (only when a teacher CE term exists) ─────────────
        if w_t > 0.0 {
            for r in 0..n {
                let (nll, g) =
                    ce_fwdbwd_row(logits_chunk.row(n + r), labels[start + r], ignore, ls, w_t);
                sum_nll_t += nll as f64;
                logits_chunk.row_mut(n + r).copy_from_slice(&g);
            }
        }

        // ── Merge KL student gradient into the student logit-grad rows ───────
        if let Some(gkl) = &grad_kl {
            for r in 0..n {
                let row = logits_chunk.row_mut(r);
                let krow = gkl.row(r);
                for j in 0..v {
                    row[j] += krow[j];
                }
            }
        }

        // ── GEMM backward ────────────────────────────────────────────────────
        let student_grad_rows = sub_rows(&logits_chunk, 0, n); // [n, V]
        let grad_h_s_chunk = mat::matmul_nn(&student_grad_rows, weight); // [n, H]
        grad_h_s.data[start * h..end * h].copy_from_slice(&grad_h_s_chunk.data);
        mat::add_assign(&mut grad_w, &mat::matmul_tn(&student_grad_rows, &h_s_chunk));

        if w_t > 0.0 {
            let teacher_grad_rows = sub_rows(&logits_chunk, n, n); // [n, V]
            match teacher {
                Teacher::Tied { .. } => {
                    let htc = h_t_chunk.as_ref().unwrap();
                    let ght = mat::matmul_nn(&teacher_grad_rows, weight); // [n, H]
                    grad_h_t.as_mut().unwrap().data[start * ht_dim..end * ht_dim]
                        .copy_from_slice(&ght.data);
                    mat::add_assign(&mut grad_w, &mat::matmul_tn(&teacher_grad_rows, htc));
                }
                Teacher::Separate { weight: wt, .. } => {
                    let htc = h_t_chunk.as_ref().unwrap();
                    let ght = mat::matmul_nn(&teacher_grad_rows, wt); // [n, H_teacher]
                    grad_h_t.as_mut().unwrap().data[start * ht_dim..end * ht_dim]
                        .copy_from_slice(&ght.data);
                    mat::add_assign(
                        grad_w_t.as_mut().unwrap(),
                        &mat::matmul_tn(&teacher_grad_rows, htc),
                    );
                }
                Teacher::Precomputed { .. } => {
                    // Teacher logits are constants: teacher CE is reported but has
                    // no gradient path. Nothing to accumulate.
                }
            }
        }
    }

    // ── Apply deferred mean normalization to all gradients ───────────────────
    mat::scale_assign(&mut grad_h_s, mean_scale);
    mat::scale_assign(&mut grad_w, mean_scale);
    if let Some(g) = grad_h_t.as_mut() {
        mat::scale_assign(g, mean_scale);
    }
    if let Some(g) = grad_w_t.as_mut() {
        mat::scale_assign(g, mean_scale);
    }

    // ── Optional INT8 round-trip of grad_W (models the training-time effect) ─
    if opts.config.quantize_grad_weight {
        let gq = if opts.config.stochastic_rounding {
            let mut rng = quant::Rng::new(opts.config.stochastic_seed.unwrap_or(0));
            quant::quantize_grad_w(&grad_w, labels, ignore, Some(&mut rng))
        } else {
            quant::quantize_grad_w(&grad_w, labels, ignore, None)
        };
        grad_w = quant::dequantize_grad_w(&gq, 1.0);
    }

    let student_ce = (sum_nll_s * mean_scale as f64) as f32;
    let teacher_ce = (sum_nll_t * mean_scale as f64) as f32;
    let kl = (kl_accum * mean_scale as f64) as f32;
    let loss = ((w_s as f64 * sum_nll_s + w_t as f64 * sum_nll_t + kd as f64 * kl_accum)
        * mean_scale as f64) as f32;

    // Silence unused warnings for the student_only flag in this reference path;
    // it documents the upstream optimization (teacher CE grid skipped).
    let _ = student_only;

    Ok(DistillationLossOutput {
        loss,
        student_ce,
        teacher_ce,
        kl,
        grad_student_hidden: grad_h_s,
        grad_weight: grad_w,
        grad_teacher_hidden: grad_h_t,
        grad_teacher_weight: grad_w_t,
        chunk_plan: plan,
    })
}

// ── f64 forward-only oracle (for finite-difference gradient checks) ──────────

/// `h_row @ Wᵀ` computed in `f64` (one logit row).
fn logits_row_f64(h_row: &[f32], w: &Mat) -> Vec<f64> {
    let mut out = vec![0.0f64; w.rows];
    for vrow in 0..w.rows {
        let wr = w.row(vrow);
        let mut acc = 0.0f64;
        for k in 0..h_row.len() {
            acc += h_row[k] as f64 * wr[k] as f64;
        }
        out[vrow] = acc;
    }
    out
}

/// Cross-entropy NLL for one row, in `f64`.
fn nll_f64(logits: &[f64], y: i64, ignore_index: i64, label_smoothing: f64) -> f64 {
    if y == ignore_index {
        return 0.0;
    }
    let v = logits.len();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut d = 0.0;
    let mut sum_x = 0.0;
    for &x in logits {
        d += (x - m).exp();
        if label_smoothing > 0.0 {
            sum_x += x;
        }
    }
    let lse = m + d.ln();
    let logit_tgt = logits[y as usize];
    if label_smoothing > 0.0 {
        lse - (1.0 - label_smoothing) * logit_tgt - (label_smoothing / v as f64) * sum_x
    } else {
        lse - logit_tgt
    }
}

/// Forward KL `Σ p_t (log p_t − log p_s)` for one row pair, in `f64` (unscaled).
fn kl_f64(student: &[f64], teacher: &[f64], temperature: f64) -> f64 {
    let inv_t = 1.0 / temperature;
    let logsm = |x: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let m = x
            .iter()
            .map(|&v| v * inv_t)
            .fold(f64::NEG_INFINITY, f64::max);
        let d: f64 = x.iter().map(|&v| (v * inv_t - m).exp()).sum();
        let log_d = d.ln();
        let p: Vec<f64> = x.iter().map(|&v| (v * inv_t - m).exp() / d).collect();
        let lp: Vec<f64> = x.iter().map(|&v| (v * inv_t - m) - log_d).collect();
        (p, lp)
    };
    let (_ps, lps) = logsm(student);
    let (pt, lpt) = logsm(teacher);
    let mut kl = 0.0;
    for j in 0..student.len() {
        kl += pt[j] * (lpt[j] - lps[j]);
    }
    kl
}

/// Teacher logits `[BT, V]` for the KL term, given the teacher and the student
/// `weight` (used by `Tied`). Exposed so gradient tests can supply a *detached*
/// teacher distribution to [`forward_loss_f64`].
pub fn teacher_logits(teacher: &Teacher, weight: &Mat) -> Mat {
    match teacher {
        Teacher::Tied { hidden } => mat::matmul_nt(hidden, weight),
        Teacher::Separate { hidden, weight: wt } => mat::matmul_nt(hidden, wt),
        Teacher::Precomputed { logits } => logits.clone(),
    }
}

/// Independent (non-chunked) `f64` forward, with an optional **detached** teacher
/// distribution for the KL term.
///
/// * `kl_teacher_logits = None` — the natural, fully-coupled loss. The KL term
///   uses live teacher logits. The returned *value* equals [`distillation_loss`]'s
///   `loss` to `f64` precision (detaching never changes the forward value).
/// * `kl_teacher_logits = Some(t)` — the KL term reads the teacher probabilities
///   from the fixed `t` (`[BT, V]`) instead of recomputing them from teacher
///   params. This mirrors ORDA's **teacher-detached KL** and is what the kernel's
///   gradients correspond to. Teacher CE still uses live teacher logits.
///
/// For finite-difference gradient checks, pass `Some(teacher_logits(teacher,
/// base_weight))` computed at the unperturbed parameters.
pub fn forward_loss_f64(
    student_hidden: &Mat,
    weight: &Mat,
    labels: &[i64],
    teacher: &Teacher,
    opts: &LossOptions,
    kl_teacher_logits: Option<&Mat>,
) -> Result<f64, String> {
    let (_v, _h, bt) = validate(student_hidden, weight, labels, teacher, opts)?;
    if opts.backend == Backend::Cuda {
        return Err("Backend::Cuda is unavailable in this build (see docs/BLOCKERS.md)".into());
    }

    let w_s = opts.student_ce_weight as f64;
    let w_t = opts
        .teacher_ce_weight
        .unwrap_or_else(|| teacher.default_teacher_ce_weight()) as f64;
    let kd = opts.kd_weight as f64;
    let temperature = opts.temperature as f64;
    let ls = opts.label_smoothing as f64;
    let ignore = opts.ignore_index;
    let t_sq = temperature * temperature;

    let n_non_ignore = labels.iter().filter(|&&t| t != ignore).count();
    let denom = match opts.reduction {
        Reduction::Mean => n_non_ignore.max(1),
        Reduction::Sum => 1,
    } as f64;

    let mut sum_nll_s = 0.0;
    let mut sum_nll_t = 0.0;
    let mut sum_kl = 0.0;

    for r in 0..bt {
        let logits_s = logits_row_f64(student_hidden.row(r), weight);
        sum_nll_s += nll_f64(&logits_s, labels[r], ignore, ls);

        let need_teacher = kd > 0.0 || w_t > 0.0;
        if need_teacher {
            // Live teacher logits (used by teacher CE, and by KL when not detached).
            let logits_t_live = match teacher {
                Teacher::Tied { hidden } => logits_row_f64(hidden.row(r), weight),
                Teacher::Separate { hidden, weight: wt } => logits_row_f64(hidden.row(r), wt),
                Teacher::Precomputed { logits } => {
                    logits.row(r).iter().map(|&x| x as f64).collect()
                }
            };
            if w_t > 0.0 {
                sum_nll_t += nll_f64(&logits_t_live, labels[r], ignore, ls);
            }
            if kd > 0.0 && labels[r] != ignore {
                let kl_teacher: Vec<f64> = match kl_teacher_logits {
                    Some(t) => t.row(r).iter().map(|&x| x as f64).collect(),
                    None => logits_t_live.clone(),
                };
                sum_kl += kl_f64(&logits_s, &kl_teacher, temperature) * t_sq;
            }
        }
    }

    Ok((w_s * sum_nll_s + w_t * sum_nll_t + kd * sum_kl) / denom)
}

/// Recompute only the scalar loss, in `f64` (the natural, fully-coupled forward).
/// Matches [`distillation_loss`]'s `loss` value to `f64` precision.
pub fn loss_only_f64(
    student_hidden: &Mat,
    weight: &Mat,
    labels: &[i64],
    teacher: &Teacher,
    opts: &LossOptions,
) -> Result<f64, String> {
    forward_loss_f64(student_hidden, weight, labels, teacher, opts, None)
}
