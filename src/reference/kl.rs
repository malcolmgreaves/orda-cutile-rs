//! CPU reference for the forward-KL distillation kernel.
//!
//! Mirrors upstream `reference/kl_python_ref.py` (`kl_python_chunk`) and the
//! Triton kernel `_kl_from_logits_chunk_kernel`. Computes, per student/teacher
//! row pair in a `logits_chunk` (`[2*n_rows, V]`, student rows first):
//!
//! * `KL_row = Σ_v p_t,v · (log p_t,v − log p_s,v)` with `p = softmax(logits/T)`,
//!   reported scaled by `T²` (Hinton temperature scaling), and
//! * the student-side gradient `grad_scale · (p_s − p_t)`.
//!
//! Math is done in `f64` so this serves as a high-accuracy oracle for the fp16
//! GPU kernel. The temperature-scaling derivation is documented in `PLAN.md`.

use crate::mat::Mat;

/// Numerically-stable softmax and log-softmax of `x * inv_t`, in `f64`.
///
/// Returns `(p, log_p)` where `p_j = softmax(x_j / T)` and `log_p_j` the matching
/// log-probability computed as `(z_j − m) − log Σ exp(z − m)`.
pub fn softmax_logsoftmax(x: &[f32], inv_t: f64) -> (Vec<f64>, Vec<f64>) {
    let mut m = f64::NEG_INFINITY;
    for &xi in x {
        let z = xi as f64 * inv_t;
        if z > m {
            m = z;
        }
    }
    let mut d = 0.0f64;
    for &xi in x {
        d += ((xi as f64 * inv_t) - m).exp();
    }
    let log_d = d.ln();
    let mut p = Vec::with_capacity(x.len());
    let mut log_p = Vec::with_capacity(x.len());
    for &xi in x {
        let zc = (xi as f64 * inv_t) - m;
        p.push(zc.exp() / d);
        log_p.push(zc - log_d);
    }
    (p, log_p)
}

/// Forward KL of a single (student, teacher) logit row pair, **unscaled** (no
/// `T²`, no weighting). Exposed for unit tests against closed-form values.
pub fn forward_kl_row(student: &[f32], teacher: &[f32], temperature: f64) -> f64 {
    let inv_t = 1.0 / temperature;
    let (_ps, log_ps) = softmax_logsoftmax(student, inv_t);
    let (pt, log_pt) = softmax_logsoftmax(teacher, inv_t);
    let mut kl = 0.0f64;
    for j in 0..student.len() {
        kl += pt[j] * (log_pt[j] - log_ps[j]);
    }
    kl
}

/// Core KL over a chunk.
///
/// Returns `(kl_sum_scaled, grad_kl_student)` where `kl_sum_scaled` is the sum
/// over rows of `KL_row · T²` (not divided — the caller applies any mean) and the
/// gradient (if requested) is `grad_scale · (p_s − p_t)` with
/// `grad_scale = kl_weight · T / grad_denom`.
///
/// Pass `grad_denom = 1.0` for the deferred-mean convention used by the
/// end-to-end path; pass `max(n_non_ignore, 1)` to bake the mean into the
/// gradient (the `kl_python_ref` convention).
pub fn kl_chunk(
    logits_chunk: &Mat,
    targets: &[i64],
    n_rows: usize,
    kl_weight: f32,
    temperature: f32,
    ignore_index: i64,
    grad_denom: f32,
    compute_grad: bool,
) -> (f32, Option<Mat>) {
    assert_eq!(
        logits_chunk.rows,
        2 * n_rows,
        "logits_chunk must have 2*n_rows rows"
    );
    assert_eq!(targets.len(), n_rows, "targets must have n_rows entries");

    let v = logits_chunk.cols;
    let inv_t = 1.0 / temperature as f64;
    let grad_scale = (kl_weight * temperature / grad_denom) as f64;
    let t_sq = (temperature * temperature) as f64;

    let mut kl_accum = 0.0f64;
    let mut grad = if compute_grad {
        Some(Mat::zeros(n_rows, v))
    } else {
        None
    };

    for i in 0..n_rows {
        let ignored = targets[i] == ignore_index;
        let s = logits_chunk.row(i);
        let t = logits_chunk.row(n_rows + i);

        let (ps, log_ps) = softmax_logsoftmax(s, inv_t);
        let (pt, log_pt) = softmax_logsoftmax(t, inv_t);

        let mut kl_row = 0.0f64;
        for j in 0..v {
            kl_row += pt[j] * (log_pt[j] - log_ps[j]);
        }
        if ignored {
            kl_row = 0.0;
        }
        kl_accum += kl_row * t_sq;

        if let Some(g) = grad.as_mut() {
            let grow = g.row_mut(i);
            if ignored {
                for gj in grow.iter_mut() {
                    *gj = 0.0;
                }
            } else {
                for j in 0..v {
                    grow[j] = (grad_scale * (ps[j] - pt[j])) as f32;
                }
            }
        }
    }

    (kl_accum as f32, grad)
}

/// Public reference matching `kl_python_chunk` exactly: the gradient bakes in the
/// mean (`grad_scale = kl_weight · T / max(n_non_ignore, 1)`) and the returned
/// scalar is `Σ_rows KL_row · T²` (the `kl_accum_delta`).
pub fn kl_reference_chunk(
    logits_chunk: &Mat,
    targets: &[i64],
    n_rows: usize,
    kl_weight: f32,
    temperature: f32,
    n_non_ignore: usize,
    ignore_index: i64,
    compute_grad: bool,
) -> (f32, Option<Mat>) {
    let denom = n_non_ignore.max(1) as f32;
    kl_chunk(
        logits_chunk,
        targets,
        n_rows,
        kl_weight,
        temperature,
        ignore_index,
        denom,
        compute_grad,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_have_zero_kl() {
        let row = vec![0.3, 1.2, -0.7, 2.1, 0.0];
        let kl = forward_kl_row(&row, &row, 1.5);
        assert!(kl.abs() < 1e-12, "KL(p||p) should be 0, got {kl}");
    }

    #[test]
    fn kl_is_nonnegative() {
        let s = vec![0.1, 0.2, 0.3, -1.0, 2.0];
        let t = vec![1.0, -0.5, 0.0, 0.7, 0.2];
        for &temp in &[0.5, 1.0, 1.5, 4.0] {
            let kl = forward_kl_row(&s, &t, temp);
            assert!(kl >= -1e-12, "KL must be >= 0, got {kl} at T={temp}");
        }
    }

    #[test]
    fn kl_matches_closed_form_two_class() {
        // For 2 logits, softmax(x/T) reduces to a sigmoid; check against a direct
        // probability-space KL computation.
        let temperature = 2.0f64;
        let inv_t = 1.0 / temperature;
        let s = [0.7f32, -0.4];
        let t = [-0.2f32, 1.1];
        let sig = |a: f32, b: f32| {
            let za = a as f64 * inv_t;
            let zb = b as f64 * inv_t;
            let m = za.max(zb);
            let d = (za - m).exp() + (zb - m).exp();
            ((za - m).exp() / d, (zb - m).exp() / d)
        };
        let (ps0, ps1) = sig(s[0], s[1]);
        let (pt0, pt1) = sig(t[0], t[1]);
        let expected = pt0 * (pt0.ln() - ps0.ln()) + pt1 * (pt1.ln() - ps1.ln());
        let got = forward_kl_row(&s, &t, temperature);
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }

    #[test]
    fn ignored_rows_contribute_zero() {
        // A chunk with one student/teacher pair, target ignored -> kl 0, grad 0.
        let logits = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 0.5, -0.5, 0.0]);
        let (kl, grad) = kl_reference_chunk(&logits, &[-100], 1, 0.4, 1.5, 1, -100, true);
        assert_eq!(kl, 0.0);
        let grad = grad.unwrap();
        assert!(grad.data.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn grad_matches_temperature_scaling() {
        // grad_scale = kl_weight * T / denom; grad = grad_scale*(p_s - p_t).
        let logits = Mat::from_vec(2, 3, vec![0.2, 0.5, -0.1, 1.0, -1.0, 0.3]);
        let (kw, temp) = (0.4f32, 1.5f32);
        let (_kl, grad) = kl_reference_chunk(&logits, &[1], 1, kw, temp, 1, -100, true);
        let grad = grad.unwrap();
        let inv_t = 1.0 / temp as f64;
        let (ps, _) = softmax_logsoftmax(logits.row(0), inv_t);
        let (pt, _) = softmax_logsoftmax(logits.row(1), inv_t);
        let gs = kw * temp / 1.0;
        for j in 0..3 {
            let expected = gs * (ps[j] - pt[j]) as f32;
            assert!((grad.get(0, j) - expected).abs() < 1e-6);
        }
    }
}
