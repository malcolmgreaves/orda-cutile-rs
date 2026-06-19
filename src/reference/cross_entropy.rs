//! CPU reference for the fused cross-entropy forward+backward kernel.
//!
//! Mirrors upstream `ops/kernels.py::_exact_ce_fwdbwd_kernel_merged`. For one
//! logit row and its target it returns the per-row NLL loss and the CE gradient
//! `(softmax(x) − onehot(y))`, including the label-smoothing variant
//! `(p − α/V) − [j==y]·(1−α)`, scaled by the per-row weight. Ignored rows yield
//! zero loss and zero gradient (the kernel zeroes the scale).
//!
//! Math is done in `f64` for oracle-grade accuracy.

/// Compute `(nll, grad)` for a single row.
///
/// * `x` — logits for the row, length `V`.
/// * `y` — target class id (or `ignore_index`).
/// * `label_smoothing` — `α` in `[0, 1]`.
/// * `scale` — the per-row CE weight (`student_ce_weight` or `teacher_ce_weight`).
///
/// `grad[j] = scale · ((p_j − α/V) − [j==y]·(1−α))` with `p = softmax(x)`.
/// For ignored rows, `nll = 0` and `grad = 0`.
pub fn ce_fwdbwd_row(
    x: &[f32],
    y: i64,
    ignore_index: i64,
    label_smoothing: f32,
    scale: f32,
) -> (f32, Vec<f32>) {
    let v = x.len();
    let ignored = y == ignore_index;
    let alpha = label_smoothing as f64;

    // Forward: max, Σexp, optionally Σx for label smoothing.
    let mut m = f64::NEG_INFINITY;
    for &xi in x {
        if (xi as f64) > m {
            m = xi as f64;
        }
    }
    let mut d = 0.0f64;
    let mut sum_x = 0.0f64;
    for &xi in x {
        d += (xi as f64 - m).exp();
        if alpha > 0.0 {
            sum_x += xi as f64;
        }
    }
    let lse = m + d.ln();

    let y_safe = if ignored { 0 } else { y as usize };
    let logit_tgt = x[y_safe] as f64;

    let nll = if alpha > 0.0 {
        lse - (1.0 - alpha) * logit_tgt - (alpha / v as f64) * sum_x
    } else {
        lse - logit_tgt
    };
    let nll = if ignored { 0.0 } else { nll };

    // Backward: grad = scale * ((p - eps) - [j==y](1-α)).
    let eff_scale = if ignored { 0.0 } else { scale as f64 };
    let eps = alpha / v as f64;
    let mut grad = vec![0.0f32; v];
    if eff_scale != 0.0 {
        for (j, gj) in grad.iter_mut().enumerate() {
            let p = (x[j] as f64 - m).exp() / d;
            let mut prob = if alpha > 0.0 { p - eps } else { p };
            if j == y_safe {
                prob -= 1.0 - alpha;
            }
            *gj = (prob * eff_scale) as f32;
        }
    }

    (nll as f32, grad)
}

/// Convenience: just the NLL for a row (no gradient), for tests.
pub fn ce_loss_row(x: &[f32], y: i64, ignore_index: i64, label_smoothing: f32) -> f32 {
    ce_fwdbwd_row(x, y, ignore_index, label_smoothing, 1.0).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(x: &[f32]) -> Vec<f64> {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let d: f64 = x.iter().map(|&xi| (xi as f64 - m).exp()).sum();
        x.iter().map(|&xi| (xi as f64 - m).exp() / d).collect()
    }

    #[test]
    fn nll_equals_negative_log_softmax_of_target() {
        let x = [1.0f32, 2.0, 0.5, -1.0];
        let y = 1i64;
        let nll = ce_loss_row(&x, y, -100, 0.0);
        let p = softmax(&x);
        let expected = -p[y as usize].ln();
        assert!((nll as f64 - expected).abs() < 1e-6, "{nll} vs {expected}");
    }

    #[test]
    fn grad_is_softmax_minus_onehot() {
        let x = [0.3f32, -0.7, 1.1, 0.0];
        let y = 2i64;
        let (_nll, grad) = ce_fwdbwd_row(&x, y, -100, 0.0, 1.0);
        let p = softmax(&x);
        for j in 0..x.len() {
            let mut expected = p[j];
            if j == y as usize {
                expected -= 1.0;
            }
            assert!(
                (grad[j] as f64 - expected).abs() < 1e-6,
                "j={j}: {} vs {expected}",
                grad[j]
            );
        }
        // Gradient sums to ~0 (probabilities sum to 1, minus the single onehot).
        let s: f64 = grad.iter().map(|&g| g as f64).sum();
        assert!(s.abs() < 1e-5, "grad sum {s}");
    }

    #[test]
    fn ignored_row_is_zero() {
        let x = [3.0f32, 1.0, -2.0];
        let (nll, grad) = ce_fwdbwd_row(&x, -100, -100, 0.0, 1.0);
        assert_eq!(nll, 0.0);
        assert!(grad.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn scale_scales_gradient() {
        let x = [0.5f32, -0.5, 1.0];
        let (_n1, g1) = ce_fwdbwd_row(&x, 0, -100, 0.0, 1.0);
        let (_n2, g2) = ce_fwdbwd_row(&x, 0, -100, 0.0, 2.5);
        for j in 0..x.len() {
            assert!((g2[j] - 2.5 * g1[j]).abs() < 1e-6);
        }
    }

    #[test]
    fn label_smoothing_grad_sums_to_zero() {
        let x = [0.2f32, 1.3, -0.4, 0.9, 0.0];
        let (_nll, grad) = ce_fwdbwd_row(&x, 3, -100, 0.1, 1.0);
        // (Σ p - V*eps) - (1-α) = 1 - α - (1-α) = 0.
        let s: f64 = grad.iter().map(|&g| g as f64).sum();
        assert!(s.abs() < 1e-5, "smoothed grad sum {s}");
    }
}
