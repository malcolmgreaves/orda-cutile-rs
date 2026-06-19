//! INT8 row-wise gradient compression, ported from upstream `ops/quant.py`.
//!
//! The weight gradient `grad_W` (`[V, H]`) is compressed row-wise to INT8 with a
//! per-row fp32 scale. Rows corresponding to actual target tokens are kept in
//! full precision (they carry the dominant `−1`-style CE signal), matching
//! `quantize_grad_w`. Two rounding modes are provided:
//!
//! * **deterministic** — round-half-to-even (banker's rounding), like torch's
//!   `.round()`. Simple but biased.
//! * **stochastic** — unbiased in expectation (`E[Q(x)] = x`), better for
//!   convergence at a small RNG cost.

use crate::mat::Mat;

/// A tiny SplitMix64-based RNG yielding reproducible `f32` uniforms in `[0, 1)`.
///
/// Used only for stochastic rounding so results are deterministic given a seed.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 24 bits of mantissa precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

/// Round half to even (banker's rounding), matching torch `.round()`.
#[inline]
pub fn round_half_even(x: f32) -> f32 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else {
        // Exactly .5 -> round to even.
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

/// Per-row INT8 quantization result.
pub struct QuantRows {
    /// Quantized values `[rows * cols]`, row-major.
    pub q: Vec<i8>,
    /// Per-row scale `[rows]`.
    pub scale: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl QuantRows {
    /// Reconstruct the dequantized matrix `q[r,c] * scale[r]`.
    pub fn dequantize(&self) -> Mat {
        let mut out = Mat::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let s = self.scale[r];
            let base = r * self.cols;
            for c in 0..self.cols {
                out.data[base + c] = self.q[base + c] as f32 * s;
            }
        }
        out
    }
}

#[inline]
fn row_scale(row: &[f32]) -> f32 {
    let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    amax.max(1e-12) / 127.0
}

/// Deterministic INT8 quantization (banker's rounding), per row.
pub fn quantize_rowwise_int8(m: &Mat) -> QuantRows {
    let (rows, cols) = (m.rows, m.cols);
    let mut q = vec![0i8; rows * cols];
    let mut scale = vec![0.0f32; rows];
    for r in 0..rows {
        let row = m.row(r);
        let s = row_scale(row);
        scale[r] = s;
        let base = r * cols;
        for c in 0..cols {
            let v = round_half_even(row[c] / s).clamp(-127.0, 127.0);
            q[base + c] = v as i8;
        }
    }
    QuantRows {
        q,
        scale,
        rows,
        cols,
    }
}

/// Stochastic INT8 quantization, unbiased in expectation, per row.
pub fn quantize_rowwise_int8_stochastic(m: &Mat, rng: &mut Rng) -> QuantRows {
    let (rows, cols) = (m.rows, m.cols);
    let mut q = vec![0i8; rows * cols];
    let mut scale = vec![0.0f32; rows];
    for r in 0..rows {
        let row = m.row(r);
        let s = row_scale(row);
        scale[r] = s;
        let base = r * cols;
        for c in 0..cols {
            let scaled = row[c] / s;
            let floor_v = scaled.floor();
            let frac = scaled - floor_v;
            let bump = if frac > rng.next_f32() { 1.0 } else { 0.0 };
            let v = (floor_v + bump).clamp(-127.0, 127.0);
            q[base + c] = v as i8;
        }
    }
    QuantRows {
        q,
        scale,
        rows,
        cols,
    }
}

/// Compressed weight gradient: INT8 body plus exact full-precision target rows.
pub struct GradWQuant {
    pub quant: QuantRows,
    /// The unique target token ids that are kept exact.
    pub unique_targets: Vec<usize>,
    /// Exact rows of `grad_W` for `unique_targets`, row-major `[targets, H]`.
    pub target_rows: Vec<f32>,
}

/// Quantize `grad_W` to INT8 while preserving the rows hit by `target` exactly,
/// mirroring `quantize_grad_w`.
pub fn quantize_grad_w(
    grad_w: &Mat,
    target: &[i64],
    ignore_index: i64,
    stochastic: Option<&mut Rng>,
) -> GradWQuant {
    let h = grad_w.cols;
    let mut unique: Vec<usize> = target
        .iter()
        .copied()
        .filter(|&t| t != ignore_index && t >= 0 && (t as usize) < grad_w.rows)
        .map(|t| t as usize)
        .collect();
    unique.sort_unstable();
    unique.dedup();

    let mut target_rows = Vec::with_capacity(unique.len() * h);
    for &t in &unique {
        target_rows.extend_from_slice(grad_w.row(t));
    }

    let quant = match stochastic {
        Some(rng) => quantize_rowwise_int8_stochastic(grad_w, rng),
        None => quantize_rowwise_int8(grad_w),
    };

    GradWQuant {
        quant,
        unique_targets: unique,
        target_rows,
    }
}

/// Reconstruct `grad_W`, restoring exact target rows, then scaling by the scalar
/// upstream gradient `grad_output`. Mirrors the scalar branch of
/// `dequantize_grad_w`.
pub fn dequantize_grad_w(g: &GradWQuant, grad_output: f32) -> Mat {
    let mut out = g.quant.dequantize();
    let h = out.cols;
    for (i, &t) in g.unique_targets.iter().enumerate() {
        let src = &g.target_rows[i * h..(i + 1) * h];
        out.row_mut(t).copy_from_slice(src);
    }
    crate::mat::scale_assign(&mut out, grad_output);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_even_cases() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-1.5), -2.0);
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
    }

    #[test]
    fn deterministic_roundtrip_bounded_error() {
        // Quantization error per element <= 0.5 * scale.
        let m = Mat::from_vec(2, 4, vec![1.0, -2.0, 0.5, 3.0, -0.1, 0.2, -0.3, 0.05]);
        let q = quantize_rowwise_int8(&m);
        let deq = q.dequantize();
        for r in 0..m.rows {
            let s = q.scale[r];
            for c in 0..m.cols {
                let err = (m.get(r, c) - deq.get(r, c)).abs();
                assert!(err <= 0.5 * s + 1e-6, "err {err} > 0.5*scale {s}");
            }
        }
    }

    #[test]
    fn stochastic_is_unbiased() {
        // Averaging many stochastic quantizations should approach the true value.
        let m = Mat::from_vec(1, 3, vec![0.37, -1.23, 2.71]);
        let mut acc = vec![0.0f64; 3];
        let n = 20_000;
        let mut rng = Rng::new(12345);
        for _ in 0..n {
            let q = quantize_rowwise_int8_stochastic(&m, &mut rng);
            let deq = q.dequantize();
            for c in 0..3 {
                acc[c] += deq.get(0, c) as f64;
            }
        }
        for c in 0..3 {
            let mean = (acc[c] / n as f64) as f32;
            assert!(
                (mean - m.get(0, c)).abs() < 1e-2,
                "col {c}: {mean} vs {}",
                m.get(0, c)
            );
        }
    }

    #[test]
    fn grad_w_target_rows_are_exact() {
        let grad_w = Mat::from_vec(
            4,
            2,
            vec![10.0, -10.0, 0.123, 0.456, -7.0, 7.0, 0.01, -0.02],
        );
        let target = vec![0i64, 2, -100];
        let gq = quantize_grad_w(&grad_w, &target, -100, None);
        assert_eq!(gq.unique_targets, vec![0, 2]);
        let deq = dequantize_grad_w(&gq, 1.0);
        // Target rows (0 and 2) reconstructed exactly.
        assert_eq!(deq.row(0), grad_w.row(0));
        assert_eq!(deq.row(2), grad_w.row(2));
        // Non-target rows are approximate but close.
        for c in 0..2 {
            assert!((deq.get(1, c) - grad_w.get(1, c)).abs() < 0.02);
        }
    }
}
