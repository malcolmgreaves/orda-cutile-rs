//! Minimal row-major `f32` matrix type and the dense linear-algebra helpers used
//! by the CPU reference path.
//!
//! These are deliberately simple (triple-loop GEMMs) — they are a correctness
//! oracle, not a performance target. The GPU path (`src/cuda`) uses cuTile Rust
//! `mma` kernels instead.

/// A dense, row-major matrix of `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    /// Create a matrix from a flat row-major buffer.
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
        Mat { rows, cols, data }
    }

    /// A `rows x cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow row `r` as a slice of length `cols`.
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Mutably borrow row `r`.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

/// `out = a @ bᵀ`, where `a` is `[m, k]` and `b` is `[n, k]` (so `out` is `[m, n]`).
///
/// This is the projection used to turn hidden states `[rows, H]` into logits
/// `[rows, V]` given a weight matrix `W` of shape `[V, H]`.
pub fn matmul_nt(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.cols, "inner dimension mismatch in matmul_nt");
    let (m, k, n) = (a.rows, a.cols, b.rows);
    let mut out = Mat::zeros(m, n);
    for i in 0..m {
        let arow = a.row(i);
        let orow = out.row_mut(i);
        for j in 0..n {
            let brow = b.row(j);
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += arow[p] * brow[p];
            }
            orow[j] = acc;
        }
    }
    out
}

/// `out = a @ b`, where `a` is `[m, k]` and `b` is `[k, n]`.
///
/// Used in the GEMM backward to turn a logit-gradient row `[rows, V]` into a
/// hidden-state gradient `[rows, H]` via the weight `W` of shape `[V, H]`.
pub fn matmul_nn(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows, "inner dimension mismatch in matmul_nn");
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(m, n);
    for i in 0..m {
        let arow = a.row(i);
        let orow = out.row_mut(i);
        for p in 0..k {
            let aip = arow[p];
            if aip == 0.0 {
                continue;
            }
            let brow = b.row(p);
            for j in 0..n {
                orow[j] += aip * brow[j];
            }
        }
    }
    out
}

/// `out = aᵀ @ b`, where `a` is `[k, m]` and `b` is `[k, n]` (so `out` is `[m, n]`).
///
/// Used to accumulate the weight gradient `grad_W += grad_logitsᵀ @ h`, where
/// `grad_logits` is `[rows, V]` and `h` is `[rows, H]`, giving `[V, H]`.
pub fn matmul_tn(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.rows, b.rows, "outer dimension mismatch in matmul_tn");
    let (k, m, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(m, n);
    for p in 0..k {
        let arow = a.row(p);
        let brow = b.row(p);
        for i in 0..m {
            let aip = arow[i];
            if aip == 0.0 {
                continue;
            }
            let orow = out.row_mut(i);
            for j in 0..n {
                orow[j] += aip * brow[j];
            }
        }
    }
    out
}

/// Add `src` into `dst` element-wise (`dst += src`). Shapes must match.
pub fn add_assign(dst: &mut Mat, src: &Mat) {
    assert_eq!(dst.shape(), src.shape(), "shape mismatch in add_assign");
    for (d, s) in dst.data.iter_mut().zip(src.data.iter()) {
        *d += *s;
    }
}

/// Scale every element of `m` by `s` (`m *= s`).
pub fn scale_assign(m: &mut Mat, s: f32) {
    for v in m.data.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_nt_basic() {
        // a = [[1,2,3]], b = [[1,0,0],[0,1,1]] -> a @ bᵀ = [[1, 5]]
        let a = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Mat::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        let c = matmul_nt(&a, &b);
        assert_eq!(c.shape(), (1, 2));
        assert_eq!(c.data, vec![1.0, 5.0]);
    }

    #[test]
    fn matmul_nn_and_tn_consistent() {
        // (aᵀ b) should equal building it the obvious way.
        let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]); // [2,2]
        let b = Mat::from_vec(2, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]); // [2,3]
        let c = matmul_tn(&a, &b); // aᵀ @ b, shape [2,3]
        assert_eq!(c.shape(), (2, 3));
        // row 0 = a[:,0]ᵀ @ b = [1,3] @ [[1,0,1],[0,1,0]] = [1,3,1]
        assert_eq!(c.row(0), &[1.0, 3.0, 1.0]);
        // row 1 = a[:,1]ᵀ @ b = [2,4] @ [[1,0,1],[0,1,0]] = [2,4,2]
        assert_eq!(c.row(1), &[2.0, 4.0, 2.0]);

        let d = matmul_nn(&a, &b); // [2,3]
        assert_eq!(d.row(0), &[1.0, 2.0, 1.0]);
        assert_eq!(d.row(1), &[3.0, 4.0, 3.0]);
    }
}
