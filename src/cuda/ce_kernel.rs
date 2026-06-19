//! cuTile Rust port of the ORDA fused cross-entropy forward/backward kernel.
//!
//! Mirrors `_exact_ce_fwdbwd_kernel_merged`. One tile-block per logit row. The
//! grid is `2*n_rows` (student rows then teacher rows), or `n_rows` when
//! `STUDENT_ONLY`. Per row it computes the NLL loss and the CE gradient
//! `scale · ((p − α/V) − [j==y]·(1−α))` with `p = softmax(x)`. Ignored rows get
//! zero loss and zero gradient.
//!
//! Writes the gradient to a separate `g` tensor instead of in place (see `super`
//! note). Default 3-pass softmax; reductions in `f32`. Assumes `V % BLOCK == 0`.
//! Host/grid convention follows `rms_norm.rs` (see `kl_kernel.rs`). NOT YET
//! COMPILED — see `docs/BLOCKERS.md`.

#![allow(clippy::too_many_arguments)]

#[cutile::module]
pub mod ce_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn exact_ce_fwdbwd<const V: i32, const BLOCK: i32, const STUDENT_ONLY: i32>(
        g: &mut Tensor<f32, { [1, V] }>, // this block's gradient row
        l: &mut Tensor<f32, { [1] }>,    // this block's per-row NLL
        x: &Tensor<f32, { [-1, V] }>,    // [grid_rows, V] logits
        y: &Tensor<i32, { [-1] }>,       // [n_rows] target ids
        n_rows: i32,
        student_scale: f32,
        teacher_scale: f32,
        label_smoothing: f32,
        ignore_index: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let r = pid.0; // logit row index in [0, grid_rows)
        let num_tiles: i32 = V / BLOCK;
        let v_f: f32 = convert_scalar(V);
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let x_part: Partition<f32, { [1, BLOCK] }> = x.partition(tile_shape);

        // Student vs teacher row; resolve target row and per-row scale.
        let is_student: bool = (STUDENT_ONLY == 1) || (r < n_rows);
        let y_row: i32 = if is_student { r } else { r - n_rows };
        let y_part: Partition<i32, { [1] }> = y.partition(const_shape![1]);
        let y_id: i32 = tile_to_scalar(y_part.load([y_row]).reshape(const_shape![]));
        let ignored: bool = y_id == ignore_index;
        let scale_pre: f32 = if is_student {
            student_scale
        } else {
            teacher_scale
        };
        let scale: f32 = if ignored { 0.0f32 } else { scale_pre };

        // ── Pass 1: row max of raw logits ────────────────────────────────────
        let mut m: f32 = -1.0e30f32;
        for j in 0i32..num_tiles {
            let xt: Tile<f32, { [1, BLOCK] }> = x_part.load([r, j]);
            m = max(
                m,
                tile_to_scalar(reduce_max(xt, 1i32).reshape(const_shape![])),
            );
        }

        // ── Pass 2: Σexp(x−m), Σx, and the target logit ──────────────────────
        // The label-smoothing formulas reduce to plain CE when α=0 (eps=0,
        // 1−α=1), so everything is computed branch-free.
        let mut d: f32 = 0.0f32;
        let mut sum_x: f32 = 0.0f32;
        let mut logit_tgt: f32 = 0.0f32;
        for j in 0i32..num_tiles {
            let xt: Tile<f32, { [1, BLOCK] }> = x_part.load([r, j]);
            let et: Tile<f32, { [1, BLOCK] }> = exp(xt - m.broadcast(tile_shape));
            d = d + tile_to_scalar(reduce_sum(et, 1i32).reshape(const_shape![]));
            sum_x = sum_x + tile_to_scalar(reduce_sum(xt, 1i32).reshape(const_shape![]));
            // Gather the target logit: exactly one column equals y_id.
            let base: i32 = j * BLOCK;
            let cols: Tile<i32, { [1, BLOCK] }> =
                base.broadcast(tile_shape) + iota(const_shape![BLOCK]).reshape(tile_shape);
            let is_tgt: Tile<bool, { [1, BLOCK] }> = eq_tile(cols, y_id.broadcast(tile_shape));
            let zero_tile: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, tile_shape);
            let picked: Tile<f32, { [1, BLOCK] }> = select(is_tgt, xt, zero_tile);
            logit_tgt =
                logit_tgt + tile_to_scalar(reduce_sum(picked, 1i32).reshape(const_shape![]));
        }

        let log_d: f32 = tile_to_scalar(log(scalar_to_tile(d)).reshape(const_shape![]));
        let lse: f32 = m + log_d;
        let one_minus_a: f32 = 1.0f32 - label_smoothing;
        let nll_full: f32 = lse - one_minus_a * logit_tgt - (label_smoothing / v_f) * sum_x;
        let nll: f32 = if ignored { 0.0f32 } else { nll_full };
        l.store(scalar_to_tile(nll).reshape(const_shape![1]));

        // ── Pass 3: gradient = scale·((p − α/V) − [j==y]·(1−α)) ──────────────
        let inv_d: f32 = 1.0f32 / d;
        let eps: f32 = label_smoothing / v_f; // 0 when no smoothing
        let mut g_part: PartitionMut<f32, { [1, BLOCK] }> = unsafe { g.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let xt: Tile<f32, { [1, BLOCK] }> = x_part.load([r, j]);
            let p: Tile<f32, { [1, BLOCK] }> =
                exp(xt - m.broadcast(tile_shape)) * inv_d.broadcast(tile_shape);
            let prob: Tile<f32, { [1, BLOCK] }> = p - eps.broadcast(tile_shape);

            // Subtract (1−α) at the target column.
            let base: i32 = j * BLOCK;
            let cols: Tile<i32, { [1, BLOCK] }> =
                base.broadcast(tile_shape) + iota(const_shape![BLOCK]).reshape(tile_shape);
            let is_tgt: Tile<bool, { [1, BLOCK] }> = eq_tile(cols, y_id.broadcast(tile_shape));
            let prob_tgt: Tile<f32, { [1, BLOCK] }> = prob - one_minus_a.broadcast(tile_shape);
            let prob: Tile<f32, { [1, BLOCK] }> = select(is_tgt, prob_tgt, prob);

            let grad: Tile<f32, { [1, BLOCK] }> = prob * scale.broadcast(tile_shape);
            unsafe { g_part.store(grad, [0i32, j]) };
        }
    }
}

pub use ce_module::exact_ce_fwdbwd;
