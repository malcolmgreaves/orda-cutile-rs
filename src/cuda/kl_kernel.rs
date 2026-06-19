//! cuTile Rust port of the ORDA forward-KL kernel.
//!
//! Mirrors `_kl_from_logits_chunk_kernel`. One tile-block per student row; the
//! teacher counterpart is the same column index offset by `n_rows`. Per row it
//! computes:
//!
//! * `KL_row = Σ_v p_t,v · (log p_t,v − log p_s,v)` with `p = softmax(logits/T)`,
//!   stored scaled by `T²`, and
//! * the student gradient `grad_scale · (p_s − p_t)`.
//!
//! Default 3-pass fixed-shift softmax (ORDA `online_softmax=False`); reductions
//! in `f32`. Assumes `V % BLOCK == 0`.
//!
//! Host/grid convention follows `rms_norm.rs`: the mutable outputs are per-row
//! views (`g: [1, V]`, `k: [1]`) and the host partitions them as `[1, V]` / `[1]`
//! over the full `[n_rows, V]` / `[n_rows]` tensors, so the launch grid is
//! `(n_rows, 1, 1)`. The immutable `x` is the full `[2*n_rows, V]` logits indexed
//! by absolute row. NOT YET COMPILED — see `super` docs and `docs/BLOCKERS.md`.

#![allow(clippy::too_many_arguments)]

#[cutile::module]
pub mod kl_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn kl_from_logits_chunk<const V: i32, const BLOCK: i32>(
        g: &mut Tensor<f32, { [1, V] }>, // this block's student-grad row
        k: &mut Tensor<f32, { [1] }>,    // this block's per-row KL loss
        x: &Tensor<f32, { [-1, V] }>,    // [2*n_rows, V] logits (student then teacher)
        y: &Tensor<i32, { [-1] }>,       // [n_rows] target ids
        n_rows: i32,
        inv_t: f32,
        grad_scale: f32,
        t_sq: f32,
        ignore_index: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let i = pid.0; // student row index in [0, n_rows)
        let num_tiles: i32 = V / BLOCK;
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let x_part: Partition<f32, { [1, BLOCK] }> = x.partition(tile_shape);

        // Target / ignore mask for this row.
        let y_part: Partition<i32, { [1] }> = y.partition(const_shape![1]);
        let y_id: i32 = tile_to_scalar(y_part.load([i]).reshape(const_shape![]));
        let ignored: bool = y_id == ignore_index;

        // ── Pass 1: row maxima of z = logits * inv_t ─────────────────────────
        let mut m_s: f32 = -1.0e30f32;
        let mut m_t: f32 = -1.0e30f32;
        for j in 0i32..num_tiles {
            let zs: Tile<f32, { [1, BLOCK] }> = x_part.load([i, j]) * inv_t.broadcast(tile_shape);
            let zt: Tile<f32, { [1, BLOCK] }> =
                x_part.load([i + n_rows, j]) * inv_t.broadcast(tile_shape);
            m_s = max(
                m_s,
                tile_to_scalar(reduce_max(zs, 1i32).reshape(const_shape![])),
            );
            m_t = max(
                m_t,
                tile_to_scalar(reduce_max(zt, 1i32).reshape(const_shape![])),
            );
        }

        // ── Pass 2: denominators d = Σ exp(z − m) ────────────────────────────
        let mut d_s: f32 = 0.0f32;
        let mut d_t: f32 = 0.0f32;
        for j in 0i32..num_tiles {
            let zs: Tile<f32, { [1, BLOCK] }> = x_part.load([i, j]) * inv_t.broadcast(tile_shape);
            let zt: Tile<f32, { [1, BLOCK] }> =
                x_part.load([i + n_rows, j]) * inv_t.broadcast(tile_shape);
            let es: Tile<f32, { [1, BLOCK] }> = exp(zs - m_s.broadcast(tile_shape));
            let et: Tile<f32, { [1, BLOCK] }> = exp(zt - m_t.broadcast(tile_shape));
            d_s = d_s + tile_to_scalar(reduce_sum(es, 1i32).reshape(const_shape![]));
            d_t = d_t + tile_to_scalar(reduce_sum(et, 1i32).reshape(const_shape![]));
        }
        let log_d_s: f32 = tile_to_scalar(log(scalar_to_tile(d_s)).reshape(const_shape![]));
        let log_d_t: f32 = tile_to_scalar(log(scalar_to_tile(d_t)).reshape(const_shape![]));
        let inv_d_s: f32 = 1.0f32 / d_s;
        let inv_d_t: f32 = 1.0f32 / d_t;

        // ── Pass 3: KL sum and gradient ──────────────────────────────────────
        let mut g_part: PartitionMut<f32, { [1, BLOCK] }> = unsafe { g.partition_mut(tile_shape) };
        let mut kl_row: f32 = 0.0f32;
        for j in 0i32..num_tiles {
            let zs: Tile<f32, { [1, BLOCK] }> = x_part.load([i, j]) * inv_t.broadcast(tile_shape);
            let zt: Tile<f32, { [1, BLOCK] }> =
                x_part.load([i + n_rows, j]) * inv_t.broadcast(tile_shape);

            let cs: Tile<f32, { [1, BLOCK] }> = zs - m_s.broadcast(tile_shape);
            let ct: Tile<f32, { [1, BLOCK] }> = zt - m_t.broadcast(tile_shape);
            let p_s: Tile<f32, { [1, BLOCK] }> = exp(cs) * inv_d_s.broadcast(tile_shape);
            let p_t: Tile<f32, { [1, BLOCK] }> = exp(ct) * inv_d_t.broadcast(tile_shape);
            let log_p_s: Tile<f32, { [1, BLOCK] }> = cs - log_d_s.broadcast(tile_shape);
            let log_p_t: Tile<f32, { [1, BLOCK] }> = ct - log_d_t.broadcast(tile_shape);

            let kl_tile: Tile<f32, { [1, BLOCK] }> = p_t * (log_p_t - log_p_s);
            kl_row = kl_row + tile_to_scalar(reduce_sum(kl_tile, 1i32).reshape(const_shape![]));

            let grad: Tile<f32, { [1, BLOCK] }> = (p_s - p_t) * grad_scale.broadcast(tile_shape);
            let zero_tile: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, tile_shape);
            let ignore_mask: Tile<bool, { [1, BLOCK] }> = constant(ignored, tile_shape);
            let grad: Tile<f32, { [1, BLOCK] }> = select(ignore_mask, zero_tile, grad);
            unsafe { g_part.store(grad, [0i32, j]) };
        }

        let kl_scaled: f32 = if ignored { 0.0f32 } else { kl_row * t_sq };
        k.store(scalar_to_tile(kl_scaled).reshape(const_shape![1]));
    }
}

pub use kl_module::kl_from_logits_chunk;
