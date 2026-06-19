//! Chunk-size resolution and the OOM-retry chunk-doubling schedule.
//!
//! Direct port of upstream `utils/resolver.py` and the chunk arithmetic in
//! `utils/dispatcher.py`. This is pure integer/float arithmetic, so it is
//! identical on CPU and GPU and is exhaustively unit-tested for parity with the
//! Python rules.

/// How the caller selects the chunk size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChunkSize {
    /// Let the resolver pick based on the memory-pressure heuristic.
    #[default]
    Auto,
    /// Use a fixed number of rows per chunk (clamped to `BT`).
    Fixed(usize),
}

/// Ceiling division `ceil(a / b)`.
#[inline]
pub fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Map a continuous memory-pressure score to a power-of-two chunk count.
///
/// Mirrors `_chunks_from_raw`: `1` below the `1.5` threshold, otherwise
/// `2^(floor(log2(raw/1.5)) + 1)`.
pub fn chunks_from_raw(raw: f64) -> usize {
    if raw < 1.5 {
        return 1;
    }
    let exp = (raw / 1.5).log2().floor();
    // Guard against pathological shifts; the caller clamps with `min` anyway.
    let exp = (exp as i64 + 1).clamp(0, 60) as u32;
    1usize << exp
}

/// Resolved chunking decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    /// Rows per chunk.
    pub chunk_size: usize,
    /// Number of chunks covering `BT` rows.
    pub num_chunks: usize,
}

/// Compute `(chunk_size, num_chunks)`.
///
/// Faithful port of `resolve_chunk_size(BT, chunk_size_arg, V, max_chunks)`.
/// `bt` must be > 0. For `ChunkSize::Auto` with a known vocab `v`, the
/// memory-pressure heuristic `(BT/1024)·(V/32768)²` (floored by `BT/4096`) is
/// used and clamped by occupancy and the `max_chunks` ceiling.
pub fn resolve_chunk_size(
    bt: usize,
    chunk_size: ChunkSize,
    v: Option<usize>,
    max_chunks: Option<usize>,
) -> ChunkPlan {
    assert!(bt > 0, "BT (batch * sequence length) must be > 0");

    match chunk_size {
        ChunkSize::Fixed(cs) => {
            let cs = cs.min(bt).max(1);
            ChunkPlan {
                chunk_size: cs,
                num_chunks: ceil_div(bt, cs),
            }
        }
        ChunkSize::Auto => {
            let v = match v {
                Some(v) => v,
                None => {
                    return ChunkPlan {
                        chunk_size: bt,
                        num_chunks: 1,
                    }
                }
            };

            let max_useful_chunks = (bt / 512).max(1);
            let max_chunks = max_chunks.unwrap_or(2 * max_useful_chunks);

            let raw_pressure = (bt as f64 / 1024.0) * (v as f64 / 32768.0).powi(2);
            let raw_bt_floor = bt as f64 / 4096.0;
            let raw = raw_pressure.max(raw_bt_floor);

            let num_chunks = chunks_from_raw(raw)
                .min(max_chunks)
                .min(max_useful_chunks)
                .min(bt)
                .max(1);

            ChunkPlan {
                chunk_size: ceil_div(bt, num_chunks),
                num_chunks,
            }
        }
    }
}

/// The effective `max_chunks` ceiling when the caller leaves it unset, matching
/// the dispatcher: `2 * max(1, BT // 512)`.
pub fn default_max_chunks_limit(bt: usize) -> usize {
    2 * (bt / 512).max(1)
}

/// The sequence of `num_chunks` values the OOM-retry dispatcher would try,
/// starting from `start` and doubling until it reaches `BT` or the limit.
///
/// Mirrors the `dynamic_chunk` loop: `num_chunks = min(num_chunks*2, BT, limit)`,
/// stopping once `num_chunks >= BT` or `>= limit`. On the CPU path there is no
/// real OOM, but exposing this lets us unit-test the escalation policy and
/// document the exact resolution behavior for the GPU path.
pub fn chunk_doubling_schedule(bt: usize, start: usize, limit: usize) -> Vec<usize> {
    let limit = limit.max(1).min(bt);
    let mut n = start.clamp(1, limit);
    let mut out = vec![n];
    while n < bt && n < limit {
        n = (n * 2).min(bt).min(limit);
        out.push(n);
        if n >= bt || n >= limit {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_chunking() {
        let p = resolve_chunk_size(1000, ChunkSize::Fixed(256), None, None);
        assert_eq!(p.chunk_size, 256);
        assert_eq!(p.num_chunks, ceil_div(1000, 256)); // 4
                                                       // Fixed larger than BT clamps to BT, single chunk.
        let p = resolve_chunk_size(100, ChunkSize::Fixed(4096), None, None);
        assert_eq!(
            p,
            ChunkPlan {
                chunk_size: 100,
                num_chunks: 1
            }
        );
    }

    #[test]
    fn auto_without_vocab_is_single_chunk() {
        let p = resolve_chunk_size(8192, ChunkSize::Auto, None, None);
        assert_eq!(
            p,
            ChunkPlan {
                chunk_size: 8192,
                num_chunks: 1
            }
        );
    }

    #[test]
    fn chunks_from_raw_matches_python_rule() {
        assert_eq!(chunks_from_raw(0.0), 1);
        assert_eq!(chunks_from_raw(1.49), 1);
        // raw=1.5 -> 2^(floor(log2(1)) +1) = 2^1 = 2
        assert_eq!(chunks_from_raw(1.5), 2);
        // raw=3.0 -> 2^(floor(log2(2)) +1) = 2^2 = 4
        assert_eq!(chunks_from_raw(3.0), 4);
        // raw=6.0 -> 2^(floor(log2(4)) +1) = 2^3 = 8
        assert_eq!(chunks_from_raw(6.0), 8);
    }

    #[test]
    fn auto_large_vocab_example() {
        // README benchmark case: BT=8192, V=131072. raw_pressure dominates.
        // (8192/1024)*(131072/32768)^2 = 8 * 16 = 128 -> chunks_from_raw(128)
        // = 2^(floor(log2(85.33))+1) = 2^(6+1)=128, clamped by max_useful=16 and
        // default max_chunks=32 -> 16. README quotes num_chunks=16, n_rows=512.
        let p = resolve_chunk_size(8192, ChunkSize::Auto, Some(131072), None);
        assert_eq!(p.num_chunks, 16);
        assert_eq!(p.chunk_size, 512);
    }

    #[test]
    fn doubling_schedule_escalates_to_limit() {
        // Start at 4, limit 32 (<BT): 4,8,16,32
        assert_eq!(chunk_doubling_schedule(10_000, 4, 32), vec![4, 8, 16, 32]);
        // Limited by BT.
        assert_eq!(chunk_doubling_schedule(20, 4, 1000), vec![4, 8, 16, 20]);
    }
}
