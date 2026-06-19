# Plan: Implement ORDA Knowledge-Distillation Kernel in Rust with cuTile Rust

This document is the implementation plan, derived from reviewing both upstream
repositories. It is followed by the implementation in this repo.

## 1. What ORDA is (upstream review)

`ORDA-Knowledge-Distillation-Kernel` (Triton/PyTorch) is a **fused Cross-Entropy +
forward-KL knowledge-distillation loss**. It is memory-efficient: instead of
materializing full `[BT, V]` softmax tensors for both student and teacher, it
processes the batch-token dimension `BT` in **chunks**, and reuses one shared
`logits_chunk` buffer (`[2*n_rows, V]`, student rows first, teacher rows second)
for both the KL kernel and the in-place CE forward/backward kernel.

Per chunk the upstream flow is:

1. Project hidden states → `logits_chunk` via one GEMM.
2. **KL kernel** reads *clean* logits → per-row `KL(teacher‖student)` loss and the
   student-side KL gradient `grad_kl_student` (separate buffer).
3. **CE kernel** reads logits → per-row NLL loss, then **overwrites `logits_chunk`
   in place** with the CE gradient (`softmax − onehot`, with optional label
   smoothing), scaled per student/teacher row.
4. Host merges `grad_kl_student` into the student slice of `logits_chunk`.
5. GEMM backward: `grad_h = logits_chunk @ W`, `grad_W += logits_chunkᵀ @ h`.

Key math (verified against `reference/kl_python_ref.py` and `ops/kernels.py`):

- `KL_row = Σ_v p_t,v · (log p_t,v − log p_s,v)`, with `p = softmax(logits / T)`.
- Reported KL loss is scaled by `T²` (Hinton temperature scaling).
- Student KL gradient: `kd · T · (p_s − p_t)`.
- CE gradient: `(softmax(x) − onehot(y))`, label-smoothing variant
  `(p − α/V) − [v==y]·(1−α)`, times the per-row CE weight.
- Mean normalization `1/n_non_ignore` is **deferred** (kept out of the fp16 kernel
  to avoid underflow) and applied once at the end in fp32.

Three **teacher modes**: `Tied` (shared head, concat + single GEMM), `Separate`
(teacher has its own weight), `Precomputed` (teacher logits supplied, no teacher
gradient). Auto **chunk resolver** uses a memory-pressure heuristic
`(BT/1024)·(V/32768)²` and an OOM-retry dispatcher that doubles the chunk count.
Optional **INT8 row-wise quantization** of `grad_W` (deterministic or stochastic
rounding), keeping target rows in full precision.

## 2. What cuTile Rust is (upstream review)

`cutile-rs` writes memory-safe GPU kernels in Rust. A `#[cutile::module]` block
holds `#[cutile::entry()]` kernels written against `cutile::core::*`: tile types
(`Tile<E,{[..]}>`), `reduce_max`/`reduce_sum`, `exp`/`log`/`exp2`, `select`,
`broadcast`, `mma`, partitioned tensor loads (`partition` + `.load([row, j])`),
and in-place `store`. The host partitions the mutable output, then `.sync()` JIT
compiles the captured AST through CUDA Tile IR to a cubin and launches it.

**Hard constraint:** the `cuda-bindings` `build.rs` requires a CUDA 13.2+ toolkit
to even compile (bindgen). On this dev box (macOS, no CUDA) anything depending on
`cutile` cannot build or run. See `docs/BLOCKERS.md`.

## 3. Mapping ORDA → cuTile Rust

| ORDA / Triton | cuTile Rust |
|---|---|
| `tl.program_id(0)` = row | launch grid `(rows,1,1)`, `get_tile_block_id().0` |
| `for start in range(0,V,BLOCK)` masked loads | partition `[1, BLOCK]`, loop `j in 0..V/BLOCK`, `part.load([row,j])` |
| `tl.max`, `tl.sum` over block | `reduce_max`, `reduce_sum` over tile axis 1 |
| online/3-pass softmax | running `(m,d)` scalars across vocab tiles (Milakov) or 3 passes |
| `tl.math.exp2(x*log2e)` fast path | `exp2(x * log2e)`; precise path `exp` |
| in-place grad write to `X` | `PartitionMut` store back into the logits tensor |
| GEMM projection / backward | `mma`-based kernels (see `cutile-examples/gemm.rs`) |

## 4. This repo's implementation

Because the GPU path cannot build here, the crate is split so the **maximum
amount is runnable now** and the GPU kernels are written but feature-gated:

```
src/
  mat.rs            Minimal row-major f32 matrix + GEMM helpers (CPU)
  config.rs         KernelConfig, Profile, Reduction, Backend
  teacher.rs        Teacher: Tied | Separate | Precomputed
  resolver.rs       resolve_chunk_size + OOM-style chunk doubling (port of resolver.py/dispatcher.py)
  quant.rs          INT8 row-wise quant/dequant + grad_W quant (port of quant.py)
  reference/
    cross_entropy.rs  CE fwd/bwd reference (mirrors ops/kernels.py)
    kl.rs             KL fwd/grad reference (mirrors reference/kl_python_ref.py)
  api.rs            distillation_loss (CPU backend), DistillationLossOutput, chunked orchestration
  cuda/             [feature = "cuda"] cuTile Rust GPU kernels (the GPU deliverable)
    kl_kernel.rs      #[cutile::module] KL kernel
    ce_kernel.rs      #[cutile::module] fused CE fwd/bwd kernel
    launch.rs         host orchestration sketch
  lib.rs            crate root + is_available()
  main.rs           runnable CPU demo
tests/              resolver parity, quant, KL/CE reference vs analytic, end-to-end
                    finite-difference gradient checks across all 3 teacher modes
docs/BLOCKERS.md    GPU blocker + exact resolution
```

### Validation strategy without a GPU

1. **Unit parity** — chunk resolver reproduces the exact integer outputs of the
   Python heuristic; INT8 quant round-trips and is unbiased in expectation.
2. **Reference vs analytic** — KL and CE references checked against independently
   derived closed-form values and softmax identities.
3. **Finite-difference gradient checks** — the end-to-end loss is differentiated
   numerically and compared to the analytic gradients the kernels produce, for all
   teacher modes, with/without label smoothing and KD. This proves the gradient
   formulas (the core of the kernel) independent of any GPU.
4. **Chunk invariance** — the chunked path equals the single-shot path bit-for-bit
   (mean is deferred, so chunking is exact).

## 5. Blocker (GPU execution)

The cuTile Rust kernels in `src/cuda/` are the real GPU deliverable but **cannot be
compiled or run on this machine** (no CUDA toolkit). Resolution: build/run on an
NVIDIA sm_80+ GPU with CUDA 13.3 and Rust 1.89+ (`cargo test --features cuda`,
`cargo run --features cuda --example ...`). Details and a hardware bring-up
checklist are in `docs/BLOCKERS.md`.
