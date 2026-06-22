Implementation of the ORDA-Knowledge-Distillation-Kernel in cuTile.rs.
=======

A Rust port of the **ORDA** fused Cross-Entropy + forward-KL knowledge-
distillation loss kernel, targeting **cuTile Rust** for GPU execution.

- Upstream kernel: [`ORDA-Knowledge-Distillation-Kernel`](https://github.com/hiwuhgds-pixel/ORDA-Knowledge-Distillation-Kernel) (Triton / PyTorch)
- GPU framework: [`cutile-rs`](https://github.com/NVlabs/cutile-rs) (NVlabs)

> **TL;DR.** The whole pipeline is implemented and the parts that don't need a GPU
> are runnable and tested *now* (including finite-difference gradient checks). The
> actual cuTile Rust GPU kernels are written too, but cannot be compiled on this
> machine (no CUDA toolkit) — that's documented as a blocker with a precise
> resolution in [`docs/BLOCKERS.md`](docs/BLOCKERS.md).

## What ORDA does

ORDA computes a distillation loss `w_s·CE(student) + w_t·CE(teacher) + kd·KL(teacher‖student)·T²`
without ever materializing full `[batch·tokens, vocab]` probability tensors. It
chunks the batch-token dimension, and within each chunk reuses **one** shared
`logits_chunk` buffer (`[2·n_rows, V]`, student rows then teacher rows) for both
the KL kernel and the fused, in-place cross-entropy forward/backward kernel. See
[`PLAN.md`](PLAN.md) for the full algorithm and math.

## What this repo provides

| Component | File(s) | Runs without GPU? |
|---|---|---|
| KL forward + gradient reference | `src/reference/kl.rs` | ✅ |
| Fused CE forward/backward reference | `src/reference/cross_entropy.rs` | ✅ |
| Chunk resolver + OOM doubling schedule | `src/resolver.rs` | ✅ |
| INT8 row-wise grad-W quantization | `src/quant.rs` | ✅ |
| `distillation_loss` (Tied / Separate / Precomputed teachers) | `src/api.rs` | ✅ |
| cuTile Rust **KL kernel** | `src/cuda/kl_kernel.rs` | ⛔ needs CUDA |
| cuTile Rust **CE kernel** | `src/cuda/ce_kernel.rs` | ⛔ needs CUDA |
| GPU host orchestration | `src/cuda/launch.rs` | ⛔ needs CUDA |

The GPU kernels encode exactly the same math as the CPU reference, which is
validated end-to-end.

## Quick start (CPU, no GPU required)

```bash
cargo run          # runs a small tied-teacher distillation step, prints loss + grads
cargo test         # unit + integration tests (incl. finite-difference gradient checks)
cargo clippy       # clean
```

Library usage:

```rust
use orda_cutile_rs::{distillation_loss, LossOptions, Mat, Teacher};

let (bt, h, v) = (2048, 4096, 32000);
let student = Mat::from_vec(bt, h, /* [BT, H] */ vec![0.0; bt * h]);
let weight  = Mat::from_vec(v, h,  /* [V, H]  */ vec![0.0; v * h]);
let teacher = Teacher::Tied { hidden: Mat::from_vec(bt, h, vec![0.0; bt * h]) };
let labels: Vec<i64> = (0..bt).map(|i| (i % v) as i64).collect();

let opts = LossOptions { kd_weight: 0.4, temperature: 1.5, ..Default::default() };
let out = distillation_loss(&student, &weight, &labels, &teacher, &opts).unwrap();

// out.loss, out.student_ce, out.teacher_ce, out.kl
// out.grad_student_hidden, out.grad_weight, out.grad_teacher_hidden, out.grad_teacher_weight
```

## How correctness is established without a GPU

1. **Reference vs analytic** — KL and CE references checked against closed-form
   values and softmax identities (`src/reference/*` unit tests).
2. **Finite-difference gradient checks** — the analytic gradients the kernels
   produce are compared against numerical derivatives of an independent `f64`
   forward, for all three teacher modes, with/without label smoothing and KD
   (`tests/end_to_end.rs`). This validates the gradient formulas — the heart of
   the kernel — independent of any GPU. ORDA's **teacher-detached KL** semantics
   are reproduced and checked.
3. **Chunk invariance** — the chunked path equals the single-shot path (mean is
   deferred), asserted directly.
4. **Resolver parity** — the chunk heuristic reproduces upstream's integer
   outputs (e.g. the README's `BT=8192, V=131072 → 16 chunks × 512 rows`).

The GPU kernels are written to match this validated reference; the remaining step
is to compile and run the GPU correctness gate on hardware (see below).

## GPU path

The GPU backend is behind a Cargo feature:

```bash
cargo build --features cuda    # requires NVIDIA sm_80+ GPU, CUDA 13.3, Rust 1.89+, Linux
```

It **cannot be built natively on this development machine** (macOS, no CUDA
toolkit). The exact blocker, evidence, and a hardware bring-up + validation
checklist are in [`docs/BLOCKERS.md`](docs/BLOCKERS.md).

### Docker (GPU)

A `Dockerfile` (CUDA 13.3 `-devel`, Ubuntu 24.04) bundles the toolkit, so it
**compiles the `cuda` feature anywhere** — including macOS, where it closes the
build blocker:

```bash
docker build -t orda-cutile:cuda .
# Apple Silicon: add --platform linux/amd64 (builds under emulation, slow but valid)
```

Running the kernels needs a real NVIDIA GPU (sm_80+) via the NVIDIA Container
Toolkit — a **Linux + NVIDIA host**, not macOS (no GPU passthrough in Docker
Desktop):

```bash
docker run --rm --gpus all orda-cutile:cuda                            # GPU tests
docker run --rm --gpus all orda-cutile:cuda cargo run --features cuda  # GPU demo
```

## Layout

```
PLAN.md              Implementation plan: upstream review, math, and mapping to cuTile
docs/BLOCKERS.md     GPU blocker + resolution
src/
  mat.rs             Row-major f32 matrix + GEMM helpers (CPU)
  config.rs          KernelConfig, Profile, Reduction, Backend
  teacher.rs         Teacher: Tied | Separate | Precomputed
  resolver.rs        resolve_chunk_size + OOM doubling schedule
  quant.rs           INT8 row-wise quant/dequant + grad_W quant
  reference/         CPU reference kernels (KL, CE) — the oracle
  api.rs             distillation_loss, DistillationLossOutput, f64 forward oracle
  cuda/              [feature = "cuda"] cuTile Rust GPU kernels + launcher
  main.rs            CPU demo (cargo run)
tests/end_to_end.rs  Finite-difference gradient checks, chunk invariance, components
```

## License

Mozilla Public License, v. 2.0.
