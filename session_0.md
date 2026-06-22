# Session 0 — Handoff Notes

> Purpose: read this from a fresh start to understand what was done, why, and what's next.

## Goal of this session

Resolve the GPU build blocker in **orda-cutile-rs** by creating a CUDA dev-environment
`Dockerfile`, and answer whether such an image can run on macOS.

## Background / the blocker (from `docs/BLOCKERS.md`)

- The project is a Rust port of the ORDA fused CE + forward-KL knowledge-distillation
  kernel. CPU reference path (`src/reference/`, `src/api.rs`, etc.) builds + is tested.
- The cuTile Rust GPU kernels (`src/cuda/`) are **written but never compiled/run**.
- Root cause: the `cuda` Cargo feature pulls `cutile → cuda-core → cuda-bindings`,
  whose `build.rs` runs bindgen against CUDA headers and **hard-fails without a
  CUDA 13.2+ toolkit**. The dev machine is macOS (no CUDA exists for macOS).
- Deps are crates.io-pinned: `cutile`/`cuda-core`/`cuda-async` = `0.2.0` (see `Cargo.lock`).
- Rust pinned: `rust-version = "1.89"` in `Cargo.toml`. Edition 2021.

## Key conclusion (the user's central question)

A CUDA Dockerfile **can BUILD the `cuda` feature anywhere (incl. macOS)** — bindgen
only needs toolkit headers, not a GPU, and the `-devel` image ships them. This
closes the fundamental blocker.

It **CANNOT RUN the kernels on macOS** — execution needs a real NVIDIA GPU (sm_80+)
via `--gpus all` / NVIDIA Container Toolkit. Macs have no NVIDIA GPU and Docker
Desktop's VM has no CUDA passthrough. Run the same image on a Linux + NVIDIA host.
Note: image is `linux/amd64`; on Apple Silicon it builds under qemu emulation (slow).

## What was changed this session

1. **`Dockerfile`** (new, repo root)
   - Base `nvidia/cuda:13.3.0-devel-ubuntu24.04` (single `-devel` stage — kept the
     toolkit because cuTile JIT-compiles tiles at runtime; a slim `-runtime` base
     would break execution).
   - Installs `build-essential curl ca-certificates pkg-config clang libclang-dev`.
   - Installs Rust `1.89.0` via rustup (matches `Cargo.toml`).
   - `ENV CUDA_TOOLKIT_PATH=/usr/local/cuda`.
   - `RUN cargo build --features cuda --locked`.
   - `CMD ["cargo", "test", "--features", "cuda", "--locked"]`.
2. **`.dockerignore`** (new) — excludes `target/`, `.git/`, `.github/`, `**/*.md`, Docker files.
3. **`docs/BLOCKERS.md`** — added "Resolution via Docker" subsection (build resolves
   Blockers 1 & 2; run still needs a GPU host).
4. **`README.md`** — added "Docker (GPU)" subsection under the GPU path section.

These are uncommitted working-tree changes (branch `main`). Nothing was committed or pushed.

## NOT yet verified / open items

- **Docker build was never executed.** The Docker daemon wasn't running in-session
  (CLI + buildx present, socket dead). So we have NOT confirmed the image actually
  compiles the `cuda` feature.
- **Verify the base image tag exists** on Docker Hub: `nvidia/cuda:13.3.0-devel-ubuntu24.04`.
  If missing, fall back to newest `13.2+`-`devel`-`ubuntu24.04` tag (BLOCKERS.md
  requires only 13.2+).
- **libclang discovery**: if bindgen can't find it, add `ENV LIBCLANG_PATH=/usr/lib/llvm-*/lib`
  to the Dockerfile.

## Next steps

1. Start Docker Desktop, then from repo root:
   ```bash
   docker build --platform linux/amd64 -t orda-cutile:cuda .   # proves the cuda feature compiles
   ```
   Fix the tag / LIBCLANG_PATH items above if the build fails.
2. On a Linux + NVIDIA (sm_80+) host with NVIDIA Container Toolkit:
   ```bash
   docker run --rm --gpus all orda-cutile:cuda                            # GPU tests
   docker run --rm --gpus all orda-cutile:cuda cargo run --features cuda  # GPU demo
   ```
3. (Out of scope last session, but the real validation goal) Add the suggested
   GPU correctness-gate test: compare GPU kernel output vs `crate::reference` (CPU
   oracle) within fp16/fp32 tolerance — see "Suggested GPU correctness gate" in
   `docs/BLOCKERS.md`. The CPU reference is already finite-difference-validated, so
   passing this gate validates the GPU kernels end-to-end.
4. Decide whether to commit these Docker changes (not yet done).

## Useful references

- `docs/BLOCKERS.md` — full blocker description + on-hardware bring-up checklist (#1–#7).
- `PLAN.md` — algorithm/math and cuTile mapping.
- Saved plan for this session: `/Users/malcolmgreaves/.claude/plans/read-all-of-the-logical-bunny.md`.
