# CUDA dev environment for the cuTile Rust GPU backend (the `cuda` feature).
#
# The `-devel` image ships nvcc + CUDA headers (which cuda-bindings' build.rs
# feeds to bindgen) AND the runtime libs cuTile needs to JIT-compile tiles, so a
# single `-devel` stage both builds and runs the GPU path. A slim `-runtime`
# base would break execution.
#
# This is a linux/amd64 image. It RESOLVES THE COMPILE BLOCKER anywhere
# (`cargo build --features cuda` needs only the toolkit, not a GPU), but RUNNING
# the kernels needs a real NVIDIA GPU (sm_80+) exposed with `--gpus all`:
#
#   docker build -t orda-cutile:cuda .                 # builds (under emulation on Apple Silicon)
#   docker run --rm --gpus all orda-cutile:cuda        # runs the GPU tests — Linux + NVIDIA host only
#
# macOS cannot run the kernels: no NVIDIA GPU and no CUDA passthrough in Docker
# Desktop's VM. See docs/BLOCKERS.md.
#
# CUDA 13.3 (>= 13.2 required by cuda-bindings); Ubuntu 24.04 (upstream-tested).
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH

# clang/libclang-dev: bindgen needs libclang. build-essential/pkg-config: native
# build deps. curl/ca-certificates: fetch rustup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates pkg-config clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust pinned to Cargo.toml's rust-version = "1.89".
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.89.0 --profile minimal

WORKDIR /app
COPY . .

# Resolves Blockers 1 & 2: compiles the GPU feature against the toolkit (no GPU
# needed here). --locked honors Cargo.lock (cutile 0.2.0 et al).
RUN cargo build --features cuda --locked

# Default run = the GPU test/correctness gate. Requires `--gpus all` at runtime.
CMD ["cargo", "test", "--features", "cuda", "--locked"]
