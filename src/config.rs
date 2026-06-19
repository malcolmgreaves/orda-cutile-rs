//! Public configuration types: `Profile`, `KernelConfig`, `Reduction`, `Backend`.
//!
//! These mirror the knobs in upstream `api.py` / `KernelConfig`, adapted to Rust.
//! The "fast math" flags map to the Triton `FAST_MATH_EXP/LOG/MUL` constexprs and
//! the `ONLINE_SOFTMAX` flag; on the CPU reference path they do not change the
//! result (it always computes the precise value), but they are threaded through
//! so the GPU path and the public API stay faithful to upstream semantics.

/// Loss reduction over the valid (non-ignored) tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Divide CE/KL sums by the number of non-ignored tokens.
    #[default]
    Mean,
    /// Return summed CE/KL components.
    Sum,
}

/// Execution backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    /// Prefer the GPU (cuTile) path if available, else fall back to the CPU
    /// reference. On this build (no `cuda` feature) `Auto` resolves to `Cpu`.
    #[default]
    Auto,
    /// Force the CPU reference implementation.
    Cpu,
    /// Force the cuTile Rust GPU implementation. Errors if the `cuda` feature is
    /// not compiled in or no GPU is present.
    Cuda,
}

/// Preset bundles, matching upstream `profile=` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Default, numerically careful settings.
    Balanced,
    /// Enable fast-math (approximate `exp2`/native `log`, reciprocal-multiply).
    Fast,
    /// Numerical-reference/debug preset.
    Debug,
}

/// Expert kernel-tuning options. Construct directly or via [`KernelConfig::from_profile`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelConfig {
    /// Use the single-read Milakov online-softmax scan instead of the 3-pass scan.
    pub online_softmax: bool,
    /// Master fast-math switch (drives `fast_math_exp/log/mul`).
    pub fast_math: bool,
    /// INT8 row-wise compression of the weight gradient.
    pub quantize_grad_weight: bool,
    /// Stochastic (unbiased) rounding for INT8 quantization.
    pub stochastic_rounding: bool,
    /// Accumulate the weight gradient in fp32 across chunks.
    pub fp32_grad_weight_accumulation: bool,
    /// Maximum number of chunks; `None` lets the resolver pick.
    pub max_chunks: Option<usize>,
    /// Max elements processed per vocab tile/block (power of two).
    pub max_fused_size: usize,
    /// Seed for reproducible stochastic quantization.
    pub stochastic_seed: Option<u64>,
}

/// Default maximum vocab tile width, mirroring upstream `DEFAULT_MAX_FUSED_SIZE`.
pub const DEFAULT_MAX_FUSED_SIZE: usize = 65536 / 2;

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig::from_profile(Profile::Balanced)
    }
}

impl KernelConfig {
    /// Build a config from a preset, matching upstream `profile=` semantics:
    /// `Fast` enables fast math only; quantization and stochastic rounding remain
    /// explicit opt-ins; `Debug` is the precise numerical-reference preset.
    pub fn from_profile(profile: Profile) -> Self {
        let fast_math = matches!(profile, Profile::Fast);
        KernelConfig {
            online_softmax: false,
            fast_math,
            quantize_grad_weight: false,
            stochastic_rounding: false,
            fp32_grad_weight_accumulation: false,
            max_chunks: None,
            max_fused_size: DEFAULT_MAX_FUSED_SIZE,
            stochastic_seed: None,
        }
    }

    /// Whether the approximate `exp2` path is used (mirrors `FAST_MATH_EXP`).
    pub fn fast_math_exp(&self) -> bool {
        self.fast_math
    }
    /// Whether the native `log` path is used (mirrors `FAST_MATH_LOG`).
    pub fn fast_math_log(&self) -> bool {
        self.fast_math
    }
    /// Whether reciprocal-multiply is used instead of divide (mirrors `FAST_MATH_MUL`).
    pub fn fast_math_mul(&self) -> bool {
        self.fast_math
    }

    /// Validate field invariants (called by the public API before dispatch).
    pub fn validate(&self) -> Result<(), String> {
        if self.max_fused_size < 1 || !is_power_of_two(self.max_fused_size) {
            return Err(format!(
                "max_fused_size must be a power of two >= 1, got {}",
                self.max_fused_size
            ));
        }
        if let Some(mc) = self.max_chunks {
            if mc < 1 {
                return Err(format!("max_chunks must be >= 1, got {mc}"));
            }
        }
        Ok(())
    }
}

/// `true` iff `value` is a power of two (and positive).
pub fn is_power_of_two(value: usize) -> bool {
    value > 0 && (value & (value - 1)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles() {
        assert!(!KernelConfig::from_profile(Profile::Balanced).fast_math);
        assert!(KernelConfig::from_profile(Profile::Fast).fast_math);
        assert!(!KernelConfig::from_profile(Profile::Debug).fast_math);
    }

    #[test]
    fn pow2() {
        for p in [1, 2, 4, 8, 1024, 32768] {
            assert!(is_power_of_two(p));
        }
        for n in [0, 3, 6, 100, 65535] {
            assert!(!is_power_of_two(n));
        }
    }

    #[test]
    fn validate_rejects_bad_fused_size() {
        let mut c = KernelConfig::default();
        c.max_fused_size = 100; // not a power of two
        assert!(c.validate().is_err());
    }
}
