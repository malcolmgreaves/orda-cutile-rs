//! End-to-end validation of the ORDA distillation loss on the CPU reference.
//!
//! The centerpiece is a **finite-difference gradient check**: the analytic
//! gradients the kernels produce are compared against numerical derivatives of an
//! independent `f64` forward (`loss_only_f64`). This proves the gradient formulas
//! — the core of the kernel — without any GPU. We also assert chunk invariance
//! (the deferred-mean design makes chunking exact) and component reporting.

use orda_cutile_rs::api::{forward_loss_f64, loss_only_f64, teacher_logits};
use orda_cutile_rs::config::KernelConfig;
use orda_cutile_rs::{distillation_loss, ChunkSize, LossOptions, Mat, Reduction, Teacher};

/// Deterministic pseudo-random fill in `[-scale, scale)` (LCG, no deps).
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn labels(bt: usize, v: usize, ignore_every: Option<usize>, ignore_index: i64) -> Vec<i64> {
    (0..bt)
        .map(|i| match ignore_every {
            Some(k) if i % k == 0 => ignore_index,
            _ => (i * 7 + 3) as i64 % v as i64,
        })
        .collect()
}

/// Generic gradient check: perturb every element of `param` and compare a central
/// finite difference of `loss_at` against the analytic `analytic` gradient.
///
/// The relative-error guard is only enforced on gradient elements with meaningful
/// magnitude (`> 1e-3`); tiny elements are covered by the absolute tolerance.
fn check_grad<Build>(
    param: &Mat,
    analytic: &Mat,
    eps: f64,
    atol: f64,
    rtol: f64,
    mut loss_at: Build,
) where
    Build: FnMut(&Mat) -> f64,
{
    assert_eq!(param.shape(), analytic.shape(), "grad shape mismatch");
    let mut work = param.clone();
    let mut max_rel = 0.0f64;
    for idx in 0..param.data.len() {
        let base = param.data[idx];
        work.data[idx] = base + eps as f32;
        let lp = loss_at(&work);
        work.data[idx] = base - eps as f32;
        let lm = loss_at(&work);
        work.data[idx] = base; // restore
        let num = (lp - lm) / (2.0 * eps);
        let ana = analytic.data[idx] as f64;

        assert!(
            (ana - num).abs() <= atol + rtol * ana.abs().max(num.abs()),
            "grad mismatch at {idx}: analytic={ana:.6e} numeric={num:.6e}"
        );
        if ana.abs().max(num.abs()) > 1e-3 {
            max_rel = max_rel.max((ana - num).abs() / ana.abs().max(num.abs()));
        }
    }
    assert!(
        max_rel < 0.05,
        "max relative grad error too high: {max_rel}"
    );
}

fn base_opts() -> LossOptions {
    LossOptions {
        student_ce_weight: 1.0,
        kd_weight: 0.5,
        temperature: 1.5,
        ..Default::default()
    }
}

#[test]
fn tied_teacher_gradients_match_finite_difference() {
    let (bt, h, v) = (6usize, 5usize, 7usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 1, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 2, 0.2));
    let t_hidden = Mat::from_vec(bt, h, fill(bt * h, 3, 0.3));
    let labs = labels(bt, v, Some(4), -100);
    let teacher = Teacher::Tied {
        hidden: t_hidden.clone(),
    };
    let opts = LossOptions {
        teacher_ce_weight: Some(1.0),
        label_smoothing: 0.1,
        ..base_opts()
    };

    let out = distillation_loss(&student, &weight, &labs, &teacher, &opts).unwrap();

    // ORDA's KL is teacher-detached: gradients correspond to the loss with the KL
    // teacher distribution held fixed. Compute it at the base parameters.
    let kt = teacher_logits(&teacher, &weight);

    // d/d student_hidden
    check_grad(&student, &out.grad_student_hidden, 1e-3, 1e-3, 1e-2, |s| {
        forward_loss_f64(s, &weight, &labs, &teacher, &opts, Some(&kt)).unwrap()
    });
    // d/d weight (KL teacher detached; teacher CE still flows through W)
    check_grad(&weight, &out.grad_weight, 1e-3, 1e-3, 1e-2, |w| {
        forward_loss_f64(&student, w, &labs, &teacher, &opts, Some(&kt)).unwrap()
    });
    // d/d teacher_hidden (tied: teacher CE active, KL detached)
    let gth = out
        .grad_teacher_hidden
        .as_ref()
        .expect("tied + teacher CE => teacher grad");
    check_grad(&t_hidden, gth, 1e-3, 1e-3, 1e-2, |th| {
        let teach = Teacher::Tied { hidden: th.clone() };
        forward_loss_f64(&student, &weight, &labs, &teach, &opts, Some(&kt)).unwrap()
    });
}

#[test]
fn separate_teacher_gradients_match_finite_difference() {
    let (bt, h, ht, v) = (6usize, 5usize, 4usize, 7usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 11, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 12, 0.2));
    let t_hidden = Mat::from_vec(bt, ht, fill(bt * ht, 13, 0.3));
    let t_weight = Mat::from_vec(v, ht, fill(v * ht, 14, 0.2));
    let labs = labels(bt, v, None, -100);
    let opts = LossOptions {
        teacher_ce_weight: Some(0.7),
        ..base_opts()
    };

    let teacher = Teacher::Separate {
        hidden: t_hidden.clone(),
        weight: t_weight.clone(),
    };
    let out = distillation_loss(&student, &weight, &labs, &teacher, &opts).unwrap();
    let kt = teacher_logits(&teacher, &weight); // detached KL teacher, at base params

    check_grad(&student, &out.grad_student_hidden, 1e-3, 1e-3, 1e-2, |s| {
        forward_loss_f64(s, &weight, &labs, &teacher, &opts, Some(&kt)).unwrap()
    });
    check_grad(&weight, &out.grad_weight, 1e-3, 1e-3, 1e-2, |w| {
        forward_loss_f64(&student, w, &labs, &teacher, &opts, Some(&kt)).unwrap()
    });
    let gth = out
        .grad_teacher_hidden
        .as_ref()
        .expect("teacher hidden grad");
    check_grad(&t_hidden, gth, 1e-3, 1e-3, 1e-2, |th| {
        let teach = Teacher::Separate {
            hidden: th.clone(),
            weight: t_weight.clone(),
        };
        forward_loss_f64(&student, &weight, &labs, &teach, &opts, Some(&kt)).unwrap()
    });
    let gtw = out
        .grad_teacher_weight
        .as_ref()
        .expect("teacher weight grad");
    check_grad(&t_weight, gtw, 1e-3, 1e-3, 1e-2, |tw| {
        let teach = Teacher::Separate {
            hidden: t_hidden.clone(),
            weight: tw.clone(),
        };
        forward_loss_f64(&student, &weight, &labs, &teach, &opts, Some(&kt)).unwrap()
    });
}

#[test]
fn precomputed_teacher_gradients_match_finite_difference() {
    let (bt, h, v) = (6usize, 5usize, 7usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 21, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 22, 0.2));
    let t_logits = Mat::from_vec(bt, v, fill(bt * v, 23, 0.5));
    let labs = labels(bt, v, Some(5), -100);
    let opts = base_opts();

    let teacher = Teacher::Precomputed {
        logits: t_logits.clone(),
    };
    let out = distillation_loss(&student, &weight, &labs, &teacher, &opts).unwrap();

    // Only student-side gradients exist (teacher logits are constants).
    assert!(out.grad_teacher_hidden.is_none());
    assert!(out.grad_teacher_weight.is_none());

    check_grad(&student, &out.grad_student_hidden, 1e-3, 1e-3, 1e-2, |s| {
        loss_only_f64(s, &weight, &labs, &teacher, &opts).unwrap()
    });
    check_grad(&weight, &out.grad_weight, 1e-3, 1e-3, 1e-2, |w| {
        loss_only_f64(&student, w, &labs, &teacher, &opts).unwrap()
    });
}

#[test]
fn pure_student_ce_no_teacher_term() {
    // kd=0, teacher CE weight 0 -> pure student CE; teacher unused.
    let (bt, h, v) = (8usize, 4usize, 6usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 31, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 32, 0.2));
    let labs = labels(bt, v, None, -100);
    let teacher = Teacher::Tied {
        hidden: Mat::zeros(bt, h),
    };
    let opts = LossOptions {
        kd_weight: 0.0,
        teacher_ce_weight: Some(0.0),
        ..Default::default()
    };

    let out = distillation_loss(&student, &weight, &labs, &teacher, &opts).unwrap();
    assert_eq!(out.teacher_ce, 0.0);
    assert_eq!(out.kl, 0.0);
    check_grad(&student, &out.grad_student_hidden, 1e-3, 1e-3, 1e-2, |s| {
        loss_only_f64(s, &weight, &labs, &teacher, &opts).unwrap()
    });
}

#[test]
fn chunking_is_exact_invariant() {
    // The deferred-mean design makes the result independent of chunk size.
    let (bt, h, v) = (16usize, 5usize, 9usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 41, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 42, 0.2));
    let t_hidden = Mat::from_vec(bt, h, fill(bt * h, 43, 0.3));
    let labs = labels(bt, v, Some(3), -100);
    let teacher = Teacher::Tied { hidden: t_hidden };

    let mk = |cs: ChunkSize| {
        let opts = LossOptions {
            chunk_size: cs,
            teacher_ce_weight: Some(1.0),
            ..base_opts()
        };
        distillation_loss(&student, &weight, &labs, &teacher, &opts).unwrap()
    };
    let full = mk(ChunkSize::Fixed(bt));
    let by1 = mk(ChunkSize::Fixed(1));
    let by3 = mk(ChunkSize::Fixed(3));
    let auto = mk(ChunkSize::Auto);

    for other in [&by1, &by3, &auto] {
        assert!(
            (full.loss - other.loss).abs() < 1e-5,
            "loss differs across chunking"
        );
        for (a, b) in full
            .grad_weight
            .data
            .iter()
            .zip(other.grad_weight.data.iter())
        {
            assert!((a - b).abs() < 1e-5, "grad_weight differs across chunking");
        }
        for (a, b) in full
            .grad_student_hidden
            .data
            .iter()
            .zip(other.grad_student_hidden.data.iter())
        {
            assert!(
                (a - b).abs() < 1e-5,
                "grad_student_hidden differs across chunking"
            );
        }
    }
}

#[test]
fn loss_components_and_total_are_consistent() {
    let (bt, h, v) = (10usize, 4usize, 8usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 51, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 52, 0.2));
    let t_hidden = Mat::from_vec(bt, h, fill(bt * h, 53, 0.3));
    let labs = labels(bt, v, None, -100);
    let teacher = Teacher::Tied { hidden: t_hidden };

    let (w_s, w_t, kd) = (1.0f32, 0.8f32, 0.5f32);
    let opts = LossOptions {
        student_ce_weight: w_s,
        teacher_ce_weight: Some(w_t),
        kd_weight: kd,
        temperature: 2.0,
        ..Default::default()
    };
    let out = distillation_loss(&student, &weight, &labs, &teacher, &opts).unwrap();

    // total == w_s*student_ce + w_t*teacher_ce + kd*kl
    let recon = w_s * out.student_ce + w_t * out.teacher_ce + kd * out.kl;
    let total = out.loss;
    assert!((recon - total).abs() < 1e-4, "{recon} vs {total}");

    // forward-only oracle matches the reported total loss.
    let oracle = loss_only_f64(&student, &weight, &labs, &teacher, &opts).unwrap() as f32;
    assert!(
        (oracle - out.loss).abs() < 1e-3,
        "oracle {oracle} vs loss {}",
        out.loss
    );
}

#[test]
fn sum_reduction_is_mean_times_token_count() {
    let (bt, h, v) = (8usize, 4usize, 6usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 61, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 62, 0.2));
    let t_hidden = Mat::from_vec(bt, h, fill(bt * h, 63, 0.3));
    let labs = labels(bt, v, Some(4), -100); // some ignored
    let teacher = Teacher::Tied { hidden: t_hidden };
    let n_valid = labs.iter().filter(|&&t| t != -100).count() as f32;

    let mean = distillation_loss(
        &student,
        &weight,
        &labs,
        &teacher,
        &LossOptions {
            reduction: Reduction::Mean,
            teacher_ce_weight: Some(1.0),
            ..base_opts()
        },
    )
    .unwrap();
    let sum = distillation_loss(
        &student,
        &weight,
        &labs,
        &teacher,
        &LossOptions {
            reduction: Reduction::Sum,
            teacher_ce_weight: Some(1.0),
            ..base_opts()
        },
    )
    .unwrap();

    assert!((sum.loss - mean.loss * n_valid).abs() < 1e-3);
    // Mean gradients are the sum gradients divided by the valid-token count.
    for (a, b) in sum
        .grad_weight
        .data
        .iter()
        .zip(mean.grad_weight.data.iter())
    {
        assert!((a - b * n_valid).abs() < 1e-3);
    }
}

#[test]
fn int8_quant_grad_w_is_close_but_lossy() {
    // V must exceed the number of unique targets so some grad_W rows are actually
    // quantized (target rows are kept exact).
    let (bt, h, v) = (8usize, 4usize, 20usize);
    let student = Mat::from_vec(bt, h, fill(bt * h, 71, 0.3));
    let weight = Mat::from_vec(v, h, fill(v * h, 72, 0.2));
    let t_hidden = Mat::from_vec(bt, h, fill(bt * h, 73, 0.3));
    let labs = labels(bt, v, None, -100);
    let teacher = Teacher::Tied { hidden: t_hidden };

    let exact = distillation_loss(
        &student,
        &weight,
        &labs,
        &teacher,
        &LossOptions {
            teacher_ce_weight: Some(1.0),
            ..base_opts()
        },
    )
    .unwrap();

    let mut cfg = KernelConfig::default();
    cfg.quantize_grad_weight = true;
    let quant = distillation_loss(
        &student,
        &weight,
        &labs,
        &teacher,
        &LossOptions {
            config: cfg,
            teacher_ce_weight: Some(1.0),
            ..base_opts()
        },
    )
    .unwrap();

    // Loss is unaffected (only grad_W storage is quantized); grad_W close but not equal.
    assert!((exact.loss - quant.loss).abs() < 1e-6);
    let mut max_abs = 0.0f32;
    for (a, b) in exact
        .grad_weight
        .data
        .iter()
        .zip(quant.grad_weight.data.iter())
    {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs > 0.0,
        "quantization should change grad_W at least slightly"
    );
    assert!(
        max_abs < 0.05,
        "quantization error unexpectedly large: {max_abs}"
    );
}
