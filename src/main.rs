//! Runnable CPU demo for the ORDA distillation loss.
//!
//! `cargo run` exercises the full reference pipeline (no GPU needed) for a small
//! tied-teacher distillation step and prints the loss components and gradient
//! magnitudes. For the GPU path see `docs/BLOCKERS.md`.

use orda_cutile_rs::config::KernelConfig;
use orda_cutile_rs::{distillation_loss, is_available, LossOptions, Mat, Profile, Teacher};

/// Deterministic pseudo-random fill in `[-scale, scale)` (small LCG; no deps).
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32); // [0,1)
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn l2(m: &Mat) -> f32 {
    m.data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn main() {
    let (bt, h, v) = (32usize, 64usize, 128usize);

    let student = Mat::from_vec(bt, h, fill(bt * h, 1, 0.2));
    let teacher_hidden = Mat::from_vec(bt, h, fill(bt * h, 2, 0.2));
    let weight = Mat::from_vec(v, h, fill(v * h, 3, 0.05));
    let labels: Vec<i64> = (0..bt).map(|i| (i * 7 % v) as i64).collect();

    let teacher = Teacher::Tied {
        hidden: teacher_hidden,
    };

    let opts = LossOptions {
        student_ce_weight: 1.0,
        teacher_ce_weight: None, // -> 1.0 for Tied
        kd_weight: 0.4,
        temperature: 1.5,
        config: KernelConfig::from_profile(Profile::Balanced),
        ..Default::default()
    };

    let out = distillation_loss(&student, &weight, &labels, &teacher, &opts)
        .expect("distillation_loss failed");

    println!("ORDA distillation loss (CPU reference)");
    println!("  GPU backend available : {}", is_available());
    println!("  shapes                : BT={bt}, H={h}, V={v}");
    println!(
        "  chunk plan            : {} chunks x {} rows",
        out.chunk_plan.num_chunks, out.chunk_plan.chunk_size
    );
    println!("  loss (total)          : {:.6}", out.loss);
    println!("  student_ce            : {:.6}", out.student_ce);
    println!("  teacher_ce            : {:.6}", out.teacher_ce);
    println!("  kl (x T^2)            : {:.6}", out.kl);
    println!(
        "  |grad student_hidden| : {:.6}",
        l2(&out.grad_student_hidden)
    );
    println!("  |grad weight|         : {:.6}", l2(&out.grad_weight));
    if let Some(g) = &out.grad_teacher_hidden {
        println!("  |grad teacher_hidden| : {:.6}", l2(g));
    }
}
