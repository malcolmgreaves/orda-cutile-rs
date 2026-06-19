//! Teacher modes, mirroring upstream `TiedTeacher` / `SeparateTeacher` /
//! `PrecomputedTeacher`.
//!
//! All three modes feed the same internal `logits_chunk` layout
//! (`[student rows; teacher rows]`); they differ only in how the teacher logits
//! are produced and which gradient paths exist.

use crate::mat::Mat;

/// How the teacher distribution is produced.
pub enum Teacher {
    /// Shared output head: `teacher_logits = teacher_hidden @ weightᵀ` using the
    /// *same* `weight` as the student. Default `teacher_ce_weight` is `1.0`.
    Tied {
        /// Teacher hidden states `[BT, H]`.
        hidden: Mat,
    },
    /// Separate head: `teacher_logits = teacher_hidden @ teacher_weightᵀ`.
    /// Default `teacher_ce_weight` is `0.0`.
    Separate {
        /// Teacher hidden states `[BT, H_teacher]`.
        hidden: Mat,
        /// Teacher projection weight `[V, H_teacher]`.
        weight: Mat,
    },
    /// Cached logits supplied directly `[BT, V]`; no teacher gradient path.
    /// Default `teacher_ce_weight` is `0.0`.
    Precomputed {
        /// Teacher logits `[BT, V]`.
        logits: Mat,
    },
}

/// Internal discriminant used by validation/dispatch and the chunk-cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeacherMode {
    Tied,
    Separate,
    Precomputed,
}

impl Teacher {
    pub fn mode(&self) -> TeacherMode {
        match self {
            Teacher::Tied { .. } => TeacherMode::Tied,
            Teacher::Separate { .. } => TeacherMode::Separate,
            Teacher::Precomputed { .. } => TeacherMode::Precomputed,
        }
    }

    /// The default `teacher_ce_weight` when the caller passes `None`:
    /// `1.0` for `Tied`, `0.0` for `Separate`/`Precomputed` — matching
    /// `_resolve_teacher_loss_weight` upstream.
    pub fn default_teacher_ce_weight(&self) -> f32 {
        match self {
            Teacher::Tied { .. } => 1.0,
            Teacher::Separate { .. } | Teacher::Precomputed { .. } => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights() {
        let t = Teacher::Tied {
            hidden: Mat::zeros(2, 3),
        };
        assert_eq!(t.mode(), TeacherMode::Tied);
        assert_eq!(t.default_teacher_ce_weight(), 1.0);

        let s = Teacher::Separate {
            hidden: Mat::zeros(2, 3),
            weight: Mat::zeros(5, 3),
        };
        assert_eq!(s.mode(), TeacherMode::Separate);
        assert_eq!(s.default_teacher_ce_weight(), 0.0);

        let p = Teacher::Precomputed {
            logits: Mat::zeros(2, 5),
        };
        assert_eq!(p.mode(), TeacherMode::Precomputed);
        assert_eq!(p.default_teacher_ce_weight(), 0.0);
    }
}
