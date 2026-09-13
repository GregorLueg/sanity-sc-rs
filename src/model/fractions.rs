//! The stationary point of the log posterior at a fixed variance.
//!
//! SPEC section 2, SI eq. 21-28. Everything here is per gene and dense in the
//! cell axis: a cell with no counts for this gene still contributes through its
//! library size.
//!
//! The stationarity condition is one Wright omega evaluation per cell plus a
//! single scalar root find for the normalisation offset `z`. Writing
//! `u_c = v s w_c` turns SI eq. 25 into `u_c + ln u_c = x_c`, and the constraint
//! `sum_c w_c = 1` becomes `sum_c omega(x_c) = v s`, which is strictly
//! decreasing in `z` and so has a unique root.

use crate::errors::SanityErrors;
use crate::utils::wright_omega::{log_omega, omega_from_log};

/// Relative tolerance on the offset residual `sum_c omega_c - v s`.
///
/// Measured against `v s` rather than absolutely, because `v s` spans many
/// orders of magnitude across genes and variance bins.
const OFFSET_TOL: f64 = 1e-13;

/// Iteration cap for the Newton solve on the offset.
///
/// Newton on a monotone function with an exact derivative, bracketed and warm
/// started from the neighbouring variance bin, settles in a handful of steps.
/// The cap turns a pathological input into an error rather than a hang.
const OFFSET_MAX_ITER: usize = 100;

/// Cap on the bracket expansion for the offset.
///
/// Each step doubles, so this covers a span of `2^60` around the initial guess,
/// far wider than any real data can require.
const OFFSET_MAX_BRACKET: usize = 60;

/// The stationary point of one gene at one variance.
///
/// Holds only what downstream kernels need. `omega` and `log_omega` are indexed
/// by cell; the rest are scalars for the gene.
#[derive(Debug)]
pub(crate) struct Stationary {
    /// The variance this point was solved at.
    pub v: f64,
    /// `K + 1`, the total UMI count of the gene plus one.
    pub s: f64,
    /// `ln(v s)`, cached because every per-cell quantity needs it.
    pub log_vs: f64,
    /// The normalisation offset `z`, with `exp(z) = sum_c T_c exp(d*_c)`.
    pub z: f64,
}

impl Stationary {
    /// The log fold change in one cell.
    ///
    /// SI eq. 28. `d*_c = ln w_c - ln T_c + z` and `ln w_c = t_c - ln(v s)`.
    ///
    /// ### Params
    ///
    /// * `log_omega` - `ln omega(x_c)` for this cell.
    /// * `log_total` - `ln T_c` for this cell.
    ///
    /// ### Returns
    ///
    /// `d*_c` at this variance.
    #[inline(always)]
    pub(crate) fn log_fold_change(&self, log_omega: f64, log_total: f64) -> f64 {
        log_omega - self.log_vs - log_total + self.z
    }

    /// The normalised weight in one cell.
    ///
    /// `w_c = omega_c / (v s)`, summing to one over cells by construction.
    ///
    /// ### Params
    ///
    /// * `omega` - `omega(x_c)` for this cell.
    ///
    /// ### Returns
    ///
    /// `w_c` at this variance.
    #[inline(always)]
    pub(crate) fn weight(&self, omega: f64) -> f64 {
        omega / (self.v * self.s)
    }
}

/// Fill `omega` and `log_omega` for every cell at a given offset.
///
/// `x_c = v k_c + ln T_c + ln(v s) - z`. Warm starting is deliberately not done
/// here: `log_omega` is globally convergent and the cost is dominated by the
/// transcendentals, not the iteration count.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `log_vs` - `ln(v s)`.
/// * `z` - The current offset.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `out_omega` - Output, `omega(x_c)`.
/// * `out_log_omega` - Output, `ln omega(x_c)`.
///
/// ### Returns
///
/// `sum_c omega(x_c)`, which the offset solve drives to `v s`.
fn evaluate(
    v: f64,
    log_vs: f64,
    z: f64,
    counts: &[f64],
    log_totals: &[f64],
    out_omega: &mut [f64],
    out_log_omega: &mut [f64],
) -> f64 {
    let shift = log_vs - z;
    let mut total = 0.0;

    for (((&k, &lt), w), t) in counts
        .iter()
        .zip(log_totals)
        .zip(out_omega.iter_mut())
        .zip(out_log_omega.iter_mut())
    {
        let x = v * k + lt + shift;
        *t = log_omega(x);
        *w = omega_from_log(x, *t);
        total += *w;
    }

    total
}

/// Solve the stationarity condition for one gene at one variance.
///
/// SI eq. 27. Brackets the root of `F(z) = sum_c omega(x_c) - v s` by doubling,
/// then runs Newton with the exact derivative `F'(z) = -sum_c omega_c / (1 + omega_c)`,
/// falling back to bisection whenever a Newton step leaves the bracket.
///
/// ### Params
///
/// * `v` - The variance bin.
/// * `s` - `K + 1` for this gene.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `guess` - Starting offset. Pass the previous bin's solution when sweeping
///   an ordered grid; otherwise `ln(sum_c T_c) + v / 2`.
/// * `omega` - Scratch and output, `omega(x_c)` at the solution.
/// * `log_omega` - Scratch and output, `ln omega(x_c)` at the solution.
///
/// ### Returns
///
/// The stationary point, or [`SanityErrors::FractionSolveDiverged`] if neither
/// the bracket nor the Newton loop settles.
pub(crate) fn solve_stationary(
    v: f64,
    s: f64,
    counts: &[f64],
    log_totals: &[f64],
    guess: f64,
    omega: &mut [f64],
    log_omega: &mut [f64],
) -> Result<Stationary, SanityErrors> {
    let vs = v * s;
    let log_vs = vs.ln();
    let tol = OFFSET_TOL * vs;

    let residual = |z: f64, omega: &mut [f64], log_omega: &mut [f64]| {
        evaluate(v, log_vs, z, counts, log_totals, omega, log_omega) - vs
    };

    // `F` is strictly decreasing in `z`, so a positive residual means `z` is too
    // small. Expand outwards by doubling until the sign flips.
    let mut z = guess;
    let mut f = residual(z, omega, log_omega);
    let (mut lo, mut hi);
    if f > 0.0 {
        lo = z;
        let mut step = 1.0;
        loop {
            hi = lo + step;
            if residual(hi, omega, log_omega) <= 0.0 {
                break;
            }
            lo = hi;
            step *= 2.0;
            if step > (1u64 << OFFSET_MAX_BRACKET) as f64 {
                return Err(SanityErrors::FractionSolveDiverged {
                    iterations: 0,
                    residual: f,
                });
            }
        }
    } else {
        hi = z;
        let mut step = 1.0;
        loop {
            lo = hi - step;
            if residual(lo, omega, log_omega) >= 0.0 {
                break;
            }
            hi = lo;
            step *= 2.0;
            if step > (1u64 << OFFSET_MAX_BRACKET) as f64 {
                return Err(SanityErrors::FractionSolveDiverged {
                    iterations: 0,
                    residual: f,
                });
            }
        }
    }

    z = 0.5 * (lo + hi);
    for iteration in 0..OFFSET_MAX_ITER {
        f = residual(z, omega, log_omega);
        if f.abs() <= tol {
            return Ok(Stationary { v, s, log_vs, z });
        }
        if f > 0.0 { lo = z } else { hi = z }

        // F'(z) = -sum_c omega_c / (1 + omega_c), from d omega / dx = omega / (1 + omega).
        let slope: f64 = omega.iter().map(|&w| w / (1.0 + w)).sum();
        let next = if slope > 0.0 { z + f / slope } else { f64::NAN };

        z = if next.is_finite() && next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };

        if iteration + 1 == OFFSET_MAX_ITER {
            return Err(SanityErrors::FractionSolveDiverged {
                iterations: OFFSET_MAX_ITER,
                residual: f,
            });
        }
    }

    // The loop above always returns.
    unreachable!()
}

/// Re-evaluate the per-cell state at an offset that is already known.
///
/// The second pass of the marginalising rule (SPEC section 6.1) has `z(v_b)`
/// from the first pass, so it needs no iteration at all: one sweep refills
/// `omega` and `log_omega`.
///
/// ### Params
///
/// * `point` - The stationary point from the first pass.
/// * `counts` - Dense UMI counts for this gene, length `n_cells`.
/// * `log_totals` - `ln T_c` for every cell, length `n_cells`.
/// * `omega` - Output, `omega(x_c)`.
/// * `log_omega` - Output, `ln omega(x_c)`.
///
/// ### Returns
///
/// Nothing; `omega` and `log_omega` are overwritten.
pub(crate) fn refresh(
    point: &Stationary,
    counts: &[f64],
    log_totals: &[f64],
    omega: &mut [f64],
    log_omega: &mut [f64],
) {
    evaluate(
        point.v,
        point.log_vs,
        point.z,
        counts,
        log_totals,
        omega,
        log_omega,
    );
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A small deterministic gene: counts, cell totals, and a variance.
    fn fixture() -> (Vec<f64>, Vec<f64>, f64, f64) {
        let counts = vec![0.0, 3.0, 1.0, 0.0, 7.0, 0.0, 2.0, 0.0];
        let totals = vec![1200.0, 980.0, 1500.0, 700.0, 2100.0, 450.0, 1330.0, 1010.0];
        let s = counts.iter().sum::<f64>() + 1.0;
        (counts, totals, s, 0.75)
    }

    #[test]
    fn test_fractions_sum_to_one() {
        let (counts, totals, s, v) = fixture();
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = totals.iter().sum::<f64>().ln() + v / 2.0;

        let point = solve_stationary(
            v,
            s,
            &counts,
            &log_totals,
            guess,
            &mut omega,
            &mut log_omega,
        )
        .expect("the offset solve converges on well formed input");

        let sum: f64 = omega.iter().map(|&w| point.weight(w)).sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn test_stationarity_condition_holds() {
        // The gradient of SI eq. 20 must vanish: -d_c / v + k_c - s w_c = 0.
        let (counts, totals, s, v) = fixture();
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = totals.iter().sum::<f64>().ln();

        let point =
            solve_stationary(v, s, &counts, &log_totals, guess, &mut omega, &mut log_omega)
                .expect("converges");

        for c in 0..counts.len() {
            let d = point.log_fold_change(log_omega[c], log_totals[c]);
            let grad = -d / v + counts[c] - s * point.weight(omega[c]);
            assert!(grad.abs() < 1e-9, "cell {c} gradient {grad:e}");
        }
    }

    #[test]
    fn test_offset_recovers_the_definition() {
        // exp(z) = sum_c T_c exp(d*_c).
        let (counts, totals, s, v) = fixture();
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = 0.0;

        let point =
            solve_stationary(v, s, &counts, &log_totals, guess, &mut omega, &mut log_omega)
                .expect("converges from a deliberately poor guess");

        let lhs: f64 = (0..counts.len())
            .map(|c| totals[c] * point.log_fold_change(log_omega[c], log_totals[c]).exp())
            .sum();
        assert_relative_eq!(lhs, point.z.exp(), max_relative = 1e-11);
    }

    #[test]
    fn test_small_variance_limit() {
        // As v -> 0 the equations collapse to exp(z) = sum_c T_c exp(v k_c).
        let (counts, totals, s, _) = fixture();
        let v = 1e-8;
        let log_totals: Vec<f64> = totals.iter().map(|t| t.ln()).collect();
        let mut omega = vec![0.0; counts.len()];
        let mut log_omega = vec![0.0; counts.len()];
        let guess = totals.iter().sum::<f64>().ln();

        let point =
            solve_stationary(v, s, &counts, &log_totals, guess, &mut omega, &mut log_omega)
                .expect("converges");

        let expected: f64 = (0..counts.len())
            .map(|c| totals[c] * (v * counts[c]).exp())
            .sum::<f64>()
            .ln();
        // The limit is approached to first order in `v s`, since
        // `omega(x) = e^x (1 - e^x + ...)` and `e^x` is of order
        // `v s T_c / sum_c T_c` here. Anything tighter would be testing the
        // truncation of that series rather than the solver.
        assert_relative_eq!(point.z, expected, max_relative = 1e-7);
    }
}
