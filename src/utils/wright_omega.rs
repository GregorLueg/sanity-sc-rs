//! The Wright omega function, solved in the logarithm.
//!
//! `omega(x)` is the solution of `w + ln w = x`. It is the form the Sanity
//! stationarity condition reduces to (SPEC section 2.1, SI eq. 25-26). The
//! Lambert W form of the same equation takes `exp(x)` as its argument, which
//! overflows for ordinary inputs; the Wright form does not and needs no
//! asymptotic branch.
//!
//! Everything here works with `t = ln omega` rather than `omega`. The defining
//! equation becomes `exp(t) + t = x`, which is well conditioned over the whole
//! real line, and `ln omega` is what the caller wants anyway: the log fold
//! change is built from it directly.
//!
//! ### References
//!
//! Corless, Gonnet, Hare, Jeffrey, Knuth. *On the Lambert W function.*
//! Adv. Comput. Math. 5:329-359 (1996).

/// Convergence tolerance on `t`, absolute.
///
/// `g(t) = exp(t) + t - x` has `g' >= 1` everywhere, so the residual bounds the
/// error in `t` directly and no relative test is needed. Set to a few ulp of the
/// largest `t` this crate produces, which is `ln(ln(sum of all UMI counts))`,
/// comfortably below 100.
const OMEGA_TOL: f64 = 1e-14;

/// Iteration cap for the Halley solve.
///
/// Halley is cubic and the initial guesses below are within one unit of the
/// root over the whole domain, so four iterations are ample. The cap exists to
/// make a non-finite input terminate rather than spin.
const OMEGA_MAX_ITER: usize = 8;

/// Above this argument the large-`x` initial guess `ln(x - ln x)` is used.
///
/// Below it the root is bounded above by `x` itself, since `omega = x - ln omega`
/// and `omega > 0`, and `x` is then both a safe and a tight starting point.
const OMEGA_LARGE_X: f64 = 1.0;

/// Solve `exp(t) + t = x` for `t = ln omega(x)`.
///
/// Halley iteration on `g(t) = exp(t) + t - x`. `g` is strictly increasing and
/// convex, so the root is unique and the iteration is globally convergent from
/// either side.
///
/// ### Params
///
/// * `x` - The argument of the Wright omega function.
///
/// ### Returns
///
/// `ln omega(x)`.
#[inline]
pub(crate) fn log_omega(x: f64) -> f64 {
    let mut t = if x > OMEGA_LARGE_X {
        (x - x.ln()).ln()
    } else {
        x
    };

    for _ in 0..OMEGA_MAX_ITER {
        let e = t.exp();
        let g = e + t - x;
        if g.abs() < OMEGA_TOL {
            break;
        }
        // Halley: t -= 2 g g' / (2 g'^2 - g g''), with g' = e + 1 and g'' = e.
        let d1 = e + 1.0;
        t -= 2.0 * g * d1 / (2.0 * d1 * d1 - g * e);
    }

    t
}

/// Recover `omega(x)` from `x` and `t = ln omega(x)` without cancellation.
///
/// Two exact identities are available: `omega = exp(t)`, and `omega = x - t`
/// from the defining equation. The first loses precision for large `x`, where
/// `t` is the logarithm of a large number; the second loses it for negative `x`,
/// where `t` is close to `x`. Switching at zero keeps both sides well away from
/// their bad regime.
///
/// ### Params
///
/// * `x` - The argument of the Wright omega function.
/// * `t` - `ln omega(x)`, as returned by [`log_omega`].
///
/// ### Returns
///
/// `omega(x)`.
#[inline(always)]
pub(crate) fn omega_from_log(x: f64, t: f64) -> f64 {
    if x > 0.0 { x - t } else { t.exp() }
}

/// Solve `w + ln w = x` for `w`.
///
/// Only the tests want omega without its logarithm; the kernels always need
/// both. Convenience wrapper over [`log_omega`] and [`omega_from_log`] for callers
/// that do not also need the logarithm.
///
/// ### Params
///
/// * `x` - The argument of the Wright omega function.
///
/// ### Returns
///
/// `omega(x)`.
#[cfg(test)]
pub(crate) fn omega(x: f64) -> f64 {
    omega_from_log(x, log_omega(x))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Reference values from a 40-digit mpmath Newton solve of `exp(t) + t = x`.
    /// Columns are `x`, `omega(x)`, `ln omega(x)`.
    // Kept exactly as the reference tool printed them; rounding a reference
    // value to fit the type would be editing the oracle.
    #[allow(clippy::excessive_precision)]
    const REFERENCE: [(f64, f64, f64); 13] = [
        (-30.0, 9.3576229688392989538e-14, -30.000000000000093576),
        (-10.0, 4.539786874921542957e-5, -10.000045397868749215),
        (-2.0, 0.12002823898764122948, -2.1200282389876412295),
        (-1.0, 0.27846454276107379511, -1.2784645427610737951),
        (0.0, 0.567143290409783873, -0.567143290409783873),
        (0.5, 0.76624860816175025888, -0.26624860816175025888),
        (1.0, 1.0, 0.0),
        (2.0, 1.5571455989976114169, 0.44285440100238858314),
        // omega(1 + e) = e exactly, so this row is the identity rather than a
        // solved value; mpmath agrees to its own solve tolerance.
        (1.0 + std::f64::consts::E, std::f64::consts::E, 1.0),
        (10.0, 7.9294200950196973486, 2.0705799049803026514),
        (100.0, 95.44148664557583184, 4.5585133544241681598),
        (1000.0, 993.09916947238910439, 6.9008305276108956118),
        (1e6, 999986.1845032576279, 13.815496742372096878),
    ];

    #[test]
    fn test_omega_matches_reference() {
        for (x, w, t) in REFERENCE {
            assert_relative_eq!(omega(x), w, max_relative = 1e-13);
            // `t` crosses zero at x = 1, where a relative test is meaningless.
            assert!((log_omega(x) - t).abs() < 1e-13 * t.abs().max(1.0));
        }
    }

    #[test]
    fn test_omega_satisfies_defining_equation() {
        for i in -60..60 {
            let x = i as f64 * 2.5;
            let w = omega(x);
            assert_relative_eq!(w + w.ln(), x, max_relative = 1e-13, epsilon = 1e-13);
        }
    }

    #[test]
    fn test_omega_small_argument_does_not_underflow_to_zero() {
        // The whole point of working in `t`: omega(-700) is representable only
        // through its logarithm, and the log branch must still be exact.
        let t = log_omega(-700.0);
        assert_relative_eq!(t, -700.0, max_relative = 1e-12);
    }
}
