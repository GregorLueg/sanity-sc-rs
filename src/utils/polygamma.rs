//! Digamma and trigamma.
//!
//! Needed for the gene's mean log quotient and its error bar (SPEC section 8,
//! SI eq. 44 and 46). Both are evaluated at `K + 1` where `K` is a gene's total
//! UMI count, so the argument is an integer that can run from 1 to many
//! millions. The implementations here are for general positive real arguments;
//! that costs nothing and makes them testable against `Rscript`.
//!
//! Recurrence up to a threshold, then the standard asymptotic expansion.

/// Argument above which the asymptotic expansions are used directly.
///
/// Below it, `psi(x) = psi(x + 1) - 1/x` and `psi1(x) = psi1(x + 1) + 1/x^2`
/// shift the argument up. Truncation error of both series at this threshold is
/// below `1e-16` relative, checked against `Rscript -e 'digamma(...)'` on
/// 2026-09-13.
const POLYGAMMA_SHIFT: f64 = 12.0;

/// Digamma, the logarithmic derivative of the gamma function.
///
/// ### Params
///
/// * `x` - Argument, strictly positive.
///
/// ### Returns
///
/// `psi(x)`. Non-positive arguments return NaN; this crate only ever calls it
/// with `K + 1 >= 1`.
pub(crate) fn digamma(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::NAN;
    }

    let mut z = x;
    let mut acc = 0.0;
    while z < POLYGAMMA_SHIFT {
        acc -= 1.0 / z;
        z += 1.0;
    }

    // psi(z) ~ ln z - 1/(2z) - sum_k B_{2k} / (2k z^{2k})
    let r = 1.0 / z;
    let r2 = r * r;
    acc + z.ln() - 0.5 * r
        - r2 * (1.0 / 12.0
            - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 * (1.0 / 240.0 - r2 / 132.0))))
}

/// Trigamma, the derivative of the digamma function.
///
/// ### Params
///
/// * `x` - Argument, strictly positive.
///
/// ### Returns
///
/// `psi1(x)`. Non-positive arguments return NaN.
pub(crate) fn trigamma(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::NAN;
    }

    let mut z = x;
    let mut acc = 0.0;
    while z < POLYGAMMA_SHIFT {
        acc += 1.0 / (z * z);
        z += 1.0;
    }

    // psi1(z) ~ 1/z + 1/(2z^2) + sum_k B_{2k} / z^{2k+1}
    let r = 1.0 / z;
    let r2 = r * r;
    acc + r
        + 0.5 * r2
        + r * r2
            * (1.0 / 6.0
                - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 * (1.0 / 30.0 - 5.0 * r2 / 66.0))))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `Rscript -e 'digamma(x); psigamma(x, 1)'`, 2026-09-13.
    /// Columns are `x`, `psi(x)`, `psi1(x)`.
    // Kept exactly as the reference tool printed them; rounding a reference
    // value to fit the type would be editing the oracle.
    #[allow(clippy::excessive_precision)]
    const REFERENCE: [(f64, f64, f64); 11] = [
        (1.0, -0.57721566490153231, 1.6449340668482264),
        (2.0, 0.42278433509846747, 0.64493406684822641),
        (3.0, 0.92278433509846747, 0.39493406684822635),
        (5.0, 1.5061176684318007, 0.22132295573711527),
        (10.0, 2.2517525890667214, 0.10516633568168572),
        (10.5, 2.3030010342976861, 0.09991695605912676),
        (100.0, 4.6001618527380881, 0.010050166663333566),
        (1e4, 9.2102903711428503, 0.00010000500016666658),
        (1e6, 13.81551005796419, 1.0000005000001672e-6),
        (0.5, -1.9635100260214231, 4.934802200544679),
        (1.5, 0.036489973978576895, 0.93480220054467933),
    ];

    #[test]
    fn test_polygamma_matches_r() {
        for (x, psi, psi1) in REFERENCE {
            assert_relative_eq!(digamma(x), psi, max_relative = 1e-13);
            assert_relative_eq!(trigamma(x), psi1, max_relative = 1e-13);
        }
    }

    #[test]
    fn test_digamma_recurrence_is_consistent() {
        // psi(x + 1) - psi(x) = 1/x, across the shift threshold.
        for i in 1..40 {
            let x = i as f64 * 0.7;
            assert_relative_eq!(digamma(x + 1.0) - digamma(x), 1.0 / x, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_trigamma_recurrence_is_consistent() {
        for i in 1..40 {
            let x = i as f64 * 0.7;
            assert_relative_eq!(
                trigamma(x) - trigamma(x + 1.0),
                1.0 / (x * x),
                max_relative = 1e-12
            );
        }
    }
}
