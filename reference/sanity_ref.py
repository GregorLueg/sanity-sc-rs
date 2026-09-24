#!/usr/bin/env python3
"""Numpy oracle for the Sanity kernels.

A direct, slow transcription of docs/SPEC.md with no attention paid to speed or
to memory. It exists so the Rust kernels have something to be wrong against.

Where SPEC rewrites an SI expression for numerical reasons, this file uses the
SI form instead, so the rewrite is checked rather than assumed. The digamma and
trigamma values come from the exact harmonic identities, not from an asymptotic
series, so they are independent of the Rust implementation.

    uv run --with numpy reference/sanity_ref.py --genes 300 --cells 500
    uv run --with numpy reference/sanity_ref.py --genes 40 --cells 60 \
        --out tests/data/oracle.txt
"""

from __future__ import annotations

import argparse

import numpy as np

EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992


def log_omega(x: np.ndarray) -> np.ndarray:
    """Solve exp(t) + t = x for t = ln omega(x), elementwise.

    Newton with a fixed iteration count: g is increasing and convex, so the
    iteration is globally convergent and there is nothing to branch on.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.where(x > 1.0, np.log(np.maximum(x - np.log(np.maximum(x, 1e-300)), 1e-300)), x)
    for _ in range(80):
        e = np.exp(t)
        t = t - (e + t - x) / (e + 1.0)
    return t


def omega_from_log(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Recover omega(x) from x and ln omega(x) on whichever branch is stable."""
    return np.where(x > 0.0, x - t, np.exp(np.minimum(t, 700.0)))


def solve_offset(v, s, counts, log_totals, log_total_sum):
    """Bisect SI eq. 27 for the offset z.

    Bisection rather than Newton on purpose: the Rust side uses Newton, and an
    oracle that shares the algorithm checks less.
    """
    vs = v * s
    log_vs = np.log(vs)

    def residual(z):
        x = v * counts + log_totals + log_vs - z
        return omega_from_log(x, log_omega(x)).sum() - vs

    lo = hi = log_total_sum
    step = 1.0
    while residual(lo) < 0.0:
        lo -= step
        step *= 2.0
        if step > 2.0**60:
            raise RuntimeError("no lower bracket for the offset")
    step = 1.0
    while residual(hi) > 0.0:
        hi += step
        step *= 2.0
        if step > 2.0**60:
            raise RuntimeError("no upper bracket for the offset")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stationary(v, s, counts, log_totals, z):
    """Per-cell weights and log fold changes at a known offset, SI eq. 23 and 28."""
    log_vs = np.log(v * s)
    x = v * counts + log_totals + log_vs - z
    t = log_omega(x)
    w = omega_from_log(x, t) / (v * s)
    d = t - log_vs - log_totals + z
    return w, d


def log_marginal(v, s, counts, log_totals, z, w, d):
    """SI eq. 20, 32 and 33, in the SI's own form."""
    n_cells = counts.size
    log_star = (
        -0.5 * n_cells * np.log(v)
        - 0.5 * np.sum(d * d) / v
        + np.sum(counts * d)
        - s * z
    )
    diag = s * w + 1.0 / v
    rank_one = 1.0 - np.sum(s * w * w / diag)
    log_det = np.log(rank_one) + np.sum(np.log(diag))
    return log_star - 0.5 * log_det, diag, rank_one


def cell_variances(v, s, counts, w, d, diag, rank_one):
    """SI eq. 37 for cells with counts, SI eq. 38 for cells without."""
    minor = rank_one + s * w * w / diag
    var = minor / (rank_one * diag)

    empty = counts == 0
    for c in np.nonzero(empty)[0]:
        var[c] = _half_unit_drop(v, s, w[c], d[c]) ** 2
    return var


def _half_unit_drop(v, s, w, d):
    """Bisect SI eq. 38 for sigma."""

    def drop(sigma):
        with np.errstate(over="ignore"):
            return sigma * (2.0 * d + sigma) / (2.0 * v) + s * np.log1p(w * np.expm1(sigma))

    lo = 0.0
    hi = np.sqrt(v)
    while drop(hi) < 0.5:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if drop(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def digamma_int(k: int) -> float:
    """psi(k + 1) = -gamma + sum_{j=1..k} 1/j. Exact identity, not a series."""
    return -EULER_MASCHERONI + float(np.sum(1.0 / np.arange(1, k + 1, dtype=np.float64)))


def trigamma_int(k: int) -> float:
    """psi1(k + 1) = pi^2 / 6 - sum_{j=1..k} 1/j^2. Exact identity."""
    j = np.arange(1, k + 1, dtype=np.float64)
    return float(np.pi**2 / 6.0 - np.sum(1.0 / (j * j)))


def run_gene(counts, totals, grid):
    """One gene, end to end. SPEC sections 2 to 8."""
    counts = counts.astype(np.float64)
    log_totals = np.log(totals)
    log_total_sum = np.log(totals.sum())
    s = counts.sum()  # SPEC section 1: 1/alpha prior, exponent K

    offsets = np.empty(grid.size)
    log_lik = np.empty(grid.size)
    for b, v in enumerate(grid):
        z = solve_offset(v, s, counts, log_totals, log_total_sum)
        w, d = stationary(v, s, counts, log_totals, z)
        log_lik[b], _, _ = log_marginal(v, s, counts, log_totals, z, w, d)
        offsets[b] = z

    weights = np.exp(log_lik - log_lik.max())
    weights /= weights.sum()

    mean_d = np.zeros(counts.size)
    mean_var = np.zeros(counts.size)
    all_d = np.empty((grid.size, counts.size))
    for b, v in enumerate(grid):
        w, d = stationary(v, s, counts, log_totals, offsets[b])
        _, diag, rank_one = log_marginal(v, s, counts, log_totals, offsets[b], w, d)
        var = cell_variances(v, s, counts, w, d, diag, rank_one)
        all_d[b] = d
        mean_d += weights[b] * d
        mean_var += weights[b] * var

    spread = np.sum(weights[:, None] * (all_d - mean_d) ** 2, axis=0)
    error = np.sqrt(mean_var + spread)

    k = int(round(counts.sum()))
    mean_offset = float(np.sum(weights * offsets))
    m = digamma_int(k - 1) - mean_offset  # psi(K)
    dm = np.sqrt(trigamma_int(k - 1) + np.sum(weights * (offsets - mean_offset) ** 2))
    return mean_d, error, m, float(dm), float(np.sum(weights * grid))


def simulate(n_genes, n_cells, seed, library_size=5000.0):
    """SPEC section 10, the independent recipe."""
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(0.0, 2.0, n_genes)
    mean_log_quotient = np.log(raw / raw.sum())
    variance = rng.exponential(2.0, n_genes)
    totals = np.maximum(np.round(rng.lognormal(np.log(library_size), 0.5, n_cells)), 1.0)
    d = rng.normal(0.0, 1.0, (n_genes, n_cells)) * np.sqrt(variance)[:, None]
    rate = totals[None, :] * np.exp(
        (mean_log_quotient - 0.5 * variance)[:, None] + d
    )
    counts = rng.poisson(rate)
    return counts, np.maximum(counts.sum(axis=0).astype(np.float64), 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", type=int, default=40)
    parser.add_argument("--cells", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v-min", type=float, default=1e-3)
    parser.add_argument("--v-max", type=float, default=50.0)
    parser.add_argument("--v-bins", type=int, default=64)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    counts, totals = simulate(args.genes, args.cells, args.seed)
    # K = 0 genes have an improper posterior and the crate rejects them.
    counts = counts[counts.sum(axis=1) > 0]
    args.genes = counts.shape[0]
    grid = np.exp(np.linspace(np.log(args.v_min), np.log(args.v_max), args.v_bins))

    lines = [
        "# sanity-rs oracle fixture v1",
        f"n_genes {args.genes}",
        f"n_cells {args.cells}",
        f"grid {args.v_min!r} {args.v_max!r} {args.v_bins}",
        "totals " + " ".join(repr(float(t)) for t in totals),
    ]

    for g in range(args.genes):
        column = counts[g]
        d, e, m, dm, v = run_gene(column, totals, grid)
        nz = np.nonzero(column)[0]
        lines.append(f"gene {g} {nz.size}")
        lines.append("indices " + " ".join(str(int(i)) for i in nz))
        lines.append("values " + " ".join(str(int(column[i])) for i in nz))
        lines.append(f"summary {m!r} {dm!r} {v!r}")
        lines.append("fold " + " ".join(repr(float(x)) for x in d))
        lines.append("error " + " ".join(repr(float(x)) for x in e))

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"wrote {args.out}: {args.genes} genes, {args.cells} cells")
    else:
        print(f"{args.genes} genes, {args.cells} cells, {args.v_bins} bins")
        print(f"total UMIs {int(counts.sum())}, median library {np.median(totals):.0f}")


if __name__ == "__main__":
    main()
