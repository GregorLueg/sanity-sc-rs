# SPEC

The mathematics of the Sanity model, restated in this crate's notation.

Source: Breda, Zavolan, van Nimwegen, *Nature Biotechnology* 39(8):1008-1016
(2021), Supplementary Information sections S1.1 to S1.3. Equation numbers of the
form `SI (n)` refer to that document. This file restates; it does not transcribe.
See `PROVENANCE.md`.

## 0. Notation

Our symbols differ from theirs deliberately, so that a kernel citing `SI (25)`
is citing a reference rather than copying a line. Mapping:

| ours | theirs | meaning |
|---|---|---|
| `C` | `C` | number of cells |
| `k_c` | `n_gc` | UMI count of this gene in cell `c` |
| `K` | `n_g` | `sum_c k_c`, total UMIs for this gene |
| `s` | `n_g + 1` | `K + 1`; appears in every stationarity term |
| `T_c` | `N_c` | total UMIs in cell `c`, over all genes |
| `v` | `v_g` | variance of the log fold changes for this gene |
| `d_c` | `delta_gc` | log fold change of this gene in cell `c` |
| `d*_c` | `delta*_gc` | its maximum a posteriori value at a given `v` |
| `w_c` | `f_gc` | normalised weights, `sum_c w_c = 1` |
| `z` | `q_g` | log normalisation offset, `e^z = sum_c T_c e^{d_c}` |
| `m` | `mu_g` | posterior mean of `log(alpha_g)` |
| `e_c` | `epsilon_gc` | error bar on `d_c` |
| `dm` | `delta mu_g` | error bar on `m` |

The gene subscript is dropped throughout: genes are independent under this model
and every kernel below sees exactly one gene.

## 1. The model

`SI (1)-(15)`.

The transcription activity of a gene in a cell is split into a total activity for
the cell and a transcription quotient: the fraction of the cell's mRNA that is
this gene's. Capture and sequencing are Bernoulli per molecule, which keeps the
Poisson form, so the sequenced count is Poisson with mean `T_c * alpha_c`. The
multinomial coupling across genes is dropped on the grounds that no single gene
holds an appreciable share of a cell's UMIs; this is the approximation that makes
the gene the unit of work.

Write the quotient as a gene mean times a per-cell log fold change,

    alpha_c = alpha * exp(d_c)

Marginalising `alpha` out under a uniform prior (`SI (16)`; the integral is
extended from `[0, 1]` to `[0, inf)`, which costs a relative error of order
`N^{-N(1 - K/N)}` and is negligible unless one gene holds nearly all UMIs) and
imposing a zero-mean Gaussian prior of variance `v` on the `d_c` (`SI (17)`,
maximum entropy given a variance) gives the log posterior up to a constant:

    L(d, v) = -(C/2) ln v
              - (1 / (2 v)) sum_c d_c^2
              + sum_c k_c d_c
              - s ln( sum_c T_c exp(d_c) )                              SI (20)

Inputs are raw UMI counts. Anything already log-normalised breaks the Poisson
term outright.

## 2. The stationary point at fixed `v`

`SI (21)-(28)`.

Define

    z    with   exp(z) = sum_c T_c exp(d_c)                             SI (22)
    w_c  =  exp(-z) T_c exp(d_c),        sum_c w_c = 1                  SI (23)
    y_c  =  v k_c + ln T_c                                              SI (24)

Setting `dL/dd_c = 0` and multiplying by `v` turns the stationarity condition
into one scalar equation per cell:

    ln w_c + v s w_c = y_c - z                                          SI (25)

### 2.1 Wright omega, not Lambert W

`SI (26)` writes the solution with the Lambert W function. Substituting
`u_c = v s w_c` gives instead

    u_c + ln u_c = x_c,      x_c = y_c - z + ln(v s)

which is the Wright omega function, `u_c = omega(x_c)`. This crate solves that
form. The Lambert W form takes `exp(x_c)` as its argument, which overflows for
ordinary inputs and then needs an asymptotic branch to recover; the Wright form
has no such branch.

Solve in the logarithm. With `t = ln omega`, the defining equation is

    exp(t) + t = x

which is well conditioned over the whole real line and is what the downstream
kernels want anyway, since `ln w_c = t_c - ln(v s)`. Halley on `g(t) = e^t + t - x`
converges cubically from the initial guesses

    t0 = x - exp(x)        for x <= 0     (e^t is small; t is near x)
    t0 = ln(x - ln x)      for x  > 1     (omega ~ x - ln x)

Recover omega without cancellation by branch:

    omega = exp(t)     for x <= 0
    omega = x - t      for x  > 0         (exact, from omega = x - ln omega)

### 2.2 The offset

`SI (27)`. `z` is not known in advance. Imposing `sum_c w_c = 1` gives one scalar
equation:

    F(z) = sum_c omega(y_c - z + ln(v s)) - v s = 0

`omega` is increasing, so `F` is strictly decreasing in `z` and the root is
unique. `d omega / dx = omega / (1 + omega)`, so

    F'(z) = - sum_c omega_c / (1 + omega_c)

which makes Newton cheap and exact. Start from

    z0 = ln( sum_c T_c ) + v / 2

(the `v -> 0` limit is `z = ln sum_c T_c`; the `v/2` is the mean of `exp(d)`
under the prior). Bracket by stepping outwards, then Newton guarded by the
bracket. Failure to converge returns `SanityErrors::FractionSolveDiverged`.

Once `z` is known,

    d*_c = ln w_c - ln T_c + z = t_c - ln(v s) - ln T_c + z              SI (28)

### 2.3 The `v -> 0` limit

As `v -> 0` the equations degenerate to `exp(z) = sum_c T_c exp(v k_c)` and
`d*_c -> v k_c - v * (weighted mean)`. The `t` parameterisation reaches this
limit continuously; the `omega` one does not. This is the reason for solving in
`t`.

## 3. The marginal likelihood at fixed `v`

`SI (29)-(33)`.

Substituting `d*` into `SI (20)`, and using `sum_c T_c exp(d*_c) = exp(z)`:

    Lstar(v) = -(C/2) ln v
               - (1 / (2 v)) sum_c d*_c^2
               + sum_c k_c d*_c
               - s z

The curvature at the optimum is a diagonal plus a rank-one term:

    M_{c,c'} = (s w_c + 1/v) delta_{c,c'} - s w_c w_{c'}                SI (31)

The Laplace approximation then gives

    ln P(k | v) = Lstar(v) - (1/2) ln det M                             SI (32)

### 3.1 Determinant without cancellation

By the matrix determinant lemma, `SI (33)` writes

    det M = (1 - S) * prod_c (s w_c + 1/v),   S = sum_c s w_c^2 / (s w_c + 1/v)

For large `v`, `S -> 1` and `1 - S` is a difference of nearly equal numbers. Use
instead the identity

    A := 1 - S = sum_c w_c / (1 + v s w_c)

which is exact, manifestly positive, and free of cancellation. `A > 0` is an
invariant: each summand is positive and `sum_c w_c = 1`.

    ln det M = ln A + sum_c ln(s w_c + 1/v)

All reductions accumulate in `f64`, whatever the storage type. The differences
between neighbouring bins are small against the sum, and `f32` accumulation
flattens the posterior enough to make the bin choice noise.

### 3.2 Known bias of the Laplace step

Measured 2026-09-13 against an exact reference, obtained by applying the gamma
identity `A^{-s} = Gamma(s)^{-1} int_0^inf dt t^{s-1} exp(-A t)` to the coupling
term, which decouples the cells and reduces the `C`-dimensional integral to
nested 1-D quadrature. Verified to `1e-14` against direct 3-D quadrature at
`C = 3`.

The Laplace step systematically **underestimates** the posterior mean of `v`,
across fourteen gene profiles spanning `C` from 100 to 2000 and `K` from 3 to
10069. The bias is one-signed and bounded: 0 to -23%, median -12%. It vanishes
at high coverage (-0.0% at 20 UMIs per cell) and is worst for genes whose counts
sit in a small fraction of cells, where the per-cell posterior is most
asymmetric and least Gaussian.

This is left uncorrected. The bias only bites genes carrying almost no
information about `v` in the first place (SI S3.8), and the obvious fix does not
work: the next-order Laplace term, `sum_c [L'''' / (8 h^2) + 5 (L''')^2 / (24
h^3)]`, was measured and makes the estimate substantially *worse*, not better.

## 4. Per-cell variance at fixed `v`

`SI (35)-(37)`. The posterior over `d` at fixed `v` is Gaussian with covariance
`M^{-1}`, whose diagonal is the ratio of a minor to the determinant. Writing

    B_c := s w_c + 1/v
    R_c := s w_c^2 / B_c = v s w_c^2 / (1 + v s w_c)

the minor drops the `c`th term from both the sum and the product, so

    var(d_c) = (A + R_c) / (A * B_c)                                    SI (37)

### 4.1 Cells with no counts

`SI (38)`. When `k_c = 0` the log posterior is strongly asymmetric about `d*_c`:
zero counts bound `d_c` from above but are consistent with arbitrarily low
`d_c`. The Gaussian variance is then the wrong summary. Replace it with the
symmetric error bar defined by a drop of `1/2` in the log posterior on the upper
side: solve for `sigma > 0` in

    sigma (2 d*_c + sigma) / (2 v) + s ln(1 + w_c (exp(sigma) - 1)) = 1/2

and set `var(d_c) = sigma^2`. The left side is strictly increasing in `sigma`
from `0`, so the root is unique; bracket by doubling and bisect or Newton.

## 5. The posterior over `v`

`SI (34)`. The prior on `v` is a scale prior, uniform in `ln v`. Approximate the
posterior on a grid of `B` bins equally spaced in `ln v` over `[v_min, v_max]`,
so that the flat weighting over bins *is* the prior:

    W_b = exp(L_b) / sum_b' exp(L_b'),      L_b = ln P(k | v_b)

Compute in log space: subtract `max_b L_b` before exponentiating.

`v_min`, `v_max` and `B` are this crate's to choose and are named constants with
dated provenance in their doc comments. They are not taken from the reference
implementation.

The gene's variance estimate is the posterior mean, `<v> = sum_b W_b v_b`. For
very lowly expressed genes this is small even when the true variance is large;
that is the correct behaviour, not a defect, because the variation lies below the
detection limit (SI S3.8).

## 6. Aggregation over `v`

`SI (39)-(42)`.

    <d_c>  = sum_b W_b d*_c(v_b)                                        SI (39)
    e_c^2  = sum_b W_b [ var(d_c)(v_b) + (d*_c(v_b) - <d_c>)^2 ]        SI (42)

`SI (41)` is the algebraically equivalent `sum_b W_b (var + d*^2) - <d>^2`. Use
`SI (42)`: it is a variance about the mean rather than a difference of second
moments, and does not lose digits when `|<d_c>|` is large.

### 6.1 Two passes, not per-bin storage

`SI (42)` needs `<d_c>` before the bin loop can accumulate, which naively means
holding `d*` and `var` for every bin and every cell, `2 B C` floats. Avoid it:
`z_b` is a single scalar per bin, so

1. Pass one: for each bin solve `z_b` and accumulate `L_b`. Store `z_b` and
   `L_b`. Memory `O(B)`.
2. Form `W_b`.
3. Pass two: for each bin, with `z_b` already known, one `O(C)` evaluation gives
   `w_c`, `d*_c` and `var(d_c)` with no iteration. Accumulate `<d_c>` and `e_c^2`
   by weighted Welford. Memory `O(C)`.

The second pass costs one sweep per bin against the several Newton steps of the
first, so the overhead is small and the scratch is `O(B + C)`.

## 7. Variance rules

The full marginalisation of section 6 is the default. Three cheaper rules use the
same grid but collapse it to a single `v`, after which `d*_c` and `var(d_c)` are
read off at that `v` alone and `e_c^2 = var(d_c)`. Only the marginalising rule
needs the second pass; the others need two vectors of length `C`.

| rule | `v` used |
|---|---|
| `Marginalise` | none; sum over bins per section 6 |
| `PosteriorMean` | `<v> = sum_b W_b v_b`, re-solved |
| `MaxPosterior` | `v_b` at `argmax_b L_b` |
| `Fixed(v)` | supplied by the caller; no grid, no scan |

## 8. Mean expression and its error bar

`SI (43)-(47)`. At fixed `v` the posterior over the gene mean is a gamma in
`alpha`, giving

    <ln alpha>_v = psi(K + 1) - z(v)                                    SI (44)
    var(ln alpha)_v = psi1(K + 1)      (independent of v)               SI (46)

so, averaging over the grid,

    m     = psi(K + 1) - sum_b W_b z_b                                  SI (45)
    dm^2  = psi1(K + 1) + sum_b W_b (z_b - <z>)^2                       SI (47)

`psi` is the digamma function and `psi1` the trigamma. `K` is an integer, so
`psi(K + 1) = -gamma + sum_{j=1..K} 1/j`; the harmonic form is exact for small
`K` and the asymptotic expansion takes over above a threshold that is ours to
set and is checked against `Rscript -e 'digamma(...)'`. Reference values are
never written from memory.

## 9. Output

Per gene and per cell:

- `d_c`, the posterior log fold change, with error bar `e_c`.
- the log transcription quotient `m + d_c`, which is what downstream consumers
  read as the normalised expression. Its error bar is `e_c`; `dm` is the
  separate uncertainty on the gene's mean level.

Per gene: `m`, `dm`, and `<v>`.

`m` is `ln alpha`, the *geometric* mean quotient. The prior has `<d_c> = 0`, so
the arithmetic mean quotient is `exp(m + v/2)`, not `exp(m)`. Section 10's
recipe parametrises by the arithmetic one, and anything comparing the two must
put the `v/2` back.

## 10. Simulation

`SI` S1.2. The independent-genes recipe, for generating inputs that neither
implementation produced:

1. Draw a mean log quotient `m_g` per gene, and a variance `v_g` per gene from an
   exponential distribution.
2. Draw `d_gc ~ Normal(0, v_g)`.
3. Set `ln alpha_gc = m_g - v_g / 2 + d_gc`. The `-v_g/2` makes
   `<alpha_gc> = exp(m_g)`, so the quotients sum to one in expectation.
4. Take `T_c` per cell.
5. Draw `k_gc ~ Poisson(T_c alpha_gc)`.

The branched-random-walk variant replaces step 2 with a random walk over cells,
`d_gc = d_{g,parent(c)} + Normal(0, 1)`, restarting from a uniformly chosen
existing cell at a fixed branch length, then rescaling each gene's `d_g` to have
variance `v_g`.

## 11. Cell-to-cell distances

`SI` S1.3 defines an error-bar-aware squared distance between cells. It is not
implemented here: this crate's contract stops at the means and the error bars,
and the distance lives with the consumer that needs it.
