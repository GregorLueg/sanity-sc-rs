# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Crate

`sanity-rs`: Bayesian inference of gene expression states from single cell UMI
counts. A clean-room Rust implementation of the Sanity method (Breda, Zavolan,
van Nimwegen, *Nature Biotechnology* 2021, doi 10.1038/s41587-021-00875-x).
Library crate only, no binaries.

Input is a sparse UMI count matrix plus per-cell totals. Output is, per gene and
per cell, a posterior log expression level and its error bar. Downstream
consumers are `bixverse-rs` (as a normalisation) and `bonsai-rs` (which takes the
means and the error bars as its two input matrices).

## Licence rules, which are binding

**Never open, clone, read or grep `jmbreda/Sanity`.** Not here, not in a sibling
worktree, not through a subagent. That code is GPL-3.0 and a port would be a
derivative work under section 5, which is incompatible with the MIT licence on
this crate and on everything downstream of it.

Implementation reads `docs/SPEC.md` and nothing else. The spec restates the
mathematics of the paper's Supplementary Information in our own notation, which
is free to do because copyright does not cover mathematics. Every kernel cites
the SI equation number it implements.

Running their binary is fine. GPL-3.0 has no NonCommercial clause, so black-box
comparison is unconstrained. The harness for that lives outside this repository.

Do not carry over their tuned constants. Every threshold here is a named `const`
whose doc comment says where the number came from: an SI equation, or our own
measurement with the date. `PROVENANCE.md` records the full position and the
disclosure.

## Commands

```bash
cargo build --release
cargo test --release

# Single test
cargo test --release -- model::fractions::tests::test_fractions_sum_to_one --exact --nocapture

# The numpy oracle the Rust kernels are checked against
uv run --with numpy reference/sanity_ref.py --genes 300 --cells 500

# Reference values for the special functions. Never write these from memory.
Rscript -e 'cat(digamma(c(1,2,10,1e4)), psigamma(c(1,2,10,1e4), 1))'

cargo doc --no-deps --open
```

Release profile is `opt-level = 3`, `lto = "thin"`, `codegen-units = 4`.

## Architecture

### The gene is the unit of work

Genes are independent under this model. There is no cross-gene coupling and no
global iteration, so the whole method is one `par_iter` over genes and nothing
else. A gene's input is its sparse column plus the full vector of per-cell
totals, because a cell with zero counts still contributes through its library
size. The algorithm is dense in the cell axis while the input stays sparse.

### Numeric policy

Storage is generic over `SanityFloat`; every reduction accumulates in `f64`
regardless. The marginal likelihood sums over all cells, and the differences that
pick out one variance bin from its neighbours are small against that sum, so
`f32` accumulation would flatten the posterior and make the bin choice noise.

Inputs are raw UMI counts. Anything already log-normalised is the wrong input:
the method's entire purpose is to model the Poisson sampling in the raw counts.

### Wright omega, not Lambert W

The stationarity condition for the per-cell offsets reduces to `w + ln w = x`,
which is Wright omega. Solve that directly. The Lambert W form of the same
equation takes an exponential argument, which overflows for ordinary inputs and
then needs an asymptotic branch to paper over it. Halley iteration on the Wright
form has no such branch and converges cubically.

### Memory

Only the fully marginalising variance rule needs per-bin state for every cell.
The point-estimate rules need two vectors of length `n_cells`. Do not size the
scratch for the marginalising path and charge it to all four.

## Conventions and gotchas

- British English throughout (`normalise`, `optimise`, `neighbours`, `centred`).
- `#![warn(missing_docs)]` is on. Every item gets `### Params` / `### Returns`.
- One `thiserror` enum in `src/errors.rs`, grouped by subsystem. Add to the
  matching section, not the bottom. Anything a caller could hit returns `Err`;
  panics are for broken invariants the type system cannot state.
- `wide` types appear only in `src/utils/simd.rs`.
- Benches need a `harness = false` entry in `Cargo.toml` and must check their
  output before reporting a timing. An implausible speedup is the signature of a
  sweep that did nothing.
- Do not add a SIMD tier without a measurement in its doc comment, and pick the
  kernel by call count rather than by how vectorisable it looks.
- Measure a component's share of the whole before optimising it, and check that
  the pipeline calls it before doing either.

## What's tracked outside this file

- The algorithm: `docs/SPEC.md`.
- The licence position and disclosure: `PROVENANCE.md`.
