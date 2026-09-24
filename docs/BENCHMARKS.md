# Benchmarks against the reference binary

Black-box comparison with the reference Sanity binary (v2.0.0, built from
`jmbreda/Sanity`), under the terms in `docs/PROVENANCE.md`. The harness lives
outside this repository at `~/repos/others/sanity-comparison`.

## 2026-09-24, sanity-rs 41edaf1

Machine: Apple M1 Max, 10 cores, 64 GB, macOS 26.6.2, rustc 1.95.0, release
profile.

Input: this crate's simulator, 490 genes by 2000 cells (500 requested, 10 came
back empty), library size 125, 77 359 stored counts (7.9% dense). Both read the
same Matrix Market file and write text output. Run: `Marginalise` / `-v_m MARG`,
160 bins over `[1e-3, 50]`.

Wall time and peak RSS are for the whole process, from `/usr/bin/time -l`.

| threads | wall, ours | wall, reference | speedup | peak RSS, ours | peak RSS, reference |
|---|---|---|---|---|---|
| 1 | 27.8 s | 74.9 s | 2.7x | 27.6 MB | 9.9 MB |
| 8 | 4.2 s | 10.4 s | 2.5x | 29.4 MB | 55.4 MB |

Our split at 8 threads: parse 0.006 s, compute 4.03 s, write 0.18 s.

By arithmetic, not profiling: the two dense gene-by-cell `f64` outputs of
`sanity` are 15.7 MB, and the harness adds a third, 7.8 MB, for
`log_transcription_quotients`. The reference's peak grows with the thread count.

At scale, 1998 genes by 20 000 cells (2000 requested), library size 500,
3 552 556 stored counts (8.9% dense), same settings:

| threads | wall, ours | wall, reference | speedup | peak RSS, ours | peak RSS, reference |
|---|---|---|---|---|---|
| 8 | 121.4 s | 415.1 s | 3.4x | 1012 MB | 937 MB |

Our split: parse 0.26 s, compute 114.2 s, write 6.9 s. The harness's extra copy
for `log_transcription_quotients` is 320 MB of our peak.

In both runs the outputs agree: per-gene correlation of the log transcription
quotients and of their error bars is 1.0000 on every gene, and the largest
absolute difference in log transcription quotient is below `5e-4`.

## 2026-09-24, GPU against CPU, sanity-sc-rs c0c5192

Both paths of this crate, same process, `examples/profile_gpu.rs`. Machine as
above; the GPU is the M1 Max's own, through wgpu on Metal. CPU is `sanity` on
10 Rayon threads in `f64`; GPU is `sanity_gpu`, `f32` on the device with the
likelihood assembled in `f64` on the host. Simulated counts, library size 500,
161 bins over `[1e-3, 50]`. Wall clock of the call only, kernels compiled
beforehand.

Errors are the GPU against the CPU, worst gene: the log fold change in units
of the CPU's own error bar, and the absolute error in the log transcription
quotient `m + d_c`.

1998 genes by 20000 cells:

| rule | CPU | GPU | speedup | worst `d_c` / `e_c` | worst `m + d_c` |
|---|---|---|---|---|---|
| `Marginalise` | 108.1 s | 1.39 s | 77.9x | `1.9e-4` | `5.9e-5` |
| `PosteriorMean` | 72.0 s | 1.19 s | 60.8x | `1.9e-4` | `1.1e-4` |
| `MaxPosterior` | 72.7 s | 4.78 s | 15.2x | `4.4e-5` | `8.1e-6` |
| `Fixed(1.0)` | 1.28 s | 0.15 s | 8.4x | `5.5e-5` | `1.5e-6` |

500 genes by 200000 cells:

| rule | CPU | GPU | speedup | worst `d_c` / `e_c` | worst `m + d_c` |
|---|---|---|---|---|---|
| `Marginalise` | 207.2 s | 3.47 s | 59.6x | `2.7e-3` | `3.1e-4` |
| `PosteriorMean` | 188.6 s | 3.79 s | 49.8x | `4.8e-3` | `3.1e-4` |
| `MaxPosterior` | 188.8 s | 7.99 s | 23.6x | `1.0e-4` | `6.0e-6` |
| `Fixed(1.0)` | 3.16 s | 0.36 s | 8.7x | `1.5e-4` | `1.6e-6` |

Every error sits below 1% of an error bar, which is as far as the variance
grid itself is resolved, and every log transcription quotient within the
`5e-4` by which this crate and the reference binary agree above. The worst
genes are the most expressed ones (`K` near `3e5`) at 200000 cells.

`MaxPosterior` is the slow one on the GPU because its argmax is discrete: bins
the device cannot separate from the best are re-solved on the CPU in `f64`, and
genes with little data have long flat stretches of near-equal bins.

