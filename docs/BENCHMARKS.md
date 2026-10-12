# Benchmarks

Numbers for the current version, v0.4.0. Older ones live in the git history of
this file.

Machine: Apple M1 Max, 10 cores, 64 GB, macOS 26.6.2, release profile. Input is
this crate's simulator throughout. macOS background daemons kept the load
average between 7 and 80 during the 2026-09-30 runs, so treat single timings as
good to 10 to 20%.

## Against the reference binary

Black-box comparison with the reference Sanity binary (v2.0.0, built from
`jmbreda/Sanity`), under the terms in `docs/PROVENANCE.md`. The harness is
private and lives outside this repository. Both read the same Matrix Market
file and write text output. Run: `Marginalise` / `-v_m MARG`, 160 bins over
`[1e-3, 50]`. Wall time and peak RSS are for the whole process, from
`/usr/bin/time -l`.

The reference timings are from 2026-09-24, same machine; the binary has not
changed since. Ours are from 2026-09-30, on v0.3.0. v0.4.0 only touched the GPU
path, so they stand.

490 genes by 2000 cells (500 requested, 10 came back empty), library size 125,
77 359 stored counts (7.9% dense):

| threads | wall, ours | wall, reference | speedup | peak RSS, ours | peak RSS, reference |
|---|---|---|---|---|---|
| 1 | 3.28 s | 74.9 s | 23x | 27.9 MB | 9.9 MB |
| 8 | 0.59 s | 10.4 s | 18x | 30.4 MB | 55.4 MB |

1998 genes by 20000 cells (2000 requested), library size 500, 3 552 556 stored
counts (8.9% dense):

| threads | wall, ours | wall, reference | speedup | peak RSS, ours | peak RSS, reference |
|---|---|---|---|---|---|
| 8 | 22.3 s | 415.1 s | 19x | 1079 MB | 937 MB |

Our split at that size: parse 0.26 s, compute 14.7 s, write 7.3 s. A third of
the wall clock is now formatting 40 million numbers as text. The harness keeps
an extra copy for `log_transcription_quotients`, 320 MB of our peak.

The outputs agree to the precision written: per-gene correlation of the log
transcription quotients and of their error bars is 1.0000 on every gene, and
the largest absolute difference in log transcription quotient is `3e-6` at both
sizes, against six decimals in the text files.

## GPU against CPU

Both paths in one process, `examples/profile_gpu.rs`, sanity-sc-rs v0.4.0,
2026-10-12. CPU is `sanity` on 10 threads in `f64`; GPU is `sanity_gpu` through
wgpu on Metal, cubecl's WGSL translated by wgpu. Library size 500, 161 bins over
`[1e-3, 50]`. Wall clock of the call only, kernels compiled beforehand. Errors
are the GPU against the CPU, worst gene: the log fold change in units of the
CPU's own error bar, and the absolute error in the log transcription quotient
`m + d_c`.

1998 genes by 20000 cells:

| rule | CPU | GPU | speedup | worst `d_c` / `e_c` | worst `m + d_c` |
|---|---|---|---|---|---|
| `Marginalise` | 13.1 s | 0.57 s | 23.1x | `2.5e-4` | `1.0e-4` |
| `PosteriorMean` | 13.1 s | 0.51 s | 25.7x | `2.9e-4` | `1.7e-4` |
| `MaxPosterior` | 13.1 s | 0.65 s | 20.1x | `4.3e-5` | `8.1e-6` |
| `Fixed(1.0)` | 1.40 s | 0.16 s | 8.6x | `5.5e-5` | `1.6e-6` |

500 genes by 200000 cells:

| rule | CPU | GPU | speedup | worst `d_c` / `e_c` | worst `m + d_c` |
|---|---|---|---|---|---|
| `Marginalise` | 57.8 s | 2.40 s | 24.1x | `3.1e-3` | `4.6e-4` |
| `PosteriorMean` | 58.9 s | 2.33 s | 25.3x | `3.2e-3` | `4.6e-4` |
| `MaxPosterior` | 59.6 s | 2.44 s | 24.4x | `2.1e-4` | `5.7e-6` |
| `Fixed(1.0)` | 3.61 s | 0.64 s | 5.7x | `2.7e-4` | `1.6e-6` |

Every error sits below 1% of an error bar, which is as far as the variance grid
itself is resolved. The worst genes are the most expressed ones at large `v`,
where the posterior over the variance is broad and small differences in the bin
likelihoods move the weights. Bonsai reads `log_fold_changes`, so the `d_c`
column is the one that reaches it.

The CPU path agrees with the reference binary to the precision written, so the
GPU's worst `m + d_c` of `4.6e-4` at 200000 cells is also its distance from the
reference.

The worst `d_c` at 200000 cells went from `1.9e-3` in v0.3.0 to `3.1e-3`. That's
cubecl 0.11, which now fuses every multiply-add it can into an `fma`, with no
way to turn it off. Taking that pass out of a patched cubecl gave back the v0.3.0
errors exactly. Still a third of the way to the 1% line.

`MaxPosterior` re-solves bins the device cannot separate from the best in `f64`
on the CPU. Those re-solves run the same sparse path as `sanity`, so they cost
little, and the rule lands on the same bin as the CPU.
