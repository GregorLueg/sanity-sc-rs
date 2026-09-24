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
