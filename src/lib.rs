//! Bayesian inference of gene expression states from single cell UMI counts.
//!
//! A clean-room implementation of the Sanity method. Given raw UMI counts and
//! per-cell totals, this crate returns a posterior log expression level and an
//! error bar for every gene in every cell.
//!
//! # Provenance
//!
//! The reference implementation (`jmbreda/Sanity`) is GPL-3.0 and is **not** a
//! source for this crate. Everything here is written from `docs/SPEC.md`, which
//! restates the mathematics of the paper's Supplementary Information. See
//! `PROVENANCE.md` for the full position.
//!
//! # References
//!
//! Breda, Zavolan, van Nimwegen. *Bayesian inference of gene expression states
//! from single-cell RNA-seq data.* Nature Biotechnology 39(8):1008-1016 (2021).

#![warn(missing_docs)]

pub mod errors;
