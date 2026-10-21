# Public data attribution

The executed biological pilot uses Gill et al., eLife 2022, DOI 10.7554/eLife.71624, GEO subseries GSE165177 within GSE165180.

Two NCBI processed log2 RPM matrices are downloaded by `cellclock.download`. Their exact URLs, byte sizes and SHA256 checksums appear in `provenance.json`. Original matrices are cached in ignored `data/raw/` and are not committed or included in the release archive. Public availability does not imply that original source datasets inherit this repository's license.

The included culture labels and original measurements are processed into per-sample predictions, matched donor-level contrasts and numerical summaries. Original source ownership is retained. Cite Gill et al. and GEO when reusing this analysis. All other catalogue datasets are prospective resources, not executed experiments.

The source files contain 95 sample columns in total. Selection uses exact recovered-condition names and yields 34 included samples. Donor codes are public study labels. No new participant data or private clinical information is present.

Synthetic data are generated from explicitly documented equations with invented regulator variables; they are not biological observations.
