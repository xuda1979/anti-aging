# GSE113957 adult fibroblast pilot data card

Source investigators: Fleischer et al. (2018), https://doi.org/10.1186/s13059-018-1599-6 . Accession: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113957 . The original authors generated the RNA-sequencing data; this project performed a separate, limited reanalysis.

The download script retrieves the published 143-sample, 27,142-transcript FPKM table and two platform-specific GEO series matrices. It verifies exact SHA-256 digests before use. Original matrices remain in ignored `data/raw/`; the public release distributes retrieval code, hashes and derived research artifacts. No controlled-access data or newly collected samples were used. Dataset-specific downstream commercial rights have not been established by this project.

`sample_manifest.csv` records public GSM accession, sample title, public cell ID, reported age, disease label, sex, platform, eligibility and split. Disease and age come from GEO characteristics, not the sample-title string. Ten HGPS and 22 pediatric normal samples are excluded from primary training/evaluation, leaving 111 healthy adults (ages 19–96), split 88 training / 23 holdout by public cell ID. Public IDs are only a donor proxy; family relationships and full culture history are unknown.

Current metadata include one normal sample age 96, while the paper summary states 1–94. The downloaded metadata are preserved and the discrepancy is disclosed. Age/sex/culture/platform confounding, source processing and cohort representativeness limit interpretation. No intervention, cell-function, health-outcome or clinical response labels support this pilot. Use it for reproducibility and association benchmarking, not diagnosis, treatment or rejuvenation claims.
