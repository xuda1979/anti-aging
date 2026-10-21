# Prospective data schema

Each experimental record requires `study_id`, `donor_id`, `species`, `cell_type`, `condition_id`, `biological_time`, `induction_duration`, `recovery_time`, `assay_batch`, and modality observation masks. Preserve the original sample identifier and units.

Intervention records require `target_id`, `target_type`, `assignment`, `guide_or_reagent_id`, `direction`, `exposure`, `measured_target_activity`, and matched control assignment. Post-treatment expression is not a replacement for assignment. Regulatory DNA elements additionally require chromosome, start, end, reference assembly and an explicit linked-gene annotation with confidence.

RNA assays retain raw counts and library size when available. Methylation assays retain methylated and total coverage counts, chemistry, genome assembly and CpG coordinates. Accessibility retains the peak vocabulary and assay-specific counts. Protein measurements retain scale and detection limits. Functional, identity, genomic-integrity and viability targets retain eligibility masks and independent assay identifiers. An absent outcome cannot be filled from an ageing clock.

For destructive measurements, `lineage_id` or a genuine tracking identifier is optional and must not be invented from computational matching. Repeated time points without tracking are conditional distributions or donor-level time courses, not observed same-cell pairs.

For future training, split groups before preprocessing. All samples from a donor belong to one outer donor split. An entire held-out study is excluded from fitting, feature selection and calibration. Predictive representations, priors, endpoints and thresholds must have documented freeze dates.
