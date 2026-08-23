# Data-Fit Note — Calibrated xG on StatsBomb Open Data
*MSDS 696 Practicum II · Pod Lab: “Vet your data against your question” · 2026-07-14*

**Research question (v1).** Can event-level features from StatsBomb Open Data support a well-calibrated expected-goals (xG) model — judged on calibration (reliability curves, ECE, Brier) rather than raw accuracy?

## Fit
The event schema contains exactly what pre-shot xG needs: shot location, body part,
technique, play pattern, shooter position, pressure flags, and a 2-D freeze frame on
~99% of sampled shots, plus a clean binary outcome
(goal rate 11.7%, penalties 1.1% of shots;
0 shootout penalties excluded from the sample as non-in-game events).
Goal rate falls monotonically with derived shot distance — the signal the question needs
is present. Mean StatsBomb xG (0.099) tracks the realized goal
rate, so a calibrated model on these inputs is demonstrably achievable.

## Provenance
StatsBomb Ltd's public research release (github.com/statsbomb/open-data): broadcast-video
annotation against a versioned public spec; free for non-commercial research with
attribution. Inventory at pull time: **24 competitions,
80 competition-seasons, 3,961 matches
(1958-06-24 → 2025-07-27)**; 360 freeze-frame data on
12 competition-seasons. Data is served from a moving `master` branch →
capstone will cache one pull to parquet and record the pull date. (This audit:
0 match-index pulls and 0 event pulls failed.)

## Three biggest limitations
1. **Convenience sample, not a population.** La Liga alone is
   22% of matches (Messi-era Barcelona bias); tournaments
   overrepresented; 33% women's matches. External validity is
   limited → competition-grouped splits and per-competition reliability reporting.
2. **Schema/annotation drift.** Only 95% of matches are on
   the current `data_version 1.1.0`; sparse boolean flags encode False as NaN, so naive
   missingness reads are wrong → explicit flag-fill policy and per-season coverage gates.
3. **Context ceiling.** Freeze frames ≠ tracking data (no ball speed, limited GK detail;
   360 only on 12/80 competition-seasons).
   Residual error is partly irreducible → benchmark against StatsBomb's own xG and report
   the gap as measured context deficit (the "63% vs Vegas" framing from Practicum I).

## Feasibility
27.2 shots/match on the audit sample
(16 matches, 10 seasons) ⇒
**≈107,937 shots / ≈12,626 goals**
corpus-wide: ample for 10-bin reliability curves, grouped CV, and a calibration holdout.
One-time full pull (~3,961 event files) cached to parquet keeps
iteration fast. Leakage controls fixed up front: `shot_statsbomb_xg` (benchmark only),
`shot_outcome`, `shot_end_location`, `shot_deflected`, and match final scores are barred
from the feature set.

## Partner stress-test → survived
Class imbalance (binned calibration handles ~10% positives at this n); Barcelona bias
(made visible via grouped splits, not hidden); "why not use StatsBomb xG as a feature"
(label-adjacent leakage — benchmark only); low ceiling (the model-vs-StatsBomb gap *is* a
finding); pooled vs stratified across genders/formats (empirical sub-question, both reported).

## Decision — KEEP (question reshaped, not the data)
Using StatsBomb Open Data shots (penalty-shootout events excluded, penalties modeled separately), can a model built only on pre-shot features produce goal probabilities that are well-calibrated within *and across* competitions — measured by reliability curves, ECE, and Brier score under competition-grouped validation, benchmarked against StatsBomb's own xG?
*Contingency, not plan:* add 360 freeze-frame features (12 competition-seasons) if
standard freeze-frame features plateau.
