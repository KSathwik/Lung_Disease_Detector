> **Archived 2026-11-04 (superseded):** superseded by [NOVELTY_ROADMAP.md](NOVELTY_ROADMAP.md) (plan) and [NOVELTY_PROTOCOL.md](../NOVELTY_PROTOCOL.md) (approved decision rules). Kept for traceability; relative paths such as `lit_search/` below are relative to `docs/publication/`.

# Novelty plan: spectral localisation and suppression of source shortcuts

Status: **proposal, superseded by `NOVELTY_ROADMAP.md` (plan) and `NOVELTY_PROTOCOL.md` (decision rules, draft)** (2026-11-03). Nothing in this file has been run yet. It extends the BSPC journal manuscript; the IEEE EMBC 2027 paper stays as it is (see `CONFERENCE_VS_JOURNAL.md`). Every claim of novelty below must be re-checked with a deeper literature search before writing.

## 1. Why the current work needs a stronger contribution

The current manuscript is a careful evaluation study: a leakage-controlled, multi-source six-class benchmark, a once-only test and external evaluation, a source-identifiability probe and an audit of an earlier pipeline. It uses a standard model (DenseNet-121) and standard preprocessing. `docs/manuscripts/journal_bspc/REVIEWER_RISK.md` expects reviewers to ask for multiple seeds, ablations and more external cohorts, and to judge the methodological novelty as limited. Seeds and a preprocessing ablation are running (runs 3–5). This plan adds a contribution that is new, fits a signal-processing journal, and runs on the available laptop CPU.

## 2. What the literature already shows

| Finding | Source |
|---|---|
| Chest X-ray CNNs identify the hospital system and department with very high accuracy; external performance is worse; models may exploit confounders | Zech et al., PLoS Med 2018 ([arXiv 1807.00431](https://arxiv.org/pdf/1807.00431v2.pdf)) |
| Adding a second hospital can create a spurious source–label correlation; in 43% of settings worst-group accuracy got worse | [arXiv 2308.04431](https://ar5iv.labs.arxiv.org/html/2308.04431) |
| Internal performance is overestimated by up to ~20% on average because of shortcut learning (13 datasets) | [npj Digital Medicine 2024, s41746-024-01118-4](https://www.Nature.com/articles/s41746-024-01118-4) |
| Neural networks learn frequency shortcuts (low or high bands) that transfer to OOD data and are not removed by capacity or augmentation | Wang et al., ICCV 2023 ([CVF open access](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_What_do_neural_networks_learn_in_image_classification_A_frequency_ICCV_2023_paper.html)) |
| ImageNet-pretrained models focus on mid-to-high frequencies in medical images and are prone to shortcuts; shortcuts can be high- or low-frequency (CT, knee MRI; chest X-ray only as validation) | Lu, Juodelyte, Victor, Cheplygina 2024 ([arXiv 2407.11379](https://arxiv.org/html/2407.11379v2)) |
| Acquisition-dependent contrast and sharpness can act as shortcuts in COVID-19 chest X-ray classification | Scientific Reports 2023 (found in search; exact citation to be verified) |
| Lung segmentation helps cross-dataset generalisation only modestly; source bias remains | Teixeira et al., Sensors 2021 ([PMC8587284](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8587284/)); Bassi & Attux ([arXiv 2104.06176](https://ar5iv.labs.arxiv.org/html/2104.06176)) |
| Iterative nullspace projection (INLP) removes linearly decodable protected attributes from representations | Ravfogel et al., ACL 2020 ([2020.acl-main.647](https://aclanthology.org/2020.acl-main.647)) |
| Preserving high-frequency content (wavelets) changes medical image classification | McIntosh et al. ([arXiv 2205.03898](https://arxiv.org/pdf/2205.03898)) |
| BSPC publishes chest X-ray domain-generalisation work | Wang & Xia, BSPC 81 (2023), DELCOM ([record](https://pure.nwpu.edu.cn/en/publications/domain-ensemble-learning-with-cross-domain-mixup-for-thoracic-dis/)) |

**Gap (provisional):** no study found that measures, frequency band by frequency band, how much *source* information versus *disease* information multi-source chest radiographs carry on a patient-grouped, leakage-controlled benchmark, and uses that map to design preprocessing. This must be confirmed by an extended search (Scopus, Google Scholar, BSPC, CBM, MedIA, IEEE TMI/JBHI, MICCAI 2022–2026) before the claim is written.

## 3. What this project already has

- Six sources in one six-class benchmark, splits grouped by patient where identity is known, and leakage checks.
- A source probe: source predictable at 93.8% balanced accuracy over 6 sources (chance 16.7%), and 97.2% within Normal (5 sources).
- Grad-CAM attention outside the lungs for COVID-19 and Normal.
- Preprocessing that already includes a Gaussian low-pass filter, with a no-filter ablation (run 5) training now.
- An external cohort (Montgomery), and run 1 vs run 2 behaviour on it.

## 4. Proposed contributions

### C1. Spectral source–disease map (measurement; no training)

- **Bands.** Radially symmetric band-pass filters in the 2-D Fourier domain of the 224×224 preprocessed image: about 6 octave bands from DC to Nyquist, plus the cumulative low-pass and high-pass versions. Exact cut-offs fixed before running.
- **Features.** Frozen DenseNet-121 global-average-pooled features (ImageNet weights, and the served model's backbone) for each band-filtered image.
- **Probes.** Multinomial logistic regression (as in `lungai/evaluation/source_probe.py`) trained on train-split samples, scored on test-split samples:
  - source prediction (balanced accuracy, chance = 1/number of sources);
  - disease prediction (macro-F1), overall and within each source.
- **New index.** Source–disease separability per band, e.g. SDS(b) = normalised source accuracy − normalised disease macro-F1. It shows bands that carry hospital fingerprint but little pathology.
- **Output.** A band × {source, disease} figure, a table with bootstrap confidence intervals, and within-class source accuracy per band.
- **Cost.** About 2,000 images × 8 band variants × 2 extractors ≈ 30k CPU forward passes ≈ 1–2 h.

### C2. Band-selective source suppression (method; needs training)

- **Design.** From C1, take the bands with high source and low disease information and remove them at preprocessing time (band-stop). The rule for choosing bands is fixed *before* training and applied on validation data only.
- **Comparison, same split, same schedule:**
  - (a) the current pipeline (blur + CLAHE + low-pass, runs 2–4);
  - (b) no filters (run 5);
  - (c) band-selective suppression.
- **Evaluation.** Internal test (once), Montgomery (once), the source probe after suppression (does source information fall?), and the C4 matrix.
- **Cost.** One run ≈ 6 h CPU; 2–3 seeds for the journal (two or three nights, run one after another).

### C3. Source-information removal in feature space (method; little training)

- **Method.** Apply INLP to the served model's penultimate features: repeatedly train a linear source classifier and project the features onto its nullspace. Then retrain only the linear disease head.
- **Report.** Source probe accuracy and disease performance before and after, internally and on Montgomery. If disease accuracy collapses when source information is removed, that quantifies the entanglement.
- **Cost.** About 1 h; features are extracted once.

### C4. Leave-one-source-out generalisation matrix (evaluation; light training)

- **Method.** With frozen features and a linear head, train on all sources but one and test on the held-out source, for every source that has at least two classes. Repeat with C2/C3 features.
- **Output.** A source × source matrix. It turns one external cohort into several held-out-source tests, which addresses the reviewer objection "only one external cohort".
- **Cost.** About 1–2 h.

## 5. Working title and positioning

*Spectral localisation and suppression of source shortcuts in multi-source chest radiograph classification: a leakage-controlled six-class study.*

- **Measurement contribution:** C1 and C4, where source information lives in the spectrum, and how classifiers transfer across sources.
- **Method contribution:** C2 and C3, suppressing source information in the signal or in the features, judged by external and held-out-source performance, not internal accuracy.
- **Foundation:** the existing leakage-controlled benchmark, once-only evaluation, seed variability and the V5 audit.

## 6. Integrity rules for this extension

- Decision rules (band cut-offs, band selection for C2, INLP iterations, probe settings) are written into this repository **before** the corresponding results are computed, as was done for the run-selection protocol.
- Test and Montgomery are used once per final configuration; selection uses validation data only.
- Negative or mixed results are reported as such. If source and disease share the same bands, C1 is still a finding, and C2 is reported as not helpful.
- No claim of state of the art or clinical validity. Novelty is claimed only after the extended literature search.
- Every number goes through the generated-macro pipeline (`render_docs.py`), as in the rest of the manuscript.

## 7. Risks

| Risk | Mitigation |
|---|---|
| A prior study already did C1 for chest X-rays | Extended search first; if found, cite it and position C1 as a replication on a leakage-controlled six-class benchmark, with C2–C4 as the contribution |
| Source and disease information overlap in all bands | Report as a finding; strengthen C3/C4 |
| Band suppression lowers internal accuracy | Expected trade-off; judge by external and held-out-source results, report both |
| CPU time | C1, C3 and C4 need no full training; C2 runs one after another overnight |
| Single external cohort | C4 adds held-out-source tests; a second external cohort (e.g. VinDr, JSRT) needs data access |

## 8. Timeline (after the M.Tech deliverables, which stay the first priority)

| Step | When | Output |
|---|---|---|
| Extended literature search on C1/C3 | Day 1 | Novelty confirmed or repositioned; references verified |
| Write the C1–C4 protocol into the repository | Day 1 | Fixed band edges, selection rule, probe settings |
| C1, C3, C4 analyses | Day 1–2 | Tables, figures, macros |
| C2 training (1 seed, then 2 more) | Nights 2–4 | Run reports, seed variability |
| Manuscript revision (BSPC) | Day 3–5 | New sections, figures, discussion; REVIEWER_RISK update |

## 9. Decision needed

1. Start with the no-training analyses (C1, C3, C4) after the M.Tech deliverables are final.
2. Or run the full plan including C2 retraining (2–3 extra nights of CPU training).

## Sources

- [Wang et al., ICCV 2023 — frequency shortcut perspective](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_What_do_neural_networks_learn_in_image_classification_A_frequency_ICCV_2023_paper.html)
- [Zech et al. 2018 — confounding and hospital identification in chest X-rays](https://arxiv.org/pdf/1807.00431v2.pdf)
- [npj Digital Medicine 2024 — external accuracy overestimation (PEst)](https://www.Nature.com/articles/s41746-024-01118-4)
- [arXiv 2308.04431 — multi-hospital data can create spurious correlations](https://ar5iv.labs.arxiv.org/html/2308.04431)
- [Teixeira et al., Sensors 2021 — impact of lung segmentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8587284/)
- [Bassi & Attux — lung segmentation and generalisation](https://ar5iv.labs.arxiv.org/html/2104.06176)
- [Lu et al. 2024 — spectral analysis and transfer learning in medical imaging](https://arxiv.org/html/2407.11379v2)
- [Ravfogel et al., ACL 2020 — iterative nullspace projection](https://aclanthology.org/2020.acl-main.647)
- [McIntosh et al. — preserving high-frequency content](https://arxiv.org/pdf/2205.03898)
- [Wang & Xia, BSPC 2023 — domain-ensemble learning for unseen chest X-ray domains](https://pure.nwpu.edu.cn/en/publications/domain-ensemble-learning-with-cross-domain-mixup-for-thoracic-dis/)
