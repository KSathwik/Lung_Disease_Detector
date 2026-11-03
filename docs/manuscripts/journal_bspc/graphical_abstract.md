# Graphical abstract: description of the intended figure

BSPC accepts an optional graphical abstract: one image, readable at 5 × 13 cm (531 × 1328 pixels at 96 dpi or proportionally larger), TIFF/EPS/PDF/PNG. It has not been drawn yet. It should be produced from the same assets as `fig_pipeline` and the result figures **after the final model is fixed**, with numbers taken from `../shared/results.json`, not typed by hand.

## Layout (left to right, three panels)

**Panel 1: "Build" (result-independent).**
Seven source icons (NIH ChestX-ray14, Kermany, Shenzhen, COVID-19 image data collection, TBX11K, COVID-19 Radiography DB; Montgomery set apart in a different colour, labelled "external hospital, never trained on"). Arrows into a funnel labelled with the four controls:
- official download + SHA-256 provenance
- label rules fixed before training (proxy and latent labels excluded)
- patient grouping + pHash/correlation near-duplicate merging
- leakage checks = 0 (groups, files, external files)

**Panel 2: "Train" (result-independent).**
Small schematic: blur → CLAHE → low-pass → DenseNet-121 → 6 classes. Caption "selected on validation macro-F1; test and external evaluated once".

**Panel 3: "Internal vs external" (result-dependent; fill from results.json).**
Two side-by-side bars or gauges:
- left, "Internal test (same sources)": accuracy `TestAcc`% and macro ROC-AUC `TestMacroAUC`;
- right, "Unseen hospital (Montgomery, normal vs TB)": sensitivity `ExtSensK/ExtSensN` **with** specificity `ExtSpecK/ExtSpecN`, and exact TB recall `ExtTBK/ExtTBN`.
A small stacked bar under the right gauge shows the predicted classes for Montgomery's normal and TB scans (same data as `fig_external_breakdown`): for the final model most scans fall into Pneumonia and Pulmonary Nodule / Mass, classes the cohort does not contain, and the Tuberculosis segment is `ExtTBK` of `ExtTBN`.

Headline of panel 3: **"High internal scores; most healthy patients from an unseen hospital flagged abnormal."** Optional one-line footnote under the panel, from `results.json`: "Source of an image predictable from frozen features: `ProbeImgBalAcc`% balanced accuracy (chance `ProbeImgChance`%)". Do not show the run-1 vs run-2 comparison in the graphical abstract (it needs the shared-sources caveat, which does not fit).

## Title strip
"Internal accuracy is not generalisation" (short form of the manuscript title).

## Rules
- No clinical claims, no "AI diagnoses TB" wording.
- External sensitivity never shown without specificity.
- Colour-blind-safe palette; same class colours as the manuscript figures.
- Single-run caveat in small print: "single training run (seed 42); no clinical validation".
