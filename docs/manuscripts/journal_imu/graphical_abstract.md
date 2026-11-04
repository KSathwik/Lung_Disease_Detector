# Graphical abstract: description of the intended figure (IMU)

IMU's guide has a graphical-abstract heading, but the requirement wording was not captured on 2026-11-03; check the live Guide for Authors for size and format. Not drawn yet. Produce it from the same assets as the result figures, with numbers taken from `../shared/results.json` (the same keys as the `\R...` macros), never typed by hand.

## Layout (left to right, three panels)

1. **Build (result-independent).** Six development sources plus Montgomery (set apart, "external hospital"), a funnel with the four controls: official download with SHA-256 provenance; labelling rules fixed before training; patient and near-duplicate grouping; leakage checks (identity known for some sources only).
2. **Measure (result-independent schematic).** Frozen DenseNet-121 features; three diagnostics: six frequency bands (A to F), linear erasure (INLP, LEACE), leave-one-source-out.
3. **Finding (result-dependent; from `results.json`).** Source balanced accuracy by band (`SpecMainTestPassASrcBA` to `...FSrcBA`) as six bars against the 16.7% chance line; beside it the source accuracy before and after erasure (`InlpImgTestSrcBAKZero`, `InlpImgTestSrcBAKStar`) with the disease macro-F1 change; beside that the held-out gaps (`LosoImg...Gap`). Headline: "Source information is in every band and entangled with disease information." Footnote: "Band-stop retraining not run: the rule fixed in advance returned NO-GO."

## Rules
- No clinical claims. External sensitivity never shown without specificity.
- Say "indirect evidence" for any statement about the trained model.
- Colour-blind-safe palette; reuse the colours of `fig_spectral_map` (source blue, disease orange).
- Small print: "frozen features, linear methods, one benchmark, one run; patient-identity caveat".
