# Results facts for the fill pass (2026-11-03, ~17:00)

Interpretation statements that writers may now use in place of `\pending{}` / `[PENDING FINAL RESULTS]`. Every number must still come from `results_macros.tex` / `results.json` (macro names given). The sources are generated files; nothing here is a new number.

**Still pending until ~07:30 on 2026-11-04 (keep their placeholders):** seed variability (runs 3 and 4: `\RMS…` macros), the preprocessing ablation (run 5: `\RAbl…` macros). Write these sentences as `\pending{seed variability}` / `\pending{ablation}` or `[PENDING: seed runs]`, `[PENDING: ablation]`.

## F1. Which model is final and why
- The served/final model is **v6.1 = run 2** (`v6r2_densenet121_seed42`, build `v6_build_002`, seed 42, checkpoint `\RCheckpointShort`), selected over run 1 by the rule fixed before run 2's results were seen (`docs/data/run_selection_protocol.md`).
- Common selection set: `\RCmpN` validation images that neither build used for training or testing. Macro-F1 run 1 `\RCmpRunOneMacroFOne`% vs run 2 `\RCmpRunTwoMacroFOne`%; accuracy `\RCmpRunOneAcc`% vs `\RCmpRunTwoAcc`%; macro ROC-AUC `\RCmpRunOneMacroAUC` vs `\RCmpRunTwoMacroAUC`. McNemar discordant pairs: run 1 right/run 2 wrong `\RCmpMcNemarB`, the reverse `\RCmpMcNemarC`, exact p = `\RCmpMcNemarP` (descriptive).
- **Caveat that must be stated:** on the `\RCmpRunOneSharedN` common images from the four sources both runs trained on, accuracy is almost equal (`\RCmpRunOneSharedAcc`% vs `\RCmpRunTwoSharedAcc`%) and the macro-F1 gap is smaller (`\RCmpRunOneSharedMacroFOne`% vs `\RCmpRunTwoSharedMacroFOne`%). Much of run 2's advantage is on TBX11K and COVID-19 Radiography images, which are in-distribution for run 2 only.
- Run 2 was started after run 1's external results were known (timeline in the protocol); the selection rule itself was fixed before run 2's results.

## F2. Training
- Run 2 completed its full schedule: `\REpochsTrained` epochs (2 head + 14 fine-tune), best validation macro-F1 at epoch `\RSelectedEpoch` (the last epoch; the validation curve was still rising slightly, so longer training might help — do not claim it would).

## F3. Internal test (evaluated once)
- Accuracy `\RTestAcc`% (95% bootstrap CI `\RTestAccLo`–`\RTestAccHi`), macro-F1 `\RTestMacroFOne`%, macro ROC-AUC `\RTestMacroAUC`, ECE `\RTestECE`.
- Strongest classes: COVID-19 (`\RFOneCovid`% F1), Tuberculosis (`\RFOneTB`%), Pneumonia (`\RFOnePneumonia`%). Weakest: Pulmonary Nodule / Mass (`\RFOneNodule`%), then Normal (`\RFOneNormal`%). Nodule/Mass and Pleural Effusion are both NIH-only classes with NLP-mined labels.
- Per-source accuracy differs (macros `\RSrcAcc…`): use the per-source table; do not rank sources by quality of data.

## F4. External cohort (Montgomery, evaluated once)
- Abnormal-vs-normal: sensitivity `\RExtSensK/\RExtSensN` (`\RExtSens`%), specificity `\RExtSpecK/\RExtSpecN` (`\RExtSpec`%), ROC-AUC of P(abnormal) `\RExtAUC`; exact Tuberculosis recall `\RExtTBK/\RExtTBN`.
- Montgomery scans were mostly assigned to Pneumonia and Pulmonary Nodule / Mass (fig_external_breakdown); only `\RExtCovidCalls` were called COVID-19.
- Comparison with run 1 (history, evaluated once each): run 1 called 71 of 138 Montgomery scans COVID-19 and 43/80 normals Normal; run 2 calls 3 COVID-19 and 19/80 Normal. Interpretation: adding a large second COVID-19 source removed the run-1 COVID-19 shortcut, but the external errors shifted to other abnormal classes; internal gains did not translate into better external specificity or any TB recognition. (Run-1 values: `artifacts/evaluation_reports/v6_external/metrics_external.json`; quote them as numbers with that source, they are not in the macros.)
- Low specificity means the model would flag most healthy patients from this hospital as abnormal.

## F5. Source confounding (control)
- Source-identifiability probe (`artifacts/data_quality/source_probe_v6_build_002.json`; macros `\RProbe…`): a linear classifier on frozen features predicts the source of a test image with balanced accuracy `\RProbeImgBalAcc`% across `\RProbeImgNSources` sources (chance `\RProbeImgChance`%) using ImageNet features, and `\RProbeMdlBalAcc`% using the trained model's backbone. Within a single class the source is still identifiable: Normal `\RProbeImgWithinNormal`% (chance `\RProbeImgWithinNormalChance`%, `\RProbeImgWithinNormalNSources` sources), Tuberculosis `\RProbeImgWithinTB`%, Pneumonia `\RProbeImgWithinPneumonia`%, COVID-19 `\RProbeImgWithinCovid`%.
- Interpretation allowed: source and class are confounded; source information is linearly available in the features the classifier uses, so a source shortcut is possible. Not allowed: "the model uses source cues" as a proven statement. Support it only jointly with F4 and F6, and call the evidence indirect.
- Fine-tuning changed only the last 17 weighted backbone layers, so the model's features stay close to ImageNet's (verified by weight comparison).

## F6. Grad-CAM (qualitative)
- Pre-specified selection: one correctly classified test image per class at the median confidence (rule in `lungai/evaluation/make_figures.py::fig_gradcam_examples`, fixed before viewing heatmaps).
- Observation: COVID-19 and Normal heatmaps concentrate on the neck/trachea, shoulders and image borders; Nodule/Mass and Tuberculosis heatmaps fall inside the lung fields; Pneumonia (paediatric Kermany image) is diffuse. One image per class: illustrative only.

## F7. Application
- CXR gate: heuristic threshold recalibrated 0.83 → 0.703 on validation radiographs; rejections of real test radiographs 63.6% → 9.3% (build 2 test), Montgomery 60.1% → 13.0%; all synthetic non-radiographs still rejected; no real non-radiograph set (`artifacts/app_checks/*.json`).
- Tests: Python suite green for v6.1 (≈150 passed, 1 skip that applies only without a checkpoint); browser end-to-end check of v6.1 pending at time of writing (main session will update).

## F8. What the work does and does not show (for conclusions)
- Shows: with leakage-controlled, patient-grouped splits, a standard DenseNet-121 reaches the internal results above, and those results overstate behaviour on an unseen hospital; class–source confounding is strong in public CXR collections; adding sources changes which shortcut appears rather than removing the external gap.
- Does not show: clinical utility; superiority over other methods; that the preprocessing helps (pending ablation); robustness across seeds (pending).
