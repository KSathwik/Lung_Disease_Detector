# Validation, Discussion and Limitations

## Validation protocol

Validation here means the checks that make the reported numbers trustworthy, not only the metrics themselves. They fall into five groups.

**Data validation.** The build stops when patient groups, identical files or external-cohort files span splits. Every image has recorded hashes, and the split has a fingerprint. These checks are evaluated for both builds in [[tab:checks]]. Separation is proven only for rows with known patient identity; the build report states how many rows have unknown identity.

**Process validation.** The trainer reads only the training and validation splits, and a run records the build and split fingerprints. The evaluator refuses to compute metrics if its input is inconsistent with the manifest, if a split or checkpoint is mixed, or if stored class indices disagree with the stored probabilities. The test split and the external cohort require an explicit final flag. A script that publishes the metrics to the application refuses to combine reports of different checkpoints.

**Code validation.** The test suites check the metric implementations against hand-computed values and scikit-learn, the preprocessing against a frozen copy, the evaluator against tampered inputs, and the splitting rules against the known leakage of V5 (the test that pins the 154 leaked images keeps the defect detectable). Appendix D summarises the results.

**Result validation.** Per-image predictions are saved, so every reported number can be recomputed. The documentation is generated from the served metrics file, and a test fails if a number in the documentation differs from it. This report reads its numbers from a generated JSON file at build time.

**Application validation.** The scores displayed by the frontend were compared with the API response in a manual browser check, and the application states the metrics of the checkpoint it loaded.

### The V5 audit as a validation case

The audit of the earlier pipeline is itself a validation result. [[tab:v5]] compares what V5 recorded with what the audit found. The comparison shows how the same architecture can produce an attractive internal accuracy and a useless external behaviour when the data are not controlled.

Table: Recorded results of the earlier V5 model and what the audit found. These numbers describe a superseded model whose checkpoint is lost; they are history and are not comparable with the V6 results. {#tab:v5}

| Aspect | V5 as recorded | Audit finding |
|---|---|---|
| Internal test accuracy | {{VFiveAcc}}% on {{VFiveN}} images | 154 test images (9.8%) share a patient with training images |
| External specificity | {{VFiveExtSpecK}} / {{VFiveExtSpecN}} | The model called no normal scan Normal |
| External sensitivity | 58 / 58 | Quoted without specificity, it looked perfect |
| Pneumonia class | 2,805 images | 1,395 were TBX11K "sick but not TB" images |
| Tuberculosis class | 1,348 images | 124 were latent TB |
| Checkpoint | Recorded SHA-256 | The file is lost; no per-image predictions exist |

## Discussion

The main findings of the final model, {{ModelVersion}}, are discussed in five points.

**Internal and external performance.** The internal test accuracy is {{TestAcc}}% on a split that is free of known patient and duplicate leakage, and the macro ROC-AUC is {{TestMacroAUC}}. On the external cohort the same model calls {{ExtSpecK}} of {{ExtSpecN}} normal scans Normal and none of the {{ExtTBN}} tuberculosis scans Tuberculosis. Removing leakage therefore does not remove the gap: a clean internal split measures performance on the same sources, not on a new hospital.

**Weak classes.** Pulmonary Nodule / Mass (F1 {{FOneNodule}}%) and Normal ({{FOneNormal}}%) are the weakest. Nodule / Mass and Pleural Effusion come from NIH alone, with labels mined from reports, and are confused with Normal and with each other. Label noise is a plausible contributor but was not measured.

**Source shortcuts.** For COVID-19, the first run's external behaviour ({{RunOneExtCovidCalls}} of {{NExternal}} Montgomery scans assigned COVID-19) suggested a shortcut tied to its single source. The second run largely lost that behaviour with build 2, which changed several sources at once, but the errors moved to other abnormal classes. For Tuberculosis, the good internal result does not appear on Montgomery. The probe shows that the source is predictable from the features, also within a class, and the COVID-19 and Normal heatmaps fall outside the lung fields. This evidence is indirect and consistent with, but does not prove, the use of source cues.

**Comparison of the two runs.** Run 2 was better on the common validation set ({{CmpRunTwoMacroFOne}}% against {{CmpRunOneMacroFOne}}% macro-F1), but on the images from sources both runs trained on, the accuracy is almost equal ({{CmpRunTwoSharedAcc}}% against {{CmpRunOneSharedAcc}}%). Much of the gain is on sources only run 2 saw. Each run is a single seed, but the seed study (Chapter 7) shows that seeds of the run-2 recipe differ by only {{MSTestMacroFOneSD}} points of macro-F1 (standard deviation over three seeds); the run-1 recipe was not repeated.

**Calibration.** The expected calibration error is {{TestECE}} on the internal test split, so the displayed scores are reasonably calibrated there. This was not checked externally, no calibration was fitted, and a score is not a probability of disease.

### Findings that do not depend on the final model

Three observations hold whichever model is selected.

First, the dominant cost of a trustworthy result was not the network but the data. Most of the work went into acquisition, label rules, duplicate handling and checks. The training itself is a standard transfer-learning recipe.

Second, the external cohort contains only normal and TB scans. It can say whether the model separates abnormal from normal at a hospital it has not seen, and whether it can name TB, but it cannot assess the other four classes. Any statement about those four classes outside the internal test set would be unsupported.

Third, sources define classes in this benchmark. A class with a single type of source (COVID-19 from web-collected sets, TB mostly from two screening sets) is exposed to the risk of being recognised by image style. Per-source reporting is therefore not an extra; it is required for an honest reading of the overall accuracy. The reasoning is supported by the literature on shortcut learning [@geirhos2020shortcut; @degrave2021ai; @zech2018variable], although the extent in this project is a result of the final model and is given in the discussion above.

## Threats to validity

- **Internal validity.** One run for the first model and three seeds for the served recipe (standard deviation of accuracy {{MSTestAccSD}} points, a rough estimate from three runs). The ranking of two runs that differ by a few points is not reliable. A learning curve that is still rising at the time limit means that the model is not converged.
- **Label validity.** NIH labels are mined from reports and not read from the images [@oakdenrayner2020exploring]. The Kermany and Cohen labels come from their sources. No re-reading by a radiologist was done.
- **Construct validity.** The single-label rule means that the test set contains clear single-finding cases. It overstates performance on the real mixture of findings.
- **External validity.** One external cohort, from one programme and with two classes. Population effects, scanners and prevalence are not controlled. No demographic data are available, so subgroup performance [@seyyedkalantari2021underdiagnosis] cannot be assessed.
- **Statistical validity.** Wilson intervals describe sampling error of the test set only. They do not include training variance, and several sources contribute few images to the test split.

## Limitations

The following limitations come from the project's limitation record. Those that depend on the final model's behaviour are listed at the end of the first group.

### Model and training

- CPU, time-boxed. Run-to-run variation is measured for the served recipe with three seeds on one split (a rough estimate), not for run 1, other splits or other architectures; the effect of the preprocessing is tested with a single ablation run at one seed. Differences of a few points are not evidence of a better model.
- The six classes are treated as mutually exclusive.
- Preprocessing steps run at native resolution, before the resize, so the same anatomy scanned at another resolution receives a different effective filter. Training and serving use the same code and therefore agree with each other.
- Grad-CAM shows which regions influenced the score. It does not show radiological reasoning, and saliency maps have been found to be unreliable as localisation evidence [@arun2021assessing; @ghassemi2021false].
- The softmax output is not a probability of disease. The 0.5 inconclusive threshold is not validated. The expected calibration error is reported but no calibration was fitted [@guo2017calibration].
- External behaviour: on Montgomery the model flags most normal scans as abnormal (specificity {{ExtSpecK}} of {{ExtSpecN}}), so it would flag most healthy patients of that hospital. The abnormal sensitivity ({{ExtSensK}} of {{ExtSensN}}) is not tuberculosis detection, because no tuberculosis scan is named Tuberculosis ({{ExtTBK}} of {{ExtTBN}}). Tuberculosis therefore does not transfer.
- Source shortcut: the first run assigned COVID-19 to many Montgomery scans; the final run does not, but its errors moved to other classes, and source information is predictable from its features. The Grad-CAM examples for COVID-19 and Normal fall outside the lung fields. The evidence is indirect.
- Weak classes: Pulmonary Nodule / Mass and Normal have the lowest F1, and their labels, like those of Pleural Effusion, come from report mining.
- Per-source differences: accuracy differs between sources in the test split, so overall accuracy depends on the source mix.
- Much of the gain of the final run over the first is on sources only the final run trained on.
- Run 2 was started after run 1's external results were known; only the selection rule was fixed beforehand.

### Data

- Patient identity is unknown for TBX11K and the COVID-19 Radiography Database. Their images are grouped one per image plus exact and near-duplicates, so the same patient could still appear in two splits. For NIH, Kermany, Shenzhen and Cohen the grouping uses real or file-name-derived identifiers.
- Images with more than one finding are excluded, so the test set contains only clear single-label cases.
- The Kermany images are paediatric (one to five years); all other sources are adult. No demographic data are available.
- VinDr-CXR and JSRT were not available, so Pleural Effusion and Nodule / Mass come from NIH alone.
- The images are not redistributed. The repository holds the rules, the builder and the download provenance.
- A contamination risk recorded for the earlier pipeline (an aggregated TB archive reported to contain Montgomery images) could not be ruled out by content matching. V6 does not use that archive, and every development image is checked against the external cohort by content.

### Application

- The input gate is heuristic. After recalibration it rejects {{CxrGateBTwoNewTestRejPct}}% of the real test radiographs of build 002 and {{CxrGateBTwoNewExtRejPct}}% of Montgomery radiographs, down from {{CxrGateBTwoOldTestRejPct}}% and {{CxrGateBTwoOldExtRejPct}}% at the original threshold on the same build-2 test split and the same Montgomery set. Its rejection of real non-radiographs is not established, because only seven synthetic negatives were available.
- The "typical features" and "precautions" texts are generic per class, not findings read from the image.
- Authentication is a single shared key or loopback-only access. There are no user accounts, no audit log, and uploads are kept indefinitely.
- The Docker image could not be built on the development laptop, and continuous integration has not run because the branch was not pushed.

## Ethical and clinical considerations

The system is not a medical device and was not clinically validated. It must not be used to decide a diagnosis or treatment. The application carries a notice with every prediction. The datasets were used under their published terms, and no identifiable patient data were handled beyond what the datasets publish. Reporting of sensitivity without specificity, which is the failure that hid the V5 problem, is prevented in the code of the application. Any future clinical study would need ethics approval, representative local data and prospective evaluation.
