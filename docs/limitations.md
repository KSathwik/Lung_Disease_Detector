# Limitations

## Current model

The served model is v6.1 (run 2, `v6_build_002`), chosen over run 1 by a rule fixed in advance and recorded in the repository before run 2's results ([data/run_selection_protocol.md](data/run_selection_protocol.md)). Run 2 itself was started after run 1's external (Montgomery) results had been seen, so Montgomery is not untouched by design decisions, although it was never used for training or for choosing the served model. Read with the generated results in [results/current_model.md](results/current_model.md) and `artifacts/evaluation_reports/run_comparison.json`; every number below comes from those files.

- **Three seeds, one ablation.** The served recipe was trained with three seeds (42, 43, 44; the served checkpoint is seed 42, chosen before the other seeds existed). Internal test results vary very little between seeds, external specificity varies more, and no seed names any Montgomery TB scan Tuberculosis (`artifacts/evaluation_reports/multi_seed_summary.json`). Three seeds give only a rough estimate of variability. The preprocessing ablation (no blur, CLAHE or low-pass; same split, seed and schedule) scored lower on internal accuracy and macro-F1 in that single comparison (see `artifacts/evaluation_reports/multi_seed_summary.json`) and did not change external specificity; it is one seed, and the run selected its last epoch (16 of 16, validation still rising), so a convergence-speed difference cannot be excluded.
- **External behaviour.** On Montgomery (138 scans from a hospital not used in training) the model flags almost all TB scans as abnormal but calls only about a quarter of the normal scans Normal, and it does not name any TB scan Tuberculosis; most Montgomery scans are assigned Pneumonia or Pulmonary Nodule / Mass (external prediction breakdown). The "abnormal" sensitivity is therefore not TB detection, and the low specificity means it would flag most healthy patients from this hospital.
- **COVID-19 calls on Montgomery fell, the external problem did not.** Run 1 (COVID-19 only from the Cohen collection) assigned COVID-19 to 71 of 138 Montgomery scans; run 2 assigns it to 3. Build 2 changed several sources at once (it adds the COVID-19 Radiography Database, TBX11K and two NIH archives), so this is consistent with the added COVID-19 source removing a source cue, but no ablation isolates it. Run 2's errors on Montgomery moved to other abnormal classes instead of becoming correct.
- **Grad-CAM points outside the lungs for some classes.** In the pre-specified Grad-CAM examples (`docs/figures/fig_gradcam_examples.png`, one median-confidence correct test image per class) the COVID-19 and Normal heatmaps concentrate on the neck, shoulders and image borders rather than the lung fields. This is qualitative, but consistent with source or acquisition cues being used.
- **Much of run 2's gain over run 1 is on sources only run 2 trained on.** On the common selection set, run 2's macro-F1 was far higher overall; restricted to the four sources both runs trained on, the two runs had nearly the same accuracy and a smaller macro-F1 gap (`run_comparison.json`, `shared_sources_only`).
- **Tuberculosis does not transfer.** Internal TB scores are high (TB from TBX11K and Shenzhen), but on Montgomery the TB class is not predicted.
- **Weak classes.** Pulmonary Nodule / Mass has the lowest internal F1, followed by Normal; NIH labels for these classes are mined from radiology reports by NLP, not read from the images.
- **Source differences inside the test split.** Accuracy differs by source (per-source table); overall accuracy depends on the source mix.
- **Uncalibrated scores.** The softmax output is not a probability of disease; the 0.5 "inconclusive" threshold is not validated. The expected calibration error is reported, but no calibration was fitted.

## Data (V6)

- **Patient identity is unknown for TBX11K and the COVID-19 Radiography Database.** When a build includes them, their images are grouped one per image (plus exact and near-duplicates), so the same patient could still appear in two splits. For NIH, Kermany, Shenzhen and Cohen the grouping uses real or filename-derived patient IDs (the build report's `patient_identity` counts say which applies).
- **Single-label assumption.** Images with more than one finding are excluded (NIH) or not used, so the test set contains only clear single-label cases; real radiographs often show several findings.
- **Population.** The Kermany images are paediatric (1â€“5 years); all other sources are adults. No demographic data is available.
- **Blocked sources.** VinDr-CXR (PhysioNet credentialing) and JSRT (registration) were not available, so Pleural Effusion and Nodule/Mass come from NIH only.
- **Images are not distributed.** Licences do not allow redistribution; the repository holds the rules, the builder and download provenance (URL, size, SHA-256), not the images.

## Model

- **Single-label assumption:** the six classes are treated as mutually exclusive.
- **Resolution-dependent preprocessing:** the blur, CLAHE and low-pass steps run at native resolution before resizing, so the same anatomy scanned at another resolution receives a different effective filter. Training and serving use the same code, so they match each other.
- **Grad-CAM** shows which image regions influenced the score. It does not show radiological reasoning.

## Application

- **CXR gate is a weak heuristic.** The trained gate model is absent, so a heuristic is used (`cxr_gate_mode: heuristic_fallback` in `/health`). Its original threshold (0.83, meant for the trained model's probability) was found to reject most real radiographs (first measured as 1,096 of 1,724 build-1 test radiographs, `artifacts/app_checks/cxr_gate_rejection_v6_build_001.json`). Its own threshold was then set on build-2 **validation** radiographs (`scripts/data/calibrate_cxr_gate.py`) to **0.703**, the lowest value that still rejects every available non-radiograph test image. Like-for-like on the same build-2 **test** split it rejects **1,494 of 2,350** radiographs (63.6%) at 0.83 and **218 of 2,350** (9.3%) at 0.703; on Montgomery **83 → 18 of 138** (`cxr_gate_rejection_v6_build_002_heuristic083.json`, `…_heuristic0703.json`). The non-radiograph images used are synthetic, so rejection of real non-radiographs is not established, and real radiographs that score low are still refused (HTTP 422 `INVALID_CXR`). A gate trained on real radiographs and non-radiographs is the proper fix.
- **Static guidance:** the "Typical features" and "Precautions" texts are generic text for each class, not findings read from the image.
- **Authentication:** a single shared API key, or loopback-only access when no key is set. There are no user accounts and no audit log, and uploads are kept indefinitely.

## History: limitations of the V5 model (superseded)

- **Checkpoint lost; recorded results only.** No per-image predictions were saved, so ROC-AUC, PR-AUC and calibration could not be verified.
- **Patient leakage:** 154 of 1,570 internal-test images (9.8%) shared a patient with training images.
- **Proxy and latent labels:** half of the Pneumonia class was TBX11K "sick but not TB"; 124 Tuberculosis labels were latent TB; 1,440 of 1,467 VinDr images carried two or more findings.
- **External failure:** on Montgomery, 0 of 80 normal scans were called Normal.
- **Montgomery contamination (R1), unresolved.** The Kaggle Rahman TB archive used for V5 is reported to contain NLM Montgomery images. A content check (MD5 and perceptual hash) found no match, but the same check also found none for Shenzhen, which that archive is described as containing, so the method cannot rule the overlap out ([artifacts/data_quality/external_overlap_rahman_tb/README.md](../artifacts/data_quality/external_overlap_rahman_tb/README.md)). V6 does not use that archive.
