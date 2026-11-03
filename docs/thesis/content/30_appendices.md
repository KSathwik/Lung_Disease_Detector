# API Endpoints

The base path is /api/v1. Interactive documentation is served at /docs when the backend runs. If the environment variable LUNGAI_API_KEY is set, every route except the health route requires the header X-API-Key; otherwise only loopback clients are served. Failures return 401.

Table: Endpoints of the backend. {#tab:api}

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| GET | /health | Service status, model readiness, gate mode | 200 |
| GET | /model | Model identity, class mapping, expected checkpoint SHA-256, load status | 200, 401 |
| GET | /model-metrics | Generated evaluation results of the configured checkpoint, with evidence level and a flag for a mismatch | 200, 401 |
| POST | /predict | Multipart file with optional patient_id, scan_type, notes; returns the prediction | 200, 400, 401, 404, 413, 422, 500, 503 |
| GET | /predictions | List stored predictions (skip, limit) | 200, 401 |
| GET | /predictions/{prediction_id} | One stored prediction with its six probabilities | 200, 401, 404 |
| POST | /reports/generate/{prediction_id} | Text report for a stored prediction | 200, 401, 404 |
| GET | /reports | List reports | 200, 401 |
| GET, POST, PUT | /patients | Patient records | 200, 201, 401, 404, 409, 422 |

The response of a successful prediction contains: the prediction and scan identifiers; the model identifier, version, display name and checkpoint SHA-256 with the verification flag; the class order; the predicted class and index; the confidence (top score) and the inconclusive flag; the six probabilities keyed by class name; the Grad-CAM overlay, original image, target class and disclaimer; a summary block with urgency, alternative conditions, key findings and precautions; the external-generalisation limitation notice; and the creation time.

Table: Error bodies of the prediction endpoint. {#tab:errors}

| Status | Body |
|---|---|
| 422, gate | detail with error INVALID_CXR, a message, a reason and the score |
| 503 | detail with error MODEL_NOT_LOADED, a message and a path-free reason |
| 400, 404, 413, 500 | detail with a message string |

# Configuration

The single configuration file lungai/configs/lungai_densenet121.json defines the model. [[tab:config]] lists its main groups.

Table: Main groups of the model configuration. {#tab:config}

| Group | Content |
|---|---|
| Identity | Model identifier, display name, version, architecture description |
| Classes | Six class identifiers and names in the fixed order, indices 0 to 5 |
| Layer names | gap, bn, dense_features, dropout, class_predictions and the Grad-CAM convolution name |
| Preprocessing | Blur kernel and sigma, CLAHE clip limit and tiles, low-pass sigma and kernel, interpolation, target size, ImageNet statistics |
| Training | Batch size, learning rates, number of unfrozen layers, phases, seed, checkpoint selection rule |
| Augmentation | Flip probability, rotation range, border mode |
| Checkpoint | File name and expected SHA-256 |
| Inference | Confidence floor 0.5, input gate parameters (model threshold 0.83, heuristic threshold 0.703) |
| Dataset | Description of the V6 build, split fingerprint; the V5 description is kept under a historical key |

The dataset rules are in lungai/configs/dataset_v6_proposal.json: the seed 20261102, the split fractions 0.70, 0.15 and 0.15, the stratification keys, the near-duplicate thresholds (at most 6 bits of perceptual hash difference and thumbnail correlation of at least 0.97), the per-source label rules and the sampling caps.

Environment variables that change behaviour: LUNGAI_API_KEY (authentication), LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT (permits a checkpoint whose hash differs, flagged in every response) and MODELS_DIR (location of the checkpoint).

# Reproducibility Commands

The images are not distributed, so the data steps need a download. The commands below are those used by the project; replace the angle-bracket placeholders.

```
# 1. Acquire a source (provenance is logged under data/raw/_provenance)
python scripts/data/acquire.py <source> --execute

# 2. Build the dataset (rules, caps, patient-grouped split, leakage checks)
python scripts/data/build_v6_manifest.py --raw-root data/raw --out data/derived/<build> --sources <ids...>

# 3. Train (train and validation only)
python -m lungai.training.train_v6 --build data/derived/<build> --run-name <run> --max-hours <H>

# 4. Evaluate: validation, then test and external once with --final
B=data/derived/<build>; C=runs/<run>/model.h5; R=artifacts/evaluation_reports
python -m lungai.evaluation.evaluate_v6 --build $B --checkpoint $C --split val --out $R/<prefix>_val
python -m lungai.evaluation.evaluate_v6 --build $B --checkpoint $C --split test --final --out $R/<prefix>_test
python -m lungai.evaluation.evaluate_v6 --build $B --checkpoint $C --split external --final --out $R/<prefix>_external

# 5. Publish metrics to the application and regenerate the documentation
python -m lungai.evaluation.served_metrics --eval-prefix <prefix> --run runs/<run>
python -m lungai.evaluation.render_docs

# 6. Install the checkpoint (only if its SHA-256 matches the configuration)
python scripts/install_checkpoint.py --from runs/<run>/model.h5

# 7. Tests
python -m pytest
cd frontend && CI=true npx react-scripts test --watchAll=false
cd frontend && CI=true npm run build

# 8. This report
python docs/thesis/build_report.py
```

A rebuild from the same sources, rules and seed reproduces the split; its fingerprint is compared with the one in the build report. Training uses one seed and is not bit-deterministic, so small differences between repeated runs are expected. Steps that need the images or the checkpoint cannot be reproduced from the repository alone. Re-running steps 5 and 8 from saved reports is possible.

# Test Summary

The tests run on Windows 11 with Python 3.12.4, TensorFlow 2.16.1 and Node 18. The Python suite was re-run after the final model (v6.1) was integrated; the frontend figures below were recorded on 2 November 2026.

Table: Test results recorded on the project branch. {#tab:tests}

| Suite | Result |
|---|---|
| Python tests (package and backend) | Python suite: all tests pass except one skip that applies only when no checkpoint is installed |
| Browser end-to-end check of v6.1 (2026-11-03, backend serving the production build on 127.0.0.1:8001) | Passed: Model Performance page equals `/api/v1/model-metrics` in every displayed value; a test-split Tuberculosis radiograph was accepted by the CXR gate and predicted Tuberculosis, the six displayed scores equal the API probabilities, version v6.1 and the verified checkpoint hash were returned, the Grad-CAM overlay rendered, and the limitation notice quoted the generated v6.1 metrics; the stored record holds version and hash. The version label on the History page was added after this check and is covered by an API test, not re-checked in the browser |
| Frontend Jest tests (errors, prediction, metrics modules) | 18 passed |
| Frontend production build, warnings as errors | Compiled successfully |
| V5 manifest validation | Runs; overall FAIL, which is the expected result because it detects the known V5 patient leakage |
| Recorded V5 results check | 5 of 5 runs consistent, 0 mismatches |

The remaining skip applies only when no checkpoint is installed. Image-level checks, training and evaluation on the real data are not automated tests because they need the images.

Table: What the test suites cover. {#tab:coverage}

| Area | Covers |
|---|---|
| Configuration | Fixed class order; consistency with recorded mappings; invalid configurations rejected |
| Preprocessing | Equality with a frozen copy on five sizes and channel layouts; bytes path equals file path; missing, corrupt and tiny inputs raise; evaluation never augments |
| Model | Six-way softmax; layer names; save and load round trip; SHA-256 refusal |
| Metrics | Hand-computed confusion values; agreement with scikit-learn; Wilson intervals; ECE; invalid inputs rejected |
| Evaluation | Recompute from a predictions file; tampering and mixed checkpoints detected; external binary task; final flag required |
| Data | Schema; split counts; known V5 leakage pinned; duplicates; external isolation; deterministic grouped split with no group spanning partitions |
| Consistency | No competing model names or re-declared class lists; documentation numbers match the generated files |
| Backend API | Persistence, 503, 400, 413, 422, 500 behaviours, path traversal, inconclusive state, authentication, gate, metrics and health content, database migration |
| Backend and model | Hash refusal, bad checkpoints, API probabilities equal the raw softmax, Grad-CAM wiring, determinism |

# History of the V5 Audit

This appendix records what the audit of the earlier model found, for the record. The V5 numbers describe a superseded model whose checkpoint is lost, and must not be compared with V6 results.

**Dataset.** V5 unified eight public sources into 10,547 images (7,398 training, 1,579 validation, 1,570 test; Montgomery as a 138-image external cohort). It was built in stages over several versions, each keeping the split of the patient identifiers it inherited. For most sources the patient identifiers were synthetic and one per image, so the split was effectively by image.

**Findings.**

1. Patient leakage: 289 derived patient groups spanned splits; 154 test images (Kermany Pneumonia 110, Normal 44) and 191 validation images shared a patient with training images.
2. Unknown identity: 8,478 rows had synthetic identifiers.
3. Proxy labels: 1,395 TBX11K "sick but not TB" images were labelled Pneumonia (49.7% of the class); 124 latent-TB images were labelled Tuberculosis; 1,440 of 1,467 VinDr images carried two or more findings.
4. A near-duplicate pair crossed validation and test with conflicting labels; the stored perceptual hash was not informative for near-duplicate detection.
5. Confounding: COVID-19 came from a single source and Pleural Effusion was 88% VinDr.
6. External failure: on Montgomery, 0 of 80 normal scans were called Normal while all 58 TB scans were flagged abnormal.
7. Possible contamination of the external cohort by an aggregated TB archive, which could not be resolved by content matching.
8. The checkpoint (SHA-256 recorded) is lost, and no per-image predictions were saved, so ROC-AUC, PR-AUC and calibration could not be verified.

**Variants.** Five variants (filter, empirical risk minimisation, Deep CORAL, domain-adversarial training, and filter with CORAL) were each trained once for four epochs; the kept checkpoint was the final epoch. The recorded results of the five runs were found internally consistent (29 checks per run), but their data were leaked.

**Consequence.** V6 was built to remove these defects: see [[tab:defects]].

# Reporting Checklist Mapping

Table: Mapping of the report to items of the CLAIM checklist [@mongan2020claim] and the TRIPOD+AI statement [@collins2024tripodai]. This is a self-assessment by the author of the report and has not been reviewed by others. {#tab:claim}

| Item | Where addressed |
|---|---|
| Study design and objectives | Chapters 1 and 3 |
| Data sources and access | Chapter 5, [[tab:licences]] |
| Inclusion and exclusion criteria | Chapter 5, [[tab:rules]] and [[tab:exclusions]] |
| Reference standard and labelling | Chapter 5; limitations on label origin in Chapter 8 |
| Data partitions and leakage control | Chapter 5, [[tab:checks]] |
| Preprocessing | Chapter 5 |
| Model, training and selection | Chapters 4 and 6 |
| Evaluation metrics and uncertainty | Chapters 4 and 7 |
| External validation | Chapter 7, [[tab:external]] |
| Failure analysis | Chapter 7, subsection on failure analysis; Chapter 8 |
| Explainability | Chapters 6 and 7 |
| Limitations and bias | Chapter 8 |
| Code and data availability | Appendix C |
| Ethics | Chapter 8 |
