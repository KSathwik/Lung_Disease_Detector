# LungAI Viva Guide

Student: Katkam Sathwik, M.Tech (CSE), Kakatiya University College of Engineering and Technology. Supervisor: Prof. Nagavelli Ramana.
Project: LungAI, a leakage-controlled DenseNet-121 system for six-class chest X-ray classification with Grad-CAM, a FastAPI backend and a React frontend.

How to use this guide. Part 1 is a one-page set of numbers to remember. Part 2 explains how to present. Part 3 holds about 70 expected questions with short answers you can say aloud. Part 4 is a checklist for the days before the viva. Every number comes from the repository files (mainly `docs/manuscripts/shared/results.json`, `RESULTS_FACTS.md` and `docs/presentation/slides.md`). If a question goes beyond these files, say "I would need to check that" and do not guess.

Words used in this guide. "Source" means the dataset an image comes from (for example NIH or Kermany). "Shortcut" means the model using a cue that is linked to the label in the data but is not the disease (for example the style of the source). "Internal test" means the test split of the same development data. "External" means Montgomery County, a hospital set that was never in training.

---

# Part 1. Key facts cheat sheet

## 1.1 Project in one sentence

A six-class chest X-ray classifier (COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis) rebuilt from official public sources after an audit of an earlier version (V5), evaluated once on an internal test split and once on an external hospital, with an analysis of source shortcuts. It is a research prototype, not a medical device.

## 1.2 Data

| Item | Value |
|---|---|
| Development sources (6) | NIH ChestX-ray14, Kermany (paediatric), COVID-19 Radiography Database, TBX11K, Cohen COVID-19 collection, Shenzhen |
| External cohort | Montgomery County: 138 images (80 Normal, 58 TB); never in training |
| Final build | `v6_build_002` (run 2 and later); run 1 used `v6_build_001` |
| Split (patient-grouped, 70/15/15) | Train 10,957; validation 2,348; test 2,350 (total 15,655) |
| Test class sizes | COVID-19 451, Normal 467, Effusion 444, Pneumonia 394, Nodule/Mass 449, TB 145 |
| Leakage checks | All zero (patient groups, identical files, external files across splits) |
| Identity caveat | Separation is proven only for rows with known patient identity. 4,130 TBX11K and COVID-19 Radiography rows are grouped per image. Genuine patient IDs: NIH and Cohen. Kermany IDs are derived from file names |
| Built-in confounding | Effusion and Nodule/Mass exist only in NIH (labels mined from reports by NLP); Pneumonia is mostly Kermany; COVID-19 comes from two web-collected sets |

## 1.3 Model and training

- DenseNet-121 (ImageNet weights), global average pooling, BatchNorm, Dense(256, ReLU), Dropout 0.3, Dense(6, softmax). Input 224 x 224 x 3.
- Preprocessing: Gaussian blur (sigma 0.8), CLAHE (clip 2.0, 8 x 8 tiles) on the lightness channel, Gaussian low-pass (sigma 1.0), Lanczos resize to 224 x 224, ImageNet normalisation.
- Phase 1: 2 epochs, backbone frozen, Adam, learning rate 1e-3. Phase 2: last 30 backbone layers trainable (17 carry weights), learning rate 1e-5, early stopping with patience 3 on validation macro-F1. Batch 32, class-balanced sample weights, horizontal flip and rotation up to 10 degrees.
- Served model v6.1 = run 2, seed 42. It trained all 16 epochs (2 + 14); best validation macro-F1 at epoch 16. CPU only (8 GB laptop), about 20 to 26 minutes per epoch.
- Checkpoint selected on validation macro-F1 only. Test and Montgomery are evaluated once.

## 1.4 Internal test (served model, evaluated once)

| Measure | Value |
|---|---|
| Accuracy | 77.4% (Wilson 95% CI 75.7 to 79.0; bootstrap 95% CI 75.8 to 79.1) |
| Macro-F1 | 78.8% |
| Macro ROC-AUC / PR-AUC | 0.955 / 0.849 |
| Expected calibration error | 0.020 |
| F1 per class | COVID-19 95.2, TB 85.7, Pneumonia 85.4, Effusion 75.6, Normal 70.8, Nodule/Mass 60.1 |
| Per-source accuracy | COVID-19 Radiography 99.5, Kermany 94.7, TBX11K 94.7, Shenzhen 80.8, NIH 62.5, Cohen 50.5 |

## 1.5 Seeds and ablation

- Seeds 42, 43, 44 (same split and recipe): accuracy mean 77.1% (SD 0.3, range 76.9 to 77.4); macro-F1 mean 78.4% (SD 0.4); macro-AUC 0.954 (SD 0.001). Three runs only, so the SD is rough. The served run is the highest of the three, within a spread of under one percentage point.
- Montgomery specificity across seeds: mean 27.1% (SD 3.8, range 23.8 to 31.2). Exact TB recall 0/58 in every seed.
- Ablation (run 5, seed 42, no blur, CLAHE or low-pass): accuracy 74.1%, macro-F1 75.8%, AUC 0.949 (below the whole seed range). Montgomery specificity 23.8% (same as run 2); exact TB 5/58 (8.6%), one run, observation only. Run 5 selected its last epoch (16 of 16), so convergence speed cannot be excluded.

## 1.6 External cohort (Montgomery, served model, evaluated once)

- Sensitivity (TB scans called abnormal): 55/58 = 94.8% (95% Wilson CI 85.9 to 98.2).
- Specificity (normal scans called Normal): 19/80 = 23.8% (CI 15.8 to 34.1).
- Binary AUC of P(abnormal): 0.809. Exact TB recall: 0/58. COVID-19 calls: 3 of 138.
- Run 1: 71 of 138 scans called COVID-19; 43/80 normals called Normal; binary AUC 0.730; 1 TB scan named TB.
- Reading: internal results overstate behaviour on an unseen hospital. Low specificity means most healthy scans from this hospital would be flagged abnormal.

## 1.7 Run selection (run 1 versus run 2)

Common validation set of 2,018 images: macro-F1 59.4% (run 1) versus 79.6% (run 2). On the 1,392 images from the four sources both runs trained on: accuracy 67.6% versus 67.8%, macro-F1 62.2% versus 68.5%. So much of run 2's lead is on sources in-distribution for run 2 only. Run 2 was started after run 1's Montgomery results were seen. The selection rule was fixed in advance and recorded in the repository (`docs/data/run_selection_protocol.md`), never "pre-registered".

## 1.8 Source probe (control)

A linear classifier on frozen 1,024-d features predicts the source of a test image with balanced accuracy 93.8% over 6 sources (chance 16.7%), both with ImageNet features and with the trained backbone. Within one class: Normal 97.2% (5 sources, chance 20%), TB 98.9% (3 sources), Pneumonia 89.1% (3 sources), COVID-19 89.0% (2 sources). Meaning: source information is available in the features. It does not prove the trained model uses it.

## 1.9 Extension results (C1, C3, C4 and C2)

| Part | Headline numbers |
|---|---|
| C1 spectral map (six bands A to F) | Full image: source accuracy 93.8% (chance 16.7%), within-source disease macro-F1 68.6%. Band-pass source accuracy: A 64.4, B 60.4, C 68.9, D 74.6, E 90.0, F 93.4%. Within-source disease macro-F1 by band: 52.9 (B) to 67.5 (F). Removing any one band changes source accuracy by at most 2.9 points (band F). No band is the carrier |
| C3 INLP and LEACE | INLP, 20 steps: source accuracy 95.8% to 73.8% (not chance); within-source macro-F1 down 4.7 points; six-class macro-F1 down 12.4. LEACE (source at chance): six-class macro-F1 down 25.2, within-source down 8.8 |
| C4 leave-one-source-out (ImageNet features, linear head) | Held-out macro-F1: Cohen 12.0, Kermany 50.2, NIH 32.8, Shenzhen 33.1, TBX11K 22.7. Gap to in-distribution reference: TBX11K 72.3 points, Shenzhen 51.6, Kermany 42.0, Cohen 33.9. NIH gap is negative (-7.5) because of a metric artefact, not better generalisation |
| C2 band-stop retraining | NO-GO. No band met the rule fixed in advance. Best band F lowered source accuracy by 2.5 points (validation) and 2.7 (test), below the 5-point threshold. No C2 number exists; it is future work |

## 1.10 V5 audit (why the rebuild)

V5 recorded 82.93% internal accuracy (1,570 test images). The audit found: 154 test images (9.8%) shared a patient with training; 1,395 TBX11K "sick but not TB" images were labelled Pneumonia (49.7% of that class); 124 TB labels were latent TB; 8,478 rows had synthetic one-per-image IDs; on Montgomery 0 of 80 normal scans were called Normal; the V5 checkpoint is lost.

## 1.11 Application, tests, CI

- Backend FastAPI (`/api/v1`: predict, health, model, model-metrics, predictions, reports, patients); SQLite; checkpoint SHA-256 verified (without a matching file: HTTP 503 `MODEL_NOT_LOADED`). Frontend React: Analyze, History, Metrics, Patients pages.
- CXR input gate: heuristic (no trained gate model). Threshold recalibrated 0.83 to 0.703 on validation radiographs: validation pass rate 38.7% to 91.0%. Same build-2 test split: 1,494/2,350 rejected at 0.83 versus 218/2,350 at 0.703; Montgomery 83 to 18 of 138. Non-radiographs get HTTP 422 `INVALID_CXR`. Only synthetic non-radiographs were tested.
- Tests (run 2026-11-04): Python suite 173 passed, 1 skipped (the skipped test needs no checkpoint and applies only when none is present); Jest 19 passed; frontend production build succeeds. Re-run before the viva.
- CI: `.github/workflows/ci.yml`, GitHub Actions on every push and pull request: a Python job (Python 3.12, install, pytest) and a frontend job (Node 18, `npm ci`, Jest, build). Green on `main` at commit `f0ea013` (2026-11-04); check once more on the day.
- Docker: Dockerfile present (two-stage: Node build, Python 3.11 runtime) but never built, because Docker is not installed on the laptop.

## 1.12 Words never to use

"Pre-registered" (say: fixed in advance and recorded in the repository). "Proves the model uses the source" (say: indirect evidence, consistent with). "State of the art". "Clinically useful" or "ready for use". "Removed the shortcut". "Scanner fingerprint". "The filters improve generalisation". "Not a lucky seed". "First".

---

# Part 2. How to present to external examiners

## 2.1 The story in six steps

1. Problem. Chest X-ray classifiers look accurate in papers but often fail in a new hospital. A decision-support tool is only useful if its evaluation is honest.
2. Audit of V5. The earlier version reported 82.93% but had patient leakage, proxy labels and 0 of 80 normal scans correct on an external hospital. Its checkpoint is lost, so I rebuilt from the raw sources.
3. Leakage-controlled rebuild. Official sources with logged hashes, written label rules, patient-grouped splits, automatic leakage checks, test and external sets used once.
4. Honest external result. Internal accuracy is 77.4%, but on Montgomery specificity is 19 of 80 and no TB scan is named TB. Internal numbers overstate behaviour on a new hospital.
5. Source-shortcut analysis as the research contribution. The source of an image can be read from the features (93.8%, chance 16.7%). Source information is spread over all frequency bands, is entangled with disease information, and a linear head drops sharply on a held-out source. A band-removal retraining idea was tested on frozen features and was a NO-GO under rules fixed in advance.
6. What it means. The contribution is the evaluation method and the diagnostic evidence, not a new network or a clinical tool. The evidence about shortcuts is indirect.

Keep one sentence ready for the end: "My claim is about how to evaluate such a system honestly, not about clinical accuracy."

## 2.2 Recommended talk plan (about 20 minutes with the demo; slide numbers follow `docs/presentation/slides.md`)

| Block | Slides | Time | What to say | Transition to next block |
|---|---|---|---|---|
| 1. Opening, motivation, problem | 1 to 4 | 2.0 min | Introduce yourself and supervisor. Say the main point: not a high accuracy number but an evaluation that can be trusted. Six classes, single label. Research prototype, not a device. Objectives O1 to O6 in one breath | "Before I describe my method, I will explain why I had to rebuild it." |
| 2. Literature and gap | 5 to 6 | 1.5 min | Six groups of work in one sentence. The gap is evaluation hygiene (grouped splits, documented labels, external test), not a missing architecture. No novelty claim beyond that | "The clearest example of this gap was in my own earlier version." |
| 3. V5 audit | 7 | 1.5 min | 82.9% but 154 leaked test images, proxy labels, 0 of 80 external normals, lost checkpoint. This justifies the rebuild | "So V6 starts from raw sources. Here is the pipeline." |
| 4. Method and system | 8 to 16 | 3.5 min | Pipeline (8). App architecture in one sentence (9). Datasets and rules (10, 11): official sources, hashes, grouped split, leakage checks, identity caveat. Preprocessing (12). Model and two-phase training (13, 14). Evaluation protocol: once-only, Wilson CI, ECE, sensitivity always with specificity (15). Gate in two sentences (16) | "Now the results, first inside the data, then on a new hospital." |
| 5. Internal results | 17 to 19 | 1.5 min | Slide 17: 77.4% (75.7 to 79.0), macro-F1 78.8%, AUC 0.955. Slide 18 and 19 quickly: best COVID-19 and TB, weakest Nodule/Mass and Normal. Warn that internal numbers are not the full story | "These are internal numbers. The external cohort tells a different story." |
| 6. External result and control | 20 to 23 | 3.0 min | Slide 20: 19/80 specificity, TB 0/58, sensitivity only means "abnormal". Slide 21 (selection) in two sentences with the shared-sources caveat. Slide 22: probe 93.8% versus chance 16.7% and Grad-CAM, state "indirect evidence". Slide 23: three seeds and ablation, one line each | "This raised a measurable question: where does source information live?" |
| 7. Extension | 24 to 26 | 2.0 min | Slide 24: question and design (bands, INLP, LEACE, leave-one-source-out; rules fixed in advance). Slide 25: source readable in every band, no carrier band, held-out drop. Slide 26: erasure costs disease information; C2 is NO-GO, stated plainly | "Let me show the application working." |
| 8. Live demo | 27 | 3.0 min | See section 2.3 | "I will now state what this work does not show." |
| 9. Close | 28 to 31, 33 | 2.0 min | Limitations (28), conclusion (29), future scope (30), publication plan in one sentence (31). Skip slide 32 (references) unless asked. End with the one-sentence claim and invite questions | "Thank you. I am ready for questions." |

Total: 2.0 + 1.5 + 1.5 + 3.5 + 1.5 + 3.0 + 2.0 + 3.0 + 2.0 = 20 minutes. Practise with a timer. If you are told 15 minutes: shorten block 4 to 2.5 minutes, skip slides 18 and 21, reduce the demo to 1.5 minutes, and keep blocks 6 and 7 intact because they carry the contribution.

The numbers on the slides are filled from `results.json`, so they match this guide.

## 2.3 Three-minute live demo plan

Setup before the viva (do it early, see Part 4): backend running on http://127.0.0.1:8000, frontend on http://localhost:3000, checkpoint installed, one test image ready on the desktop. Use a test-split image, for example the Shenzhen TB radiograph shown on slide 27 (true label TB, predicted Tuberculosis in the earlier browser check).

Script:

1. (0:00 to 0:20) Open the Analyze page. Say: "The checkpoint is verified by SHA-256 before it loads. The page shows the model version, v6.1."
2. (0:20 to 1:00) Upload the image. Say: "The backend validates the file, the CXR gate checks that it is a chest radiograph, the same preprocessing as in training is applied, then the model predicts."
3. (1:00 to 1:40) Point to the predicted class, top score and the six score bars. Say: "These are softmax scores, not calibrated disease probabilities. Below 0.5 the page says Inconclusive."
4. (1:40 to 2:20) Show the Grad-CAM overlay. Say: "This shows image regions that influenced the score. It is not radiological reasoning, and here part of the attention is outside the lung fields. One correct example is not evidence of accuracy."
5. (2:20 to 3:00) Open the Metrics page. Say: "The page reads the same metrics file as this presentation, including the external result with specificity next to sensitivity." Optionally mention that a non-radiograph (for example a photo) is rejected by the gate with HTTP 422.

If the demo fails: do not debug in front of the examiners. Say "The live demo is not working, so I will show the recorded screenshots", then open the files in `docs/figures/screenshots/`: `01_analyze_upload.png`, `02_analyze_selected.png`, `03_analyze_result.png`, `04_gradcam.png`, `05_model_performance.png`, `06_history.png`, `07_result_card.png`, `08_gradcam_card.png`. Slide 27 already shows 07 and 08. Typical causes: no checkpoint (503 `MODEL_NOT_LOADED`), backend not started, wrong Python version (use 3.11 or 3.12), frontend proxy (the dev server proxies `/api` to port 8000). Keep the screenshots open in a second window.

## 2.4 How to present negative results with confidence

- State them as findings, in your own plain words: "The external result is poor, and I report it because it is the true behaviour of the model."
- Show you designed the test that revealed the problem: the once-only external cohort, the source probe and the rules fixed in advance exist so that bad news cannot be hidden.
- Pair every negative with what it teaches: the external gap shows internal metrics overstate; C2 NO-GO shows that removing one frequency band does not remove source information in frozen features; erasure costing disease signal shows source and disease information are entangled.
- Use the hedges the repository requires: "consistent with", "indirect evidence", "in these frozen features", "one run". Do not weaken the finding, and do not strengthen it either.
- A NO-GO under a rule fixed in advance is a valid, finished result. It shows the rule was real. Say: "The rule said no, so I did not run the training night."

## 2.5 If someone says "your accuracy is low"

Suggested answer (30 seconds): "Thank you. Three points. First, 77.4% is measured on a patient-grouped test split that was used once, so it is a trustworthy number; my earlier V5 reported 82.9% but 154 of its 1,570 test images shared a patient with training, and on a new hospital it called none of 80 normal scans normal. Second, I do not compare my number with other papers, because the test sets differ and many use random image-level splits. Third, accuracy is not the goal. The goal is an evaluation that shows what the model does on new data, and that evaluation says the model is not ready for clinical use. The 95% interval is 75.7 to 79.0 percent."

Do not defend the number as high. Do not claim that other papers are wrong.

## 2.6 What NOT to claim

- Clinical use or readiness. The model is a research prototype; no clinician-reviewed ground truth; no prospective evaluation.
- State of the art or superiority over other methods. No comparison with other papers' numbers was made, by design.
- A causal proof of a source shortcut. The probe shows availability. The evidence about use is indirect (probe, class-source confounding, Grad-CAM, external result).
- "Pre-registered". Say "fixed in advance and recorded in the repository". Nothing was registered with an outside body.
- That the filters improve generalisation, or that the low-pass filter removes TB information. The ablation is one seed.
- That Montgomery was untouched. It was never used for training or for choosing the served model, but run 2 was started after run 1's external results were seen.
- That a patient never appears in two splits. Separation is proven only for rows with known patient identity.
- That adding the COVID-19 Radiography source "removed the shortcut". Build 2 changed several sources at once.
- That INLP, LEACE or leave-one-source-out are new methods. They are published; here they are measurements.

## 2.7 Delivery and body language

- Speak slowly, about 120 words per minute. Pause after each number so the examiners can write it down.
- Look at the examiners, not the screen. Glance at the slide only to move on.
- Stand still; point at the figure, not at the text. Keep your hands visible and relaxed.
- When you do not understand a question, say "Could you please repeat the question?" or restate it: "Do you mean ...?" Take two seconds before answering.
- Answer in three steps: short direct answer, one reason or number, one honest limitation. Then stop.
- If you do not know: "I did not test that, so I cannot say. My expectation would be ..., but it is untested." Never invent a number.
- Keep a printed copy of the cheat sheet (Part 1) beside you if the examiners allow notes.
- Thank the examiner for a criticism that is fair. Do not argue.

---

# Part 3. Expected viva questions and model answers

Answers are written to be spoken. Numbers are from `results.json`. Where an answer says "check", look at the cited file before the viva.

## 3.1 Motivation and problem

**Q1. Why did you choose this problem?**
Chest X-ray is the most widely available imaging test for lung disease, but trained radiologists are not available everywhere. Public datasets make decision-support research possible. However, models often lose performance in a new hospital. So my motivation is twofold: build a working tool and make its evaluation honest enough that the numbers mean something.

**Q2. What exactly is your problem statement?**
Given one frontal chest X-ray, assign it to exactly one of six classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass or Tuberculosis. The harder part is methodological: traceable labels, no known patient crossing the splits, and an external cohort to test behaviour on a new hospital.

**Q3. What are your objectives?**
Six main objectives. O1 and O2: rebuild the data from official sources with written label rules and patient-grouped splits with leakage checks. O3 and O4: train DenseNet-121 reproducibly and evaluate once on test and on an external cohort, with confidence intervals. O5: an end-to-end application with FastAPI, React, Grad-CAM and SQLite. O6: report limitations without overclaiming. The thesis adds a further objective, the analysis of source information.

**Q4. What is your contribution, since you did not invent a new network?**
The contribution is a leakage-controlled, auditable pipeline from raw data to served metrics; evidence that internal metrics overstate behaviour on a new hospital; and a measurement study of source shortcuts: where source information lives in the features and how entangled it is with disease information. Only the band-by-band comparison of source and disease information looks new, and I say that only "to our knowledge, within the searches described". I make no claim of a new method, state of the art, or clinical value.

**Q5. Why these six classes?**
They are the classes that official public sources can supply with rules written in advance. VinDr-CXR and JSRT could not be obtained in time (credentials and registration), so Pleural Effusion and Nodule/Mass come from NIH only. The six classes are treated as mutually exclusive, which is itself a limitation because real X-rays often show several findings.

**Q6. Can your system be used to diagnose patients?**
No. It is a research prototype and not a medical device. Its outputs are uncalibrated softmax scores. The external result shows that it would flag most healthy patients from an unseen hospital as abnormal and never names TB on that hospital's TB scans. Any clinical use would need ethics approval, local data and prospective evaluation by clinicians.

## 3.2 Literature

**Q7. What are the main related works?**
Large public datasets: ChestX-ray8/14 (Wang et al.), CheXpert, MIMIC-CXR, VinDr-CXR. TB and pneumonia sets: Shenzhen and Montgomery (Jaeger et al.), TBX11K (Liu et al.), Kermany. COVID-19 classifiers: COVID-Net, Chowdhury et al. Tools: DenseNet (Huang et al.), ImageNet, Grad-CAM (Selvaraju et al.). Pitfalls: Zech et al. on hospital-specific variation, DeGrave et al. on COVID-19 detectors relying on confounding factors, Geirhos et al. on shortcut learning, Roberts et al. on common pitfalls. I use the datasets and backbone, and I treat the warnings as design requirements.

**Q8. What is the research gap?**
The gap is not a missing architecture. DenseNet-121 is well known. It is in evaluation hygiene: some studies use random image-level splits, undocumented patient identity, no external test, and untested source shortcuts. Some projects also do not release a path from raw data to the numbers. I say "some" because I do not review all literature. I do not compare against their numbers.

**Q9. How do the shortcut papers relate to your work?**
They motivated my design: Zech et al. showed hospital-specific variation hurts new-hospital performance; DeGrave et al. reported that COVID-19 detectors relied on confounding factors rather than pathology (I know their abstract; I did not read all the details). Source identifiability itself is therefore not new. My addition is a measurement in my own leakage-controlled benchmark, band by band.

**Q10. Why DenseNet-121 and not a newer model such as a vision transformer?**
DenseNet-121 is a standard, well-documented backbone that trains on a CPU laptop. The architecture was fixed before V6 training and carried over from the V5 work; V6 changed the data, not the model. I did not compare architectures, and one backbone is a stated limitation. A different backbone could behave differently, and I did not test that.

## 3.3 Data and leakage

**Q11. Which datasets did you use and why?**
Six development sources: NIH ChestX-ray14, Kermany (paediatric), COVID-19 Radiography Database, TBX11K, Cohen COVID-19 collection and Shenzhen. Montgomery County is the external cohort only. All were downloaded from official locations with URL, size and SHA-256 logged in `data/raw/_provenance/`. The images are not in the repository because licences do not allow it.

**Q12. What is leakage and how did you control it?**
Leakage means information from test patients reaching training, for example the same patient in two splits or the same image under two names. I split by patient group, not by image. Groups are merged when they share an exact or near-duplicate image (perceptual hash within 6 bits and thumbnail correlation of at least 0.97). Whole groups are assigned 70/15/15. The build stops if any check is non-zero; all are zero.

**Q13. Are your splits really free of leakage?**
Free of leakage for rows with known patient identity. Identity is genuine for NIH and Cohen, and derived from file names for Kermany. For 4,130 TBX11K and COVID-19 Radiography rows identity is unknown, so each image is its own group, merged with exact and near-duplicates. A hidden second image of the same patient could therefore still fall in another split. I state this as a limitation, and the "zero" only proves separation for known identities.

**Q14. Why is the V5 pipeline not good enough?**
The audit found 154 of 1,570 test images (9.8%) shared a patient with training; 1,395 TBX11K "sick but not TB" images were labelled Pneumonia (49.7% of the class); 124 TB labels were latent TB; identities were synthetic for 8,478 rows; on Montgomery 0 of 80 normal scans were called Normal; and the checkpoint is lost, so V5 cannot be re-evaluated.

**Q15. How are labels defined?**
By written rules in a configuration file, fixed before training. Each image records its rule ID or its exclusion reason. For NIH only images with exactly one target finding are used; "No Finding" gives Normal, at most one image per patient. TB is labelled only from a TB source label (active TB in TBX11K, TB in Shenzhen). I excluded TBX11K "sick but not TB", latent TB, and an aggregated Kaggle TB archive with unclear provenance.

**Q16. NIH labels come from text mining. Is that a problem?**
Yes, it is a limitation. NIH Nodule/Mass and Effusion labels were mined from radiology reports by NLP, not read from the image. These are also the weakest classes (Nodule/Mass F1 60.1%). Effusion and Nodule/Mass exist only in NIH, so they are completely confounded with that source.

**Q17. Class and source are linked. Why did you not balance sources?**
The available official sources do not cover every class in every source. For example, Montgomery is not in training, Effusion and Nodule/Mass come only from NIH, and COVID-19 comes from two web-collected sets. I measured this confounding rather than hiding it (source probe, per-source table, within-source scores). Balancing sources would need more data; VinDr-CXR and JSRT were not available. I did not test a source-balanced design.

## 3.4 Preprocessing and ablation

**Q18. What preprocessing do you use and why?**
Gaussian blur (sigma 0.8), CLAHE on the lightness channel (clip 2.0, 8 x 8 tiles) to raise local contrast, a Gaussian low-pass filter (sigma 1.0), Lanczos resize to 224 x 224, and ImageNet normalisation. The low-pass choice (sigma 1.0) came from V5 work: ten candidate filters were compared on a validation set and sigma 1.0 had the best validation macro-F1. Training and serving use the same code, which avoids train-serve mismatch.

**Q19. Does preprocessing improve results? What did the ablation show?**
In one single-seed comparison (run 5, same seed, split and schedule, filters removed), the run with filters scored higher internally: accuracy 77.4% versus 74.1%, macro-F1 78.8% versus 75.8%, a gap larger than the seed-to-seed spread. External specificity was the same, 23.8%. I do not conclude that the filters improve generalisation. Run 5 also selected its last epoch, so a difference in convergence speed cannot be excluded.

**Q20. The no-filter run named 5 TB scans correctly on Montgomery. Is that important?**
It is an observation from a single run: 5 of 58 (8.6%), where the filtered runs gave 0 of 58 in every seed. It is a small count, so I do not call it an effect and I do not say the low-pass filter removes TB information. A repeat with more seeds would be needed.

**Q21. What is the limitation of your filtering order?**
Blur, CLAHE and low-pass run at native resolution before the resize, so their effective strength depends on the size of the input image. The same anatomy scanned at another resolution gets a different effective filter. Training and serving match each other, so the model is consistent, but a redesign that resizes first is future work. It also matters for the frequency analysis, because the cached images were resampled after filtering.

## 3.5 Model and training

**Q22. Describe your model.**
DenseNet-121 pretrained on ImageNet, without its top layer. On top: global average pooling, batch normalisation, a dense layer of 256 units with ReLU, dropout 0.3, and a dense layer of 6 units with softmax. The backbone runs in inference mode, so its batch-normalisation statistics stay frozen. The output is six non-negative numbers that sum to one.

**Q23. What is transfer learning and why use it?**
Transfer learning starts from weights learned on a large dataset (ImageNet) and adapts them to the new task. With 10,957 training images on a CPU laptop, starting from ImageNet features is cheaper and usually more stable than training from random weights. The cost: features stay close to ImageNet's, which I verified by weight comparison, because only the last 17 weighted backbone layers changed.

**Q24. Why two training phases?**
Phase 1 (2 epochs, backbone frozen, learning rate 1e-3) lets the random head adapt without distorting pretrained features. Phase 2 unfreezes the last 30 backbone layers (17 carry weights) with a small learning rate of 1e-5 to adjust them gently. The small rate protects the pretrained weights.

**Q25. What is the class-balanced loss?**
Standard cross-entropy with a weight for each sample equal to n divided by (k times n_c), where n is the number of training images, k = 6 classes and n_c the images of that class. Rare classes get larger weight, so the model does not ignore them. It is done without resampling.

**Q26. How did you decide when to stop training? Is early stopping not tuning on test?**
Early stopping uses validation macro-F1 only (patience 3 epochs). The checkpoint is the epoch with the best validation macro-F1, with validation AUC as tie-break. The trainer never reads the test split or the external cohort. Run 2 completed all 16 epochs and its best epoch was the last; its validation curve was still rising slightly, so I do not claim that longer training would or would not help.

**Q27. Why train on CPU, and what are the consequences?**
The laptop has 8 GB RAM and no GPU supported by the TensorFlow version. One epoch takes about 20 to 26 minutes, so schedules are short and time-boxed, and only three seeds and one ablation were possible. This limits variance estimates. It does not affect the correctness of the leakage controls.

**Q28. Which seed is the served model and why?**
Seed 42 (run 2). It was chosen over run 1 by a rule fixed before run 2's results were seen, and it was selected before seeds 43 and 44 existed. It is the highest of the three seeds, within a spread of under one percentage point. A small favourable draw cannot be ruled out. I never choose a seed using test or external results.

## 3.6 Evaluation

**Q29. Why do you report macro-F1 and not only accuracy?**
Accuracy depends on class sizes and the source mix. Macro-F1 is the unweighted mean of per-class F1, so a weak small class (for example TB with 145 test images) counts as much as a large class. I use macro-F1 for checkpoint selection and I report accuracy, macro-F1, macro AUC and per-class results.

**Q30. What do your two confidence intervals mean, Wilson and bootstrap?**
Both describe test sampling variation only. The Wilson interval is a formula for proportions that behaves well for small counts; for accuracy it is 75.7 to 79.0. The percentile bootstrap resamples test images 2,000 times; it gives 75.8 to 79.1. I always say which one I quote. Neither covers training variance; the three seeds give a rough view of that.

**Q31. What is calibration and what is ECE?**
Calibration asks whether a confidence of 80% is right about 80% of the time. Expected calibration error is the average gap between confidence and accuracy over 15 equal-width bins of the top-class score. My test ECE is 0.020, which is low. I report it because the app shows scores to users. No calibration method was fitted, and the scores are still not disease probabilities.

**Q32. How do you know you did not tune on the test set?**
Selection used validation data only. Test and Montgomery are each scored once per run, after the checkpoint is fixed (the `--final` flag). The selection rule between runs was recorded in the repository before run 2's results. For the extension, the thresholds and the order of access were fixed in advance, each test or Montgomery access is logged, and every number in the documents is generated from the result files.

**Q33. Why is Montgomery a good external test, and what are its limits?**
It is a separate hospital set (a TB screening programme) that never entered training; development images matching it by hash or near-duplicate were removed. Limits: only 138 images, only two classes (80 Normal, 58 TB), so it cannot test six-class accuracy and gives wide intervals. It was scored once per run, but run 2 was started after run 1's results were seen, so it is not completely untouched.

**Q34. Why is the external evaluation binary?**
Montgomery has only Normal and TB scans. I therefore report abnormal-versus-normal sensitivity and specificity, plus the exact TB recall (whether the model names Tuberculosis). Sensitivity without specificity is misleading, so they are always shown together.

**Q35. Why do you not compare with published papers?**
Test sets, splits and labels differ, so numbers are not comparable. Many published results use random image-level splits, which can leak patients. I compare only with my own earlier runs on the same data, and I compared runs on a common validation set that neither build used for training or testing.

## 3.7 Results interpretation

**Q36. Internal accuracy is 77.4% but external specificity is 23.8%. Why?**
I do not know the cause for certain. The model recognises TB inside the data (TB F1 85.7%, AUC 0.994) but on Montgomery most scans are assigned to Pneumonia and Nodule/Mass. The evidence is consistent with dependence on source-specific cues, and class-source confounding, but it does not prove it. Other explanations such as domain shift in acquisition or population are untested. The gap is the main finding: internal metrics overstate behaviour on an unseen hospital.

**Q37. Sensitivity is 94.8%. Is that not good?**
No. It means TB scans were called "abnormal", but the model also called 61 of 80 normal scans abnormal. Specificity is 19 of 80 (23.8%, CI 15.8 to 34.1). Exact TB recognition is 0 of 58. So the sensitivity is not TB detection. It shows the model calls most Montgomery scans abnormal.

**Q38. What does low specificity mean in practice?**
That the model would flag most healthy patients from this hospital as abnormal. In a screening use, this would create many false alarms and the output would carry little information. It is another reason I do not make clinical claims.

**Q39. Is TB 0/58 a sign that the model cannot detect TB?**
Not exactly. Inside the data TB is well recognised (recall 82.8%, F1 85.7%). On Montgomery it names no TB scan TB in any of the three seeds; most are called Pneumonia or Nodule/Mass. Plausible reasons (TB appearance specific to Shenzhen and TBX11K, hospital differences) are untested. I report it as an observed failure to transfer.

**Q40. Run 1 called 71 Montgomery scans COVID-19 and run 2 only 3. What does this show?**
The COVID-19 calls fell from 71 to 3 of 138. This is consistent with the added COVID-19 source removing a source cue, but build 2 changed several sources at once (TBX11K, COVID-19 Radiography, two more NIH archives), so it is not proven. Also the errors moved to other abnormal classes, and specificity did not improve: run 2 called 19 of 80 normals Normal, against 43 of 80 for run 1. Internal gains did not become external gains.

**Q41. Why did you select run 2 over run 1?**
By a rule fixed in advance, on 2,018 common validation images that neither build used for training or testing: macro-F1 79.6% versus 59.4%. But on the 1,392 images from the four sources both runs trained on, accuracy is almost equal (67.8% versus 67.6%) and macro-F1 is 68.5% versus 62.2%. So much of run 2's advantage is on sources that only run 2 trained on. Also run 2 was started after run 1's external results were seen.

**Q42. Seeds: is the served model typical?**
Across seeds 42, 43, 44 internal accuracy is 77.1% mean (SD 0.3). The served run is the highest, within under one percentage point, so with this recipe and split it is not an outlier; but three seeds cannot rule out a small favourable draw. External specificity varies more: 23.8, 31.2 and 26.2% (mean 27.1, SD 3.8), and exact TB is 0 in every seed. The seed result does not generalise to other splits, data or architectures.

## 3.8 Grad-CAM

**Q43. What is Grad-CAM and how does it work?**
Grad-CAM computes the gradient of the predicted class score with respect to the last convolutional feature maps, averages these gradients per map, weights the maps, applies ReLU and overlays the result as a heat map. It shows which image regions raised the score. I use the last convolutional block of the backbone.

**Q44. What did your Grad-CAM images show?**
For one correctly classified median-confidence test image per class (selection rule fixed before I viewed heatmaps): COVID-19 and Normal heatmaps concentrate on neck, shoulders and image borders; Nodule/Mass and TB heatmaps fall inside the lung fields; the Pneumonia (paediatric Kermany) heatmap is diffuse. One image per class is illustrative only. Together with the external result it is consistent with the source-shortcut concern.

**Q45. Can doctors trust Grad-CAM as an explanation?**
No. It shows where the network responded, not radiological reasoning, and it can highlight irrelevant regions. In my demo image, part of the attention lies outside the lung fields. I describe it as an illustrative tool and show it with a caption that says so.

## 3.9 Source shortcuts and novelty (C1 to C4)

**Q46. What is a source shortcut and how did you test for it?**
A shortcut is when the model uses a cue linked to the label in the data but not the disease, for example the style of one dataset. I tested availability with a linear probe on frozen features: it predicts the source of a test image with 93.8% balanced accuracy over 6 sources (chance 16.7%), and within one class too (Normal 97.2%, TB 98.9%). This shows source information is linearly available, not that the model uses it. So I call my evidence indirect.

**Q47. If a probe cannot prove use, what would prove it?**
An intervention: change or remove the source information and see whether the predictions change, ideally with retraining. My C3 (erasure) and C2 (band suppression) were attempts in this direction. C3 changes the features, not the trained model. C2 was not run. So I do not have a causal proof, and I do not claim one.

**Q48. What is C1 and what did you find?**
C1 is a per-band map. I split the cached 224 x 224 images into six octave frequency bands (A coarse to F fine), made band-pass and band-stop versions, extracted frozen DenseNet-121 features and trained linear probes for source and for disease (macro-F1 within each source, so source cues cannot buy the score). Source information is present in every band (64.4% in band A to 93.4% in band F against chance 16.7%), highest in the high bands. Removing any one band changes it by at most 2.9 points. No band stands out as the carrier. That null result is stated plainly.

**Q49. Why do you use within-source disease macro-F1?**
Because class and source are confounded, a six-class score can be bought with source cues. Computing disease macro-F1 inside each source and averaging removes that route. Only sources with at least two classes of 20 or more images count (five sources). The six-class macro-F1 is secondary.

**Q50. Why balanced accuracy for the source probe?**
Sources have very different sizes. Balanced accuracy is the mean of per-source recall, so a big source cannot hide errors on a small one. Chance is 1 divided by the number of sources, 16.7% for six.

**Q51. What is INLP and what did you find?**
Iterative nullspace projection: train a linear source classifier on the features, remove the directions it used, repeat. After 20 steps (the cap) the source accuracy a fresh linear probe can reach fell from 95.8% to 73.8%, not to chance. Disease scores fell too: within-source macro-F1 by 4.7 points and six-class macro-F1 by 12.4. This is consistent with source and disease information being entangled in these features. INLP is a published method; I applied it as a measurement, not as a new method.

**Q52. What is LEACE and how does it differ?**
LEACE is a one-step, closed-form linear erasure: after it, no linear classifier can read the source better than chance. In my features it brought source accuracy to 16.7% but lowered six-class macro-F1 by 25.2 points, about twice the INLP loss, and within-source macro-F1 by 8.8 points. So removing all linearly readable source information cost much of the disease signal. I do not present this as an achievement.

**Q53. What is leave-one-source-out (C4) and what did you find?**
Train a linear head on five sources and test on the sixth. With ImageNet features (the clean held-out test, because no project training touched them), held-out macro-F1 was far below the in-distribution reference for four of five sources: TBX11K 22.7% (gap 72.3 points), Shenzhen 33.1% (51.6), Kermany 50.2% (42.0), Cohen 12.0% (33.9, very few images). NIH shows a negative gap only because its reference is low (the six-class head also predicts NIH-only classes, counted as errors); it is not better generalisation. The binary abnormal-versus-normal AUCs for the two TB sources are only 0.653 (Shenzhen) and 0.605 (TBX11K). These are not external validation and every number carries the patient-identity caveat.

**Q54. Why is the trained-backbone matrix not "held-out"?**
Because the trained backbone saw all six sources during training, its features of a "held-out" source are not independent. It is a secondary matrix and I never call it held-out or external.

**Q55. What is C2 and why is it NO-GO?**
C2 would retrain the network on band-stopped images. A rule fixed in advance, using validation data only, required a band that lowers source accuracy by at least 5 points while the within-source disease macro-F1 falls by at most 2 points (upper confidence bound below 3 points; section 10 of `NOVELTY_PROTOCOL.md`), a matched-energy random control, and agreement from a cheap frozen-feature test (C2-lite). The best band (F) lowered source accuracy by 2.5 points on validation (2.7 on test), and the disease change was not distinguishable from zero but its upper bound exceeded the limit. No band satisfied criterion 1, so the decision is NO-GO and I did not train. No C2 number exists; it is future work and needs a new decision.

**Q56. Is a NO-GO a valid result? Is it not just a failure?**
It is a valid result of a designed experiment. The rule was fixed in advance, so a NO-GO shows the rule had force. It also tells us something: in frozen features, source information is spread over all bands, so dropping one band does not remove it. It does not say anything about training a network on band-stopped images, because that was not tested.

**Q57. What exactly is new and what is not new?**
Not new: source identifiability (Zech et al. and others), frequency-based shortcuts in general, INLP and LEACE (published), leave-one-domain-out on chest X-rays (Yao et al. 2019). Possibly new in its narrow form: a band-by-band measurement of source versus disease information in CNN features of chest radiographs. Even that is stated only as "to our knowledge, within the searches described", and the searches were not exhaustive (no Scopus, no Google Scholar check yet).

**Q58. What are the main caveats of the frequency analysis?**
Four. The cache is resampled to 224 x 224 after filtering, so band maps partly reflect resampling and my pre-filters (the no-filter cache differs in the high bands). The trained backbone saw all sources. Class and source are confounded, so within-source scoring is primary. Intervals cover test-sample variation only, samples are small, and everything is one run on frozen features. I therefore do not say "scanner fingerprint".

## 3.10 Software engineering

**Q59. How is the application built?**
A React frontend (Analyze, History, Metrics, Patients pages) calls a FastAPI backend under `/api/v1`. The backend validates the upload, runs the CXR gate, applies the same preprocessing as training, runs the DenseNet model, computes Grad-CAM, and stores results in SQLite. Endpoints include predict, health, model, model-metrics, predictions, reports and patients.

**Q60. What is the CXR input gate?**
A check that the uploaded image looks like a chest radiograph, so that a photo is rejected (HTTP 422 `INVALID_CXR`). A trained gate model is absent, so a heuristic is used. Its original threshold of 0.83 rejected 1,494 of 2,350 genuine build-2 test radiographs (63.6%). I recalibrated it on validation radiographs to 0.703, which gave 218 of 2,350 (9.3%) on the same test split and raised the validation pass rate from 38.7% to 91.0%. All synthetic non-radiographs are still rejected, but no real non-radiograph set exists, so it is a limited safeguard.

**Q61. What is the model integrity hash?**
The configuration holds the SHA-256 of the expected checkpoint. The backend loads a checkpoint only if its hash matches; otherwise `/predict` returns 503 `MODEL_NOT_LOADED` and nothing is written to the database. Stored predictions also keep the model version and the hash. This prevents silently serving the wrong weights.

**Q62. How is the software tested?**
A Python suite (pytest) covers configuration, preprocessing equivalence with a frozen copy of the original code, model construction and checkpoint refusal, metric definitions, evaluator validation, data splitting rules, consistency of documents with generated evidence, and every API behaviour. About 150 tests pass and 1 is skipped (it needs the checkpoint). The frontend has 19 Jest tests and a production build. Check `docs/TRACKER.md` and re-run before the viva.

**Q63. Describe your CI/CD.**
GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request. The Python job installs the backend requirements on Python 3.12 and runs pytest (checkpoint-dependent tests skip when no model is present). The frontend job installs with `npm ci` on Node 18, runs Jest and builds. It is continuous integration. There is no automatic deployment. Check on GitHub that the latest run is green.

**Q64. What is the Docker status?**
A Dockerfile exists (two stages: Node 18 builds the frontend; a Python 3.11 image serves the backend and static files). It has not been built or tested, because Docker is not installed on my development laptop. I documented what the file says, not what I observed. Deployment is described as a research prototype, not a clinical or multi-user system.

**Q65. How is the research pipeline made reproducible?**
One script per step: `acquire.py` (downloads with provenance), `build_v6_manifest.py` (rules, caps, split, leakage checks), `train_v6` and `evaluate_v6` (with `--final` for once-only), `served_metrics` and `render_docs` (generate the numbers the app and documents use). Each run records the code commit, split fingerprint, manifest hash, library versions, per-epoch history and checkpoint hash. Images are external, so a full rerun needs the downloads.

## 3.11 Ethics, licences, AI assistance

**Q66. What about ethics and patient privacy?**
All data are public and published as de-identified by their owners; I handled no identifiable patient data beyond what the datasets publish. The app carries a notice that it is not a medical device. A clinical study would need ethics approval and local data. For publication I still need to confirm the ethics-statement wording (exemption for public data) with my supervisor.

**Q67. What about the data licences?**
Images are not redistributed. The thesis lists the terms recorded for each source (for example Kermany is recorded as CC BY 4.0; Shenzhen and Montgomery terms are in the NLM read-me files and are marked unresolved). Several entries are marked "confirm on the publisher page" in the thesis table, so I should check them before any redistribution or publication that depends on them.

**Q68. Did you use AI tools? How do you declare it?**
The declaration in the thesis states that generative AI tools were used as assistance in drafting parts of the software and the report, that I reviewed the content and take responsibility, and that every reported number is read from files produced by the evaluation code. The wording must be confirmed against the university's policy (marked for me to personalise). Be ready to explain every part of the code and every number yourself.

## 3.12 Limitations and future work

**Q69. What are the main limitations?**
Three seeds only (SD is rough) and one ablation seed; CPU and time-boxed; classes assumed mutually exclusive; patient identity unknown for 4,130 rows; Effusion and Nodule/Mass from NIH only with NLP labels; a small binary external cohort (138); a heuristic CXR gate; Grad-CAM is not clinical reasoning; one backbone; no clinical validity.

**Q70. What would you do next?**
Add VinDr-CXR and JSRT (after access) to strengthen Effusion and Nodule/Mass; a multi-label formulation; more seeds and a larger multi-site external cohort; a CXR gate trained on real radiographs and non-radiographs; the C2 band-suppression training only after a new decision; and any clinical use only after prospective evaluation with clinicians, reported according to TRIPOD+AI and CLAIM.

## 3.13 Publication plan

**Q71. Where will you publish?**
Two outputs are planned and nothing has been submitted. A four-page conference paper for IEEE EMBC 2027 (submission deadline 24 January 2027, 11:59 PM AoE; check the portal) and a journal paper. The journal target is Informatics in Medicine Unlocked (Elsevier), with Biomedical Signal Processing and Control as fallback, chosen after the C2 NO-GO made the work a diagnostic study without a new technique. Author order: Sathwik, Ramana, Tirupathi, Srinivas. The corresponding author is still to be confirmed.

**Q72. How will the journal paper extend the conference paper?**
The conference paper reports the rebuild, the V5 audit in brief, the internal and external results of the selected run, and contains no probe, spectral map, INLP or leave-one-source-out. The journal paper cites it, states that it is substantially extended, lists only completed additions (three seeds, ablation, source probe, per-source analysis, run comparison, gate recalibration, and C1, C3, C4), redraws the figures and writes the text afresh. The overlap statement goes to the editors, and the two must not be under review at the same time with substantially the same content.

**Q73. Is the 30% new-content rule a requirement?**
That figure is common practice and is not stated in the files I read. I will ask the editor. This is a question I should check, not guess.

---

# Part 3b. Hard and trap questions

**H1. "Your model fails on a new hospital, so why should we accept the project?"**
Because the project's aim is an honest evaluation, and it found the failure. The earlier version hid the same failure behind a leaked test split (82.9% internal, 0 of 80 external normals correct). The new pipeline reveals it, quantifies it, studies likely causes with controls, and states what is not shown. A project that reports a failure it could have hidden is stronger scientifically than one that reports a high number from a leaky split.

**H2. "You say the shortcut is possible, so did you prove that the model uses it?"**
No. I proved that source information is available in the features (93.8% balanced accuracy against 16.7% chance), that class and source are confounded, and that Grad-CAM and the external behaviour are consistent with it. That is indirect evidence. A proof needs an intervention on the trained model, which I did not complete (C2 was NO-GO and not run). I do not use the word "proved".

**H3. "Was your selection rule pre-registered?"**
No. It was written in a repository file (`docs/data/run_selection_protocol.md`) before run 2's results existed, and the later analysis rules were fixed in advance and recorded in the repository. Nothing was registered with an outside body, so "pre-registered" would overstate it. Also, run 2 was started after run 1's external results were seen, and I state that.

**H4. "Montgomery was used for several runs. Is it really external?"**
It was never used for training or for choosing the served model. But it was scored once per run for runs 1 to 5, and for the extension twice more (C3 and C4), and run 2 was started after I saw run 1's Montgomery result. So it is external but not untouched. I list every access, and I do not use it to select the model.

**H5. "You report 77.4%. How can you be sure about patient separation, given 4,130 images without identity?"**
I cannot be sure for those rows. The check of zero patient groups spanning splits proves separation only for rows with known identity. For TBX11K and COVID-19 Radiography, each image is its own group, merged with exact and near-duplicates (hash and thumbnail correlation). Hidden duplicates of one patient would not be caught. This is in the limitations, and per-source accuracy differs a lot (COVID-19 Radiography 99.5%, NIH 62.5%), so I do not rank sources by data quality.

**H6. "Your seed SD is 0.3. Does that mean the model is stable?"**
It is a rough estimate from three runs. It covers the training seed only, with the split and recipe fixed. It does not cover other splits, data or architectures. The served run is the highest of the three, so a small favourable draw cannot be ruled out. External specificity varies more (SD 3.8) and the TB failure is the same in all seeds.

**H7. "You removed filters and accuracy fell, so the filters help, correct?"**
In this single-seed comparison, the run with the filters scored higher internally, by more than the seed spread. That is a description, not a causal claim. One seed cannot separate the effect from run-to-run variation, run 5 selected its last epoch so convergence speed may differ, and external specificity did not change. I do not say the filters improve generalisation.

**H8. "INLP removed the shortcut. Is that not a success?"**
No, and I do not present it as one. INLP lowered the source information a fresh probe could find from 95.8% to 73.8%, not to chance, and disease scores fell as well. LEACE removed linear source information completely but lowered six-class macro-F1 by 25.2 points. It also changed only frozen features, not the served model. The result says source and disease information are entangled in these features, which is a negative result.

**H9. "Why no clinical claim, when 94.8% sensitivity looks excellent?"**
Because sensitivity is only meaningful with specificity, and specificity is 23.8%. The model named no TB scan as TB (0 of 58). A tool that calls most scans abnormal has high sensitivity and little value. Also there is no clinician-reviewed ground truth and no prospective evaluation.

**H10. "A different examiner: did an AI write your thesis and code?"**
I used generative AI tools as assistance for drafting parts of the software and the report, and this is declared in the thesis. I reviewed the content and take responsibility. Every reported number is read from files produced by the evaluation code, and I can explain the pipeline, the split rule, the metrics and each limitation. Prepare by walking through one run from raw data to a number in the thesis without notes. (Confirm the exact declaration wording with the university policy.)

**H11. "The NIH negative gap in leave-one-source-out means NIH generalises better, no?"**
No. NIH has only two testable classes in this analysis, and its in-distribution reference macro-F1 is already low (25.3% with ImageNet features) because the six-class head also predicts NIH-only classes (Effusion, Nodule/Mass), which count as errors for the two evaluated classes. The negative gap is a metric artefact. The binary abnormal-versus-normal AUC for NIH is 0.743, a more neutral reading.

**H12. "Did adding COVID-19 Radiography remove the COVID-19 shortcut?"**
I cannot say. COVID-19 calls on Montgomery fell from 71 to 3 of 138, which is consistent with the added source removing a cue. But build 2 changed several sources at once (TBX11K, COVID-19 Radiography, two more NIH archives), there is no ablation that isolates one source, and the external errors moved to other abnormal classes. So it is an observation with a possible explanation, not a result.

---

# Part 4. Before the viva: checklist

## 4.1 Documents

- [ ] Fill every personal placeholder in the report: contacts (student, supervisors), declaration and AI-assistance statement (amend to the university policy), certificate and Head of Department lines (confirm with the department whether one or two signature lines are wanted), acknowledgement. Search the report for `[TO BE PROVIDED]`, `[TO BE CONFIRMED]`, `[STUDENT TO PERSONALISE]` and `[AUTHORS TO CONFIRM]`.
- [ ] Open the final report `.docx` in Word; update fields (table of contents, list of figures and tables, cross-references); check the page numbers. The final report is `docs/thesis/LungAI_MTech_Project_Report.docx` (rebuilt 2026-11-04; the temporary `_NEW` copy was deleted).
- [ ] Print at least 3 copies of the report (examiners and supervisor) plus one for yourself, with a printed copy of Part 1 of this guide and the 33 slides as handouts (6 per page is enough).
- [ ] Confirm the viva date, time, room and the presentation rules with the department (not in the repository: check).
- [ ] Read the thesis abstract and conclusion once more; they must match the numbers in Part 1.

## 4.2 Laptop and demo setup (commands from README.md)

Run these at least a day before and again one hour before. Use Python 3.11 or 3.12, Node 18 or later.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python scripts/install_checkpoint.py --from runs/<run-name>/model.h5
cd backend
python main.py
```

In a second terminal:

```
cd frontend
npm ci
npm start
```

- Backend: http://127.0.0.1:8000 (API docs at `/docs`). Frontend: http://localhost:3000.
- Check `/api/v1/health`: it must not say degraded. Without the checkpoint, `/predict` returns 503 `MODEL_NOT_LOADED`. The expected file is `densenet121_v6r2_seed42.h5` (SHA-256 starts `7992b8da10ff`).
- Test tool: `python -m pytest` and, in `frontend`, `CI=true npx react-scripts test --watchAll=false`. Do not run them at the viva.
- Put one test-split image on the desktop (a Shenzhen TB radiograph was used for slide 27). Keep a photo (non-radiograph) ready to show the gate.
- Close other programs, charge the laptop, take the charger, switch off notifications, and set the screen to stay awake. Check the projector connection and the adapter.
- Check on GitHub that the latest CI run is green.

## 4.3 Backup

- [ ] Slides: the `.pptx` on the laptop, a copy on a USB stick and a PDF copy. Slide notes are in `docs/presentation/slides.md`.
- [ ] Screenshots from `docs/figures/screenshots/` in one folder (8 files) for use if the demo fails.
- [ ] Backup material to open on request: slides 18 and 19 (ROC, per-class table), slide 21 (run selection), slide 23 (seeds and ablation), and the full figures in `docs/figures/`.
- [ ] A copy of this guide on the laptop and on paper.

## 4.4 Final preparation

- [ ] Rehearse the 20-minute talk twice aloud with a timer, once in front of a person.
- [ ] Practise the 30-second answer to "your accuracy is low" (section 2.5) and the six hard questions you find hardest (H1 to H12).
- [ ] Learn Part 1 numbers: 77.4%, 75.7 to 79.0, 78.8%, 0.955; 19/80, 55/58, 0/58; 93.8% against 16.7%; 82.9% and 154/1,570; seeds 77.1% (SD 0.3); NO-GO.
- [ ] Prepare a short list of things you do not know: be ready to say "I did not test that".
- [ ] Sleep, eat, arrive early.
