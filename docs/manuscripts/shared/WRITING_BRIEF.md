# Writing brief (shared by every document agent)

Read this whole file before writing anything. It is the contract between the parallel writers.

## 1. Project facts

- Repository `D:\LungAI\Lung_Disease_Detector`, branch `research/lungai-clean-baseline`. **Never run git commands that change state** (no commit, checkout, reset, stash, push). Do not modify anything under `lungai/`, `backend/`, `frontend/`, `scripts/pipeline/`, `runs/`, `data/`, `artifacts/`, `models/` unless your task says so. A model is training on this 8 GB CPU laptop: keep CPU and RAM use low (no TensorFlow, no servers, no large loops over images unless your task says so).
- System: **LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier**. Classes in this fixed order: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis. FastAPI backend + React frontend, Grad-CAM, SQLite, heuristic CXR input gate.
- Sources of truth (read, do not contradict): `README.md`, `docs/model_card.md`, `docs/limitations.md`, `docs/thesis_methodology.md`, `docs/training.md`, `docs/evaluation.md`, `docs/architecture.md`, `docs/data/*.md`, `docs/results/current_model.md`, `lungai/configs/lungai_densenet121.json`, `lungai/configs/dataset_v6_proposal.json`, `PROJECT_EXECUTION_STATUS.md`, `docs/TRACKER.md`.
- History: an earlier V5 model (recorded 82.93% internal accuracy) was audited and found to have patient leakage, proxy labels and an external specificity of 0/80; its checkpoint is lost. V6 rebuilt the data from official sources with patient-grouped splits. Two V6 runs exist: run 1 (`v6_build_001`) and run 2 (`v6_build_002`, adds TBX11K and the COVID-19 Radiography Database), **the final model is chosen on a common validation set after run 2 finishes (~17:00 on 2026-11-03)**.

## 2. Numbers: never type them by hand

- LaTeX: `\input{../shared/results_macros.tex}` and use macros such as `\RTestAcc\%`, `\RTestMacroFOne\%`, `\RTestMacroAUC`, `\RExtSensK/\RExtSensN`, `\RExtSpecK/\RExtSpecN`, `\RExtTBK/\RExtTBN`, `\RNTrain`, per-class `\RFOneTB`, `\RAUCCovid`, etc. The full list is in `docs/manuscripts/shared/results_macros.tex`; JSON twin `results.json` (same keys without the `R` prefix) for python-docx / python-pptx scripts.
- These files are regenerated from the final model; today they hold run 1's values. Sentences may embed macros freely.
- **Interpretation that depends on results** (which classes are weak, whether external behaviour improved, the COVID-19 shortcut, comparisons between runs) must not be written yet. Put a clearly marked placeholder: LaTeX `\pending{what goes here}` (define `\newcommand{\pending}[1]{\textcolor{red}{[PENDING FINAL RESULTS: #1]}}`), Word/PPT text `[PENDING FINAL RESULTS: …]`.
- Dataset counts of the final build come from `data/derived/<build>/v6_build_report.json`; describe the rules, and use the macros `\RNTrain`, `\RNVal`, `\RNTest`, `\RNExternal` for sizes.

## 3. Integrity rules (non-negotiable)

- No invented results, comparisons, citations, novelty, approvals, signatures, emails or phone numbers.
- Single training run per model (seed 42), CPU, time-boxed: say so. No claim of clinical validity. Report external sensitivity only together with specificity. State patient identity is unknown for TBX11K and COVID-19 Radiography rows (grouped per image).
- Do not compare our numbers with other papers' numbers as if on the same test set.
- Contribution framing (defensible): an auditable, leakage-controlled rebuild of a multi-source six-class CXR benchmark from official sources; a reproducible end-to-end pipeline (acquisition → patient-grouped split → training → once-only test/external evaluation → served metrics); evidence that internal metrics overstate generalisation (external cohort, source shortcut analysis); an audit showing how an earlier pipeline's leakage inflated results.
- AI-assistance disclosure: include a statement where the venue requires it (Elsevier: "Declaration of generative AI and AI-assisted technologies in the writing process"); the authors must review it.

## 4. Author details (use exactly; never invent contacts)

- Student: **Katkam Sathwik**, Roll No. **24567T7604**, M.Tech, Department of Computer Science and Engineering, **KUCE&T (Kakatiya University College of Engineering and Technology), Kakatiya University**, Warangal, Telangana, India. Year: 2026.
- Supervisor: **Prof. Nagavelli Ramana**, Principal and Head of the Department (CSE & IT), KUCE&T, Kakatiya University.
- Supporting faculty: **Dr. P. Tirupathi**, Assistant Professor, Department of CSE, JNTUH University College of Engineering, Manthani. **Dr. K. Srinivas**, Assistant Professor, CSE, Swathi Reddy Institute of Technology for Women, Warangal.
- Contacts: always the literal placeholder `[TO BE PROVIDED]` (student email/mobile, supervisor email/mobile, faculty emails/mobiles).
- Manuscript author order (confirmed by the project owner): 1. Katkam Sathwik (first author), 2. Prof. Nagavelli Ramana (supervisor), 3. Dr. P. Tirupathi, 4. Dr. K. Srinivas. Corresponding author: `[TO BE CONFIRMED]`; all contact details `[TO BE PROVIDED]`. CRediT contributions must still be verified and all authors must approve the manuscript before submission: keep a visible note saying so (`[AUTHORS TO CONFIRM]` stays on CRediT roles and approval, not on the order).
- Certificates/declarations: text only, with blank signature/date/seal lines. Never imply anything has been signed or approved.

## 5. Citation keys (BibTeX file: `docs/manuscripts/shared/references.bib`)

Cite only these keys (the reference agent verifies each one and adds a `note` if a detail could not be verified; it may add more keys and lists them in `docs/manuscripts/shared/REFERENCES_STATUS.md`). Before finishing, writers must check that every key they used exists in `references.bib`; replace or remove any that the reference agent marked unverifiable.

| Key | Work |
|---|---|
| wang2017chestxray8 | Wang et al., ChestX-ray8 / NIH ChestX-ray14, CVPR 2017 |
| kermany2018identifying | Kermany et al., Cell 2018 (paediatric pneumonia CXR) |
| jaeger2014two | Jaeger et al., Two public chest X-ray datasets (Montgomery, Shenzhen), QIMS 2014 |
| candemir2014lung | Candemir et al., lung segmentation, IEEE TMI 2014 |
| liu2020rethinking | Liu et al., TBX11K, CVPR 2020 |
| cohen2020covid | Cohen et al., COVID-19 image data collection (arXiv 2020 / prospective predictions) |
| chowdhury2020can | Chowdhury et al., Can AI help in screening viral and COVID-19 pneumonia?, IEEE Access 2020 |
| rahman2021exploring | Rahman et al., Exploring the effect of image enhancement…, Comput. Biol. Med. 2021 |
| rahman2020reliable | Rahman et al., Reliable TB detection using chest X-ray…, IEEE Access 2020 |
| huang2017densely | Huang et al., DenseNet, CVPR 2017 |
| deng2009imagenet | Deng et al., ImageNet, CVPR 2009 |
| rajpurkar2017chexnet | Rajpurkar et al., CheXNet, arXiv 2017 |
| irvin2019chexpert | Irvin et al., CheXpert, AAAI 2019 |
| johnson2019mimic | Johnson et al., MIMIC-CXR, Sci. Data 2019 |
| nguyen2022vindr | Nguyen et al., VinDr-CXR, Sci. Data 2022 |
| shiraishi2000development | Shiraishi et al., JSRT database, AJR 2000 |
| zech2018variable | Zech et al., Variable generalization performance…, PLoS Med 2018 |
| degrave2021ai | DeGrave et al., AI for radiographic COVID-19 detection selects shortcuts…, Nat. Mach. Intell. 2021 |
| geirhos2020shortcut | Geirhos et al., Shortcut learning in deep neural networks, Nat. Mach. Intell. 2020 |
| roberts2021common | Roberts et al., Common pitfalls… COVID-19 ML, Nat. Mach. Intell. 2021 |
| wynants2020prediction | Wynants et al., Prediction models for COVID-19, BMJ 2020 |
| maguolo2020critic | Maguolo & Nanni, A critic evaluation of methods for COVID-19 automatic detection from X-ray images, Inf. Fusion 2021 |
| oakdenrayner2020exploring | Oakden-Rayner, Exploring large-scale public medical image datasets, Acad. Radiol. 2020 |
| seyyedkalantari2021underdiagnosis | Seyyed-Kalantari et al., Underdiagnosis bias…, Nat. Med. 2021 |
| selvaraju2017gradcam | Selvaraju et al., Grad-CAM, ICCV 2017 |
| pizer1987adaptive | Pizer et al., Adaptive histogram equalization and its variations, CVGIP 1987 |
| zuiderveld1994contrast | Zuiderveld, Contrast limited adaptive histogram equalization, Graphics Gems IV 1994 |
| guo2017calibration | Guo et al., On calibration of modern neural networks, ICML 2017 |
| naeini2015obtaining | Naeini et al., Obtaining well calibrated probabilities using Bayesian binning, AAAI 2015 |
| wilson1927probable | Wilson, Probable inference…, JASA 1927 |
| efron1994introduction | Efron & Tibshirani, An Introduction to the Bootstrap, 1994 |
| kingma2015adam | Kingma & Ba, Adam, ICLR 2015 |
| mongan2020claim | Mongan et al., CLAIM checklist, Radiol. Artif. Intell. 2020 |
| collins2024tripodai | Collins et al., TRIPOD+AI, BMJ 2024 |
| wang2020covidnet | Wang, Lin & Wong, COVID-Net, Sci. Rep. 2020 |
| apostolopoulos2020covid | Apostolopoulos & Mpesiana, Covid-19: automatic detection…, Phys. Eng. Sci. Med. 2020 |

## 6. Figures (produced by `lungai/evaluation/make_figures.py` into `docs/figures/`)

Refer to these exact file names (PNG at 300 dpi plus PDF for LaTeX). Caption text is yours; keep captions result-neutral or macro-based.

| File stem | Content | Result-dependent |
|---|---|---|
| fig_pipeline | End-to-end pipeline: acquisition → build (labels, caps, grouped split, leakage checks) → training → evaluation → served metrics → app | No |
| fig_system_architecture | Frontend, FastAPI backend, CXR gate, inference engine, Grad-CAM, SQLite, model store, metrics file | No |
| fig_dataflow | Data-flow of one prediction request (upload → validation → gate → preprocessing → model → Grad-CAM → DB → UI) | No |
| fig_model_architecture | DenseNet-121 backbone + GAP + BN + Dense(256) + Dropout(0.3) + Dense(6, softmax); two-phase training | No |
| fig_preprocessing | One radiograph through blur → CLAHE → low-pass → resize | No |
| fig_dataset_composition | Images per class and source in train/val/test | Yes (build) |
| fig_training_curves | Train loss, val accuracy/macro-F1/ROC-AUC per epoch | Yes |
| fig_confusion_matrix | Test confusion matrix (counts and row-normalised) | Yes |
| fig_roc_curves | One-vs-rest ROC per class on test | Yes |
| fig_per_class_metrics | Precision/recall/F1 per class with recall CIs | Yes |
| fig_per_source | Test accuracy by source | Yes |
| fig_external_breakdown | Montgomery: predicted class for normal and TB scans | Yes |
| fig_calibration | Reliability diagram (test) | Yes |
| fig_gradcam_examples | Grad-CAM on a few test images (generated last, needs the model) | Yes |

## 7. Output folders

| Deliverable | Folder |
|---|---|
| M.Tech report (generator + .docx) | `docs/thesis/` |
| Presentation (generator + .pptx) | `docs/presentation/` |
| Conference manuscript (IEEE LaTeX) | `docs/manuscripts/conference/` |
| Journal manuscript (Elsevier elsarticle, BSPC) | `docs/manuscripts/journal_bspc/` |
| Shared macros, references, this brief | `docs/manuscripts/shared/` |
| Figures | `docs/figures/` |
| Venue comparison, submission tracker | `docs/publication/` |
| Central tracker | `docs/TRACKER.md` (owned by the main session; propose changes in your final report instead of editing it) |

No LaTeX compiler is installed here: write clean LaTeX that compiles on Overleaf, using only standard packages and the official class files (`IEEEtran`, `elsarticle`), and list the packages used.

## 8. Style

Plain, precise scholarly English; short sentences; active voice where natural; no hype ("revolutionary", "state-of-the-art" without evidence), no filler. Define every abbreviation once. Every figure and table is numbered, captioned and discussed in the text. Captions: tables above, figures below.
