# Documentation catalogue

Every document in the repository, what it is for and its state on 2026-11-22. Status lives in [TRACKER.md](TRACKER.md) (single status page); handoff context in [../PROJECT_EXECUTION_STATUS.md](../PROJECT_EXECUTION_STATUS.md).

**Framing.** Since 2026-11-16 the thesis, the EMBC conference paper, the slides and the viva guide present LungAI as a system designed and developed to predict lung disease. The source-information analyses are supplementary there (thesis Appendix G, backup slides). The IMU journal paper keeps the source-shortcut framing.

State: **current** = up to date with the served model v6.1 and the system framing · **record** = a dated record kept as written (do not rewrite) · **history** = superseded, kept for reference only.

## 1. Deliverables

| Document | Purpose | State |
|---|---|---|
| [thesis/LungAI_MTech_Project_Report.docx](thesis/LungAI_MTech_Project_Report.docx) | M.Tech project report (9 chapters, appendices A-G), built from `thesis/content/` | current (rebuilt 2026-11-22) |
| [thesis/content/](thesis/content/) | Markdown source of the report, one file per chapter; `31_appendix_source_analyses.md` is Appendix G | current |
| [thesis/build_report.py](thesis/build_report.py) | Builds the .docx and checks tokens, references, figures and citations | current |
| [presentation/LungAI_MTech_Presentation.pptx](presentation/LungAI_MTech_Presentation.pptx) | Review presentation: 30 slides plus 3 backup slides | current (rebuilt 2026-11-22) |
| [presentation/slides.md](presentation/slides.md), [presentation/README.md](presentation/README.md) | Slide text and speaker notes; how to build | current |
| [viva/LungAI_Viva_Guide.md](viva/LungAI_Viva_Guide.md) and `.docx` | Viva preparation: key numbers, story, about 70 expected questions | current (rebuilt 2026-11-22) |
| [manuscripts/conference/](manuscripts/conference/) | IEEE EMBC 2027 paper "Deep Learning-Based Lung Disease Detection Using Chest X-Ray Images"; README and submission checklist | current; page count to confirm on Overleaf |
| [manuscripts/journal_imu/](manuscripts/journal_imu/) | Journal paper for Informatics in Medicine Unlocked: the extended source-shortcut study, with supplementary, highlights, declarations, cover letter, reviewer-risk and AI-check notes | current draft (source-shortcut framing, unchanged) |
| [manuscripts/journal_bspc/](manuscripts/journal_bspc/) | Fallback journal draft for Biomedical Signal Processing and Control | history unless IMU is dropped |
| [manuscripts/shared/](manuscripts/shared/) | Generated results macros and JSON (do not edit), bibliography, `RESULTS_FACTS.md` (allowed wording per result), `WRITING_BRIEF.md` (rules for every writer), `REFERENCES_STATUS.md`, `LITERATURE_NOTES.md` | current |
| [figures/](figures/README.md) | Generated figures (PNG and PDF) and app screenshots | current |

## 2. Status and process

| Document | Purpose | State |
|---|---|---|
| [TRACKER.md](TRACKER.md) | Single status page: engineering, deliverables, decisions waiting for the student, risks | current |
| [../PROJECT_EXECUTION_STATUS.md](../PROJECT_EXECUTION_STATUS.md) | Handoff for a new session, newest entry first | current (older entries are records) |
| [publication/SUBMISSION_TRACKER.md](publication/SUBMISSION_TRACKER.md) | Venue table, publication ethics, IMU checklist | current |
| [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) | Pre-commit audits of 3-4 Nov 2026 and their resolution | record |
| [REPO_ORGANISATION_PLAN.md](REPO_ORGANISATION_PLAN.md) | Repository reorganisation plan and what was applied (`f0ea013`) | record |

## 3. System documentation

| Document | Purpose | State |
|---|---|---|
| [../README.md](../README.md) | Project overview, quick start, results summary (generated block) | current |
| [architecture.md](architecture.md) | Components, data flow, code layout | current |
| [api.md](api.md) | REST endpoints and response format | current |
| [inference.md](inference.md) | Checkpoint loading rules, inference path, Grad-CAM, input gate | current |
| [frontend.md](frontend.md) | React pages and services | current |
| [deployment.md](deployment.md) | Local and Docker deployment (Docker untested on this laptop) | current |
| [troubleshooting.md](troubleshooting.md) | Common problems and fixes | current |
| [testing.md](testing.md) | Test commands, results and coverage; CI status | current |
| [../models/README.md](../models/README.md) | Where the checkpoint goes and how to install it | current |

## 4. Data, model and results

| Document | Purpose | State |
|---|---|---|
| [dataset.md](dataset.md) | Sources, label rules and composition | current |
| [data_splits.md](data_splits.md) | The V5 split and its leakage (tested against the V5 manifest) | record |
| [model_card.md](model_card.md) | Model card for v6.1 | current (results block generated) |
| [training.md](training.md), [evaluation.md](evaluation.md) | Training protocol and evaluation definitions | current |
| [results/current_model.md](results/current_model.md) | Full generated result tables of the served model | current (generated; do not edit) |
| [experiment_results.md](experiment_results.md) | Runs 1-5, seeds, ablation, V5 history | current (results block generated) |
| [limitations.md](limitations.md) | What the numbers do and do not show | current |
| [thesis_methodology.md](thesis_methodology.md) | Short methodology summary, what may and may not be claimed | current |
| [reproducibility.md](reproducibility.md) | How to reproduce each result | current |
| [data/](data/) | Source inventory, recovery plan, V6 rules, training plan, run selection protocol | record (rules fixed in advance) |

## 5. Research planning (supplementary source analysis)

| Document | Purpose | State |
|---|---|---|
| [publication/NOVELTY_PROTOCOL.md](publication/NOVELTY_PROTOCOL.md) | Protocol for C1-C4, approved at `f427c04`; code reads it | record (change only through its change log) |
| [publication/PROTOCOL_REVIEW_FOR_STUDENT.md](publication/PROTOCOL_REVIEW_FOR_STUDENT.md) | Plain-language review of the protocol | record |
| [publication/RESEARCH_BRIEF.md](publication/RESEARCH_BRIEF.md) | Teaching brief on the research question and venues | record |
| [publication/JOURNALS.md](publication/JOURNALS.md), [CONFERENCES.md](publication/CONFERENCES.md), [JOURNAL_SCOPE_MATCH.md](publication/JOURNAL_SCOPE_MATCH.md) | Venue research | record |
| [publication/REPORT_TEMPLATE_FINDINGS.md](publication/REPORT_TEMPLATE_FINDINGS.md) | Report format used for the thesis | record |
| [publication/lit_search/](publication/lit_search/) | Literature searches behind the novelty statement | record |

## 6. History

| Location | Contents |
|---|---|
| [publication/archive/](publication/archive/README.md) | Superseded planning notes |
| [archive/](archive/README.md) | V5-era thesis, paper and audit material with known false or outdated claims; do not cite. [archive/DUPLICATES.md](archive/DUPLICATES.md) maps removed duplicates |
| [../experiments/](../experiments/README.md) | V1-V5 experiment scripts and reports (historical archive) |
