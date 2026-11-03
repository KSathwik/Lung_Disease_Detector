# LungAI M.Tech presentation: slide text and speaker notes
# Format: slides separated by a line "---". Header lines "key: value" (layout, title, subtitle, figure, caption, colw),
# then body: "- " bullet, "| " table row (first row = header, cells split by |),
# "> " speaker-note line. {{Key}} is replaced from docs/manuscripts/shared/results.json at build time.
# Anything written [PENDING ...] is drawn in red: only seed-run and ablation placeholders remain.

layout: title
title: LungAI DenseNet-121: Six-Class Chest X-Ray Classifier
subtitle: An auditable, leakage-controlled rebuild with patient-grouped evaluation
> Good morning. I am Katkam Sathwik, and this is my M.Tech project review under the guidance of Prof. Nagavelli Ramana.
> The project is LungAI, a six-class chest X-ray classifier with a web application.
> Its main point is not a high accuracy number but an evaluation that can be trusted: data rebuilt from official sources, patient-grouped splits and a once-only test.
> I will take about twenty minutes and then welcome questions.
---
layout: bullets
title: Outline
- Motivation, problem statement and objectives
- Literature review and research gap
- Audit of the earlier V5 model
- Methodology: data, preprocessing, model, training, evaluation
- System implementation and the CXR input gate
- Results, demo, limitations, conclusion and future scope
> This is the plan of the talk. First I explain why the problem matters and what was wrong with the earlier version of the system.
> Then I describe the V6 method in the order in which it was executed: data, preprocessing, model, training and evaluation.
> After the application, I present results, limitations and the way forward.
---
layout: bullets
title: Motivation
- Chest X-ray (CXR) is the most widely available imaging test for lung disease
- Radiologist access is limited, especially outside large centres
- Public CXR datasets make deep-learning decision-support research feasible [wang2017chestxray8, rajpurkar2017chexnet]
- Reported accuracies often fail to carry over to new hospitals [zech2018variable]
- A decision-support tool is only useful if its evaluation is honest
> Chest radiography is cheap and common, but reading it needs trained radiologists who are not available everywhere.
> Large public datasets have made research on automatic CXR classification possible.
> However, studies show that models often lose performance on data from another hospital.
> So the motivation is twofold: build a working tool, and make the evaluation trustworthy enough that its numbers mean something.
---
layout: bullets
title: Problem statement and objectives
- Classify a frontal CXR into six exclusive classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis
- O1-O2: rebuild the data from official sources with rule-based labels; patient-grouped splits; leakage checks pass (identity unknown for two sources)
- O3-O4: train DenseNet-121 reproducibly; evaluate once on test and on an external cohort, with confidence intervals
- O5: end-to-end application (FastAPI, React, Grad-CAM, SQLite)
- O6: report limitations without overclaiming
> The problem is single-label, six-class classification of chest X-rays. The classes are treated as mutually exclusive, which is itself a limitation I return to.
> The harder part is methodological: traceable labels, no patient crossing splits, and an external cohort.
> O1 and O2 are about data integrity, O3 and O4 about training and evaluation, O5 the software, O6 honest reporting. The model is a research prototype, not a clinical device.
---
layout: table
title: Literature review in brief
colw: 2.4, 5.2
| Theme | Representative works (citation keys) |
| Large CXR datasets | wang2017chestxray8, irvin2019chexpert, johnson2019mimic, nguyen2022vindr |
| TB and pneumonia sets | jaeger2014two, liu2020rethinking, kermany2018identifying |
| COVID-19 CXR classifiers | wang2020covidnet, apostolopoulos2020covid, chowdhury2020can |
| Backbones and tools | huang2017densely, deng2009imagenet, selvaraju2017gradcam |
| Pitfalls and shortcuts | zech2018variable, degrave2021ai, geirhos2020shortcut, roberts2021common, maguolo2020critic |
| Dataset quality, bias | oakdenrayner2020exploring, seyyedkalantari2021underdiagnosis, wynants2020prediction |
> The literature falls into six groups. Large public datasets enabled the field, and DenseNet with ImageNet pretraining is a common backbone.
> A second body of work warns that results on COVID-19 and other CXR tasks are often inflated by shortcuts, duplicate patients or source differences.
> We build on the datasets and the backbone, and we take the warnings as design requirements.
> Full citations are in the references slide and the report.
---
layout: bullets
title: Research gap
- Some CXR studies report results on a random image-level split
- Patient identity and label provenance are often undocumented
- External evaluation is often missing, and specificity is not always reported with sensitivity [roberts2021common, zech2018variable]
- Source and acquisition shortcuts are not always tested [degrave2021ai, geirhos2020shortcut]
- Some projects do not release an auditable pipeline from raw data to served metrics
> The gap is not a missing architecture. DenseNet-121 is well known.
> The gap is in evaluation hygiene: patient-grouped splits, documented label rules, external testing and a reproducible path from raw data to the numbers shown in the application.
> Our contribution is therefore a leakage-controlled, auditable rebuild rather than a new network.
> We make no claim of novelty beyond that.
---
layout: bullets
title: Why rebuild? Audit of the V5 model
- V5 recorded {{VFiveAcc}}% internal accuracy (N = {{VFiveN}} test images)
- Patient leakage: 154 test images (9.8%) shared a patient with training
- Proxy labels: half of "Pneumonia" was TBX11K "sick but not TB"; 124 TB labels were latent TB
- External check on Montgomery: {{VFiveExtSpecK}} of {{VFiveExtSpecN}} normal scans called Normal
- V5 checkpoint is lost, so it cannot be re-evaluated; V6 starts from the raw sources
> Before this project's V6, an earlier model reported nearly 83 percent internal accuracy.
> My audit found that 154 of its 1,570 test images came from patients also present in training, that several labels were proxies rather than true diagnoses, and that on an external cohort it called none of 80 normal scans normal.
> The V5 checkpoint was lost, so I could not re-evaluate it. These findings motivated rebuilding everything from official sources with patient-grouped splits.
---
layout: fig
title: Methodology: end-to-end pipeline
figure: fig_pipeline
caption: Acquisition, build (labels, caps, grouped split, leakage checks), training, once-only evaluation, served metrics.
> This is the whole pipeline. Raw data is acquired from official sources with recorded hashes.
> The build step applies label rules, per-class caps, a patient-grouped split and leakage checks.
> Training sees only the train and validation splits. Test and external data are evaluated once.
> The final metrics file is what the web application serves, so the numbers in the app are the numbers in the report.
---
layout: figs2
title: System architecture and request data flow
figure: fig_system_architecture, fig_dataflow
caption: Left: frontend, backend, CXR gate, inference, Grad-CAM, SQLite. Right: one prediction request from upload to interface.
> The application has a React frontend and a FastAPI backend. The backend validates the upload, runs the CXR gate, preprocesses the image and calls the inference engine.
> Grad-CAM is computed from the last convolutional block and results are stored in SQLite. The checkpoint is verified by SHA-256 before loading.
> The right diagram follows one request: upload, validation, gate, preprocessing, model, Grad-CAM, database, interface.
> Using identical preprocessing code for training and serving avoids train-serve mismatch.
---
layout: table
title: Datasets and sources
colw: 3.4, 3.0, 3.0
| Source | Used for | Notes |
| NIH ChestX-ray14 [wang2017chestxray8] | Normal, Effusion, Nodule/Mass, Pneumonia | One finding only; one image per patient |
| Kermany paediatric CXR [kermany2018identifying] | Normal, Pneumonia | Mendeley v2 |
| Shenzhen (NLM) [jaeger2014two] | Normal, Tuberculosis | Source label only |
| TBX11K [liu2020rethinking] | Normal, active TB | Run 2; patient identity unknown |
| COVID-19 data [cohen2020covid, chowdhury2020can] | COVID-19 | Radiography Database added in run 2 |
| Montgomery [jaeger2014two] | External cohort only | {{NExternal}} images; never in training |
> The training data comes from public, official sources only.
> NIH ChestX-ray14 supplies effusion, nodule or mass, pneumonia and normal images; Shenzhen and TBX11K supply tuberculosis.
> VinDr-CXR and JSRT could not be obtained in time, so effusion and nodules come from NIH alone.
> Montgomery is held out as an external cohort and never enters development.
> TBX11K and the COVID-19 Radiography Database have no patient identity, so each image is its own group, which I state as a limitation.
---
layout: bullets
title: Cleaning, label rules, grouped split, leakage checks
- Every image records its rule ID, or the reason it was excluded; TB label only from a TB source label
- Excluded: multi-finding NIH images, TBX11K "sick but non-TB" and latent TB, Rahman TB archive
- Split by patient group: train {{NTrain}}, validation {{NVal}}, test {{NTest}} (final build)
- Checks: disjoint groups, MD5/SHA-256 duplicates, perceptual-hash near-duplicates; build stops on a violation
- Residual risk: unknown identity (grouped per image) for TBX11K and COVID-19 Radiography rows
> Labelling rules were fixed before any training, in a configuration file. Each image keeps its source label, the rule that produced the class, and a hash of the raw file.
> I excluded images whose label would be a proxy, such as sick-but-not-TB, and an aggregated archive whose provenance was unclear.
> Splitting is on patient groups, and the builder checks leakage automatically: disjoint groups, exact hashes and perceptual hashes. The split fingerprint and manifest hash are saved with each run.
> For sources without patient identity the group is the image itself, which cannot detect hidden duplicates of one patient. I state that residual risk openly.
---
layout: fig
title: Preprocessing
figure: fig_preprocessing
caption: One radiograph through blur, CLAHE, low-pass filter and resize to 224 x 224.
> Preprocessing has four steps: a light Gaussian blur, contrast-limited adaptive histogram equalisation on the luminance channel, a Gaussian low-pass filter, and resize to 224 by 224.
> CLAHE improves local contrast [zuiderveld1994contrast, pizer1987adaptive].
> The filters run at native resolution before resizing, which makes the effective filter resolution dependent. This is a documented limitation.
> Training and serving use the same code.
---
layout: fig
title: Model architecture
figure: fig_model_architecture
caption: DenseNet-121, global average pooling, BatchNorm, Dense(256), Dropout(0.3), Dense(6, softmax).
> The model is DenseNet-121 pretrained on ImageNet [huang2017densely, deng2009imagenet].
> On top we add global average pooling, batch normalisation, a 256-unit dense layer with dropout 0.3, and a six-way softmax.
> Training is in two phases: first only the head, then the last 30 backbone layers.
> Grad-CAM uses the last convolutional activations of the backbone [selvaraju2017gradcam].
---
layout: bullets
title: Training protocol
- Phase 1: 2 epochs, backbone frozen, Adam, learning rate 1e-3 [kingma2015adam]
- Phase 2: last 30 layers trainable, learning rate 1e-5, early stopping (patience 3)
- Batch 32, class-balanced sample weights, flip and 10 degree rotation
- Checkpoint selected on validation macro-F1 only (tie: validation AUC)
- Single run, seed 42, CPU only, time-boxed
- Final model (run 2): {{EpochsTrained}} epochs trained, best validation macro-F1 at epoch {{SelectedEpoch}} (the last)
> Training has two phases, a frozen backbone followed by gentle fine-tuning.
> The checkpoint is chosen by validation macro-F1, never by test results.
> Because this was done on a laptop CPU with a time limit, each model is a single run with seed 42. I therefore cannot report run-to-run variance.
> Run 2, the final model, completed its full schedule; the best epoch was the last one and validation was still rising slightly. I do not claim longer training would help.
---
layout: bullets
title: Evaluation protocol
- Test split and external cohort are each evaluated once, after model selection
- Accuracy; per-class precision, recall, specificity, F1; macro ROC-AUC, PR-AUC
- 95% Wilson confidence intervals for proportions [wilson1927probable]
- Calibration: ECE and reliability diagram [guo2017calibration]
- External cohort is binary (normal vs TB); sensitivity always with specificity
- No comparison with other papers' numbers on different test sets
> The evaluation protocol is deliberately strict. Test and external data are used exactly once.
> Proportions carry Wilson intervals, and calibration is reported because probabilities are shown to users.
> Montgomery has only normal and TB scans, so external evaluation is binary.
> Sensitivity without specificity is meaningless for a screening tool, so they are always shown together.
> I do not compare my numbers with published numbers, since test sets differ.
---
layout: bullets
title: Implementation and the CXR gate
- Backend: FastAPI; checkpoint SHA-256 check; predict, health, metrics endpoints
- Frontend: React upload page, probabilities, Grad-CAM overlay, history; SQLite storage
- Gate: heuristic fallback (trained gate model absent); non-radiographs get HTTP 422
- Threshold 0.83 rejected {{CxrGateBTwoOldTestRej}}/{{CxrGateBTwoOldTestN}} ({{CxrGateBTwoOldTestRejPct}}%) of build-2 test radiographs; recalibrated to {{CxrGateNewThr}} on validation
- Same split after: {{CxrGateBTwoNewTestRej}}/{{CxrGateBTwoNewTestN}} ({{CxrGateBTwoNewTestRejPct}}%); Montgomery {{CxrGateBTwoOldExtRej}} to {{CxrGateBTwoNewExtRej}} of {{CxrGateBTwoNewExtN}}
- All synthetic non-radiographs are still rejected; no real non-radiograph set exists
> The backend is FastAPI and the frontend is React. The backend refuses a checkpoint whose hash does not match the recorded one.
> The input gate is a heuristic, because a trained gate model is not available. The problem was first noticed on the build-1 test split, where the original threshold rejected {{CxrGateBOneOldTestRej}} of {{CxrGateBOneOldTestN}} genuine radiographs; on the build-2 test split it rejects {{CxrGateBTwoOldTestRej}} of {{CxrGateBTwoOldTestN}}, so the application would refuse most real X-rays.
> I recalibrated the threshold from 0.83 to {{CxrGateNewThr}} on validation radiographs. On the same build-2 test split, rejection fell from {{CxrGateBTwoOldTestRejPct}} to {{CxrGateBTwoNewTestRejPct}} percent, and on Montgomery (the same set) from {{CxrGateBTwoOldExtRejPct}} to {{CxrGateBTwoNewExtRejPct}} percent; the validation pass rate rose from {{CxrGateValPassOld}} to {{CxrGateValPassNew}} percent.
> All synthetic non-radiographs are still rejected, but there is no real set of non-radiographs, so the gate remains a limited safeguard, and the model metrics are computed without it.
---
layout: table_fig
title: Results: internal test (final model v6.1, run 2)
figure: fig_confusion_matrix
colw: 3.0, 4.2
caption: Test confusion matrix (evaluated once).
| Metric | Value |
| Test images | {{NTest}} |
| Accuracy | {{TestAcc}}% (95% Wilson CI {{TestAccLo}}-{{TestAccHi}}) |
| Macro-F1 | {{TestMacroFOne}}% |
| Macro recall | {{TestMacroRecall}}% |
| Macro ROC-AUC | {{TestMacroAUC}} |
| Macro PR-AUC | {{TestMacroPRAUC}} |
- Errors concentrate in Nodule / Mass and Normal, the weakest classes
> These are the internal test results of the final model, v6.1, which is run 2 with seed 42. The test split was evaluated once.
> The 95% Wilson interval is shown for accuracy only (the percentile bootstrap gives {{TestAccBootLo}} to {{TestAccBootHi}}); no interval is shown for macro-F1. Macro ROC-AUC is higher than accuracy because AUC measures ranking, not the decision threshold.
> The confusion matrix shows that errors concentrate on Pulmonary Nodule / Mass and Normal. Nodule/Mass and Pleural Effusion are NIH-only classes with NLP-mined labels.
> These are internal numbers; the external cohort later shows they overstate behaviour on an unseen hospital.
---
layout: figs2
title: ROC curves and calibration
figure: fig_roc_curves, fig_calibration
caption: One-vs-rest ROC curves (left); reliability diagram on the test split (right), ECE {{TestECE}}.
- AUC highest: COVID-19 {{AUCCovid}}, TB {{AUCTB}}; lowest: Normal {{AUCNormal}}, Nodule {{AUCNodule}}
> The left panel shows one-versus-rest ROC curves on the test split. By ROC-AUC, COVID-19 ({{AUCCovid}}) and Tuberculosis ({{AUCTB}}) are highest, then Pleural Effusion ({{AUCEffusion}}) and Pneumonia ({{AUCPneumonia}}); Normal ({{AUCNormal}}) and Pulmonary Nodule / Mass ({{AUCNodule}}) are lowest.
> The right panel is the reliability diagram: predicted confidence against observed accuracy, with expected calibration error {{TestECE}} in the caption. Calibration matters because the application shows probabilities to users.
> Per-class precision, recall and F1 follow in the next table.
---
layout: table
title: Per-class results (test, final model v6.1)
colw: 3.0, 1.2, 1.7, 1.5, 1.4, 1.4
| Class | n | Precision % | Recall % | F1 % | AUC |
| COVID-19 | {{SupportCovid}} | {{PrecisionCovid}} | {{RecallCovid}} | {{FOneCovid}} | {{AUCCovid}} |
| Normal | {{SupportNormal}} | {{PrecisionNormal}} | {{RecallNormal}} | {{FOneNormal}} | {{AUCNormal}} |
| Pleural Effusion | {{SupportEffusion}} | {{PrecisionEffusion}} | {{RecallEffusion}} | {{FOneEffusion}} | {{AUCEffusion}} |
| Pneumonia | {{SupportPneumonia}} | {{PrecisionPneumonia}} | {{RecallPneumonia}} | {{FOnePneumonia}} | {{AUCPneumonia}} |
| Nodule / Mass | {{SupportNodule}} | {{PrecisionNodule}} | {{RecallNodule}} | {{FOneNodule}} | {{AUCNodule}} |
| Tuberculosis | {{SupportTB}} | {{PrecisionTB}} | {{RecallTB}} | {{FOneTB}} | {{AUCTB}} |
- Best F1: COVID-19, Tuberculosis, Pneumonia; weakest: Nodule / Mass ({{FOneNodule}}%), Normal ({{FOneNormal}}%)
- Source is confounded with class (next slides): high scores are not proof of reliability
> This table gives precision, recall, F1 and AUC per class. Support differs a lot between classes, so some estimates are less certain than others.
> The strongest classes by F1 are COVID-19, Tuberculosis and Pneumonia; the weakest are Pulmonary Nodule / Mass and then Normal.
> I do not rank sources by data quality. High class scores do not by themselves show clinical reliability, because source differences can act as shortcuts, as the next slides examine.
---
layout: table
title: External evaluation (Montgomery, final model, binary)
colw: 4.6, 3.0, 2.4
| Measure | Value | 95% Wilson CI |
| Sensitivity (TB scans called abnormal) | {{ExtSensK}}/{{ExtSensN}} = {{ExtSens}}% | {{ExtSensLo}}-{{ExtSensHi}}% |
| Specificity (normal scans called normal) | {{ExtSpecK}}/{{ExtSpecN}} = {{ExtSpec}}% | {{ExtSpecLo}}-{{ExtSpecHi}}% |
| Binary AUC | {{ExtAUC}} | not computed |
| Exact TB class on TB scans | {{ExtTBK}}/{{ExtTBN}} = {{ExtTB}}% | not computed |
- Run 1: {{RunOneExtCovidCalls}} of {{NExternal}} scans called COVID-19, {{RunOneExtSpecK}}/{{ExtSpecN}} normals Normal. Run 2: {{ExtCovidCalls}} COVID-19, {{ExtSpecK}}/{{ExtSpecN}} Normal
- Run 1's COVID-19 calls largely disappeared with build 2 (which adds several sources), but errors moved to other abnormal classes: consistent with, not proof of, a source shortcut; V5 specificity was {{VFiveExtSpecK}}/{{VFiveExtSpecN}}
> The external cohort is Montgomery, with 58 TB and 80 normal scans, evaluated as a binary task. Each model was evaluated on it once.
> Sensitivity is only meaningful with specificity. Specificity is low: the model would flag most healthy patients from this hospital as abnormal. Exact TB recognition is {{ExtTBK}} of {{ExtTBN}}.
> Run 1 called {{RunOneExtCovidCalls}} of {{NExternal}} scans COVID-19 and {{RunOneExtSpecK}} of {{ExtSpecN}} normals Normal (run-1 external evaluation). Run 2 calls {{ExtCovidCalls}} COVID-19 and {{ExtSpecK}} of {{ExtSpecN}} Normal.
> Interpretation: run 1's COVID-19 calls fell sharply when build 2 was used, but the external errors shifted to other abnormal classes, mostly Pneumonia and Nodule/Mass. Build 2 changed several sources at once (TBX11K, COVID-19 Radiography, two more NIH archives), so this is consistent with, but does not prove, a source shortcut; the evidence is indirect. Internal gains did not translate into better external specificity or any TB recognition.
---
layout: table
title: Model selection: run 1 vs run 2 on a common validation set
colw: 5.4, 2.4, 2.4
| Measure | Run 1 | Run 2 |
| Common validation images | {{CmpN}} | {{CmpN}} |
| Macro-F1 | {{CmpRunOneMacroFOne}}% | {{CmpRunTwoMacroFOne}}% |
| Accuracy | {{CmpRunOneAcc}}% | {{CmpRunTwoAcc}}% |
| Macro ROC-AUC | {{CmpRunOneMacroAUC}} | {{CmpRunTwoMacroAUC}} |
| Shared sources only (n = {{CmpRunOneSharedN}}): accuracy | {{CmpRunOneSharedAcc}}% | {{CmpRunTwoSharedAcc}}% |
| Shared sources only: macro-F1 | {{CmpRunOneSharedMacroFOne}}% | {{CmpRunTwoSharedMacroFOne}}% |
- Caveat: much of run 2's edge is on TBX11K and COVID-19 Radiography images, in-distribution for run 2 only
- McNemar discordant pairs {{CmpMcNemarB}} vs {{CmpMcNemarC}}, exact p = {{CmpMcNemarP}} (descriptive)
> Run 2 is the final model, v6.1, chosen by a rule fixed before its results were seen. The comparison uses {{CmpN}} validation images that neither build used for training or testing.
> Run 2 is ahead on macro-F1, accuracy and AUC. The McNemar discordant pairs and exact p-value are descriptive.
> The caveat matters: on the {{CmpRunOneSharedN}} images from the four sources both runs trained on, accuracy is almost equal and the macro-F1 gap is smaller. Much of run 2's advantage is on sources in-distribution for run 2 only.
> Also, run 2 was started after run 1's external results were known; the timeline is in the selection protocol document.
---
layout: fig
title: Source confounding: Grad-CAM examples
figure: fig_gradcam_examples
caption: Grad-CAM, one correctly classified median-confidence test image per class (rule fixed before viewing).
- Source probe: balanced accuracy {{ProbeImgBalAcc}}% (ImageNet features), {{ProbeMdlBalAcc}}% (model backbone); chance {{ProbeImgChance}}%
> A linear probe on frozen features predicts a test image's source with balanced accuracy {{ProbeImgBalAcc}} percent using ImageNet features and {{ProbeMdlBalAcc}} percent using the trained backbone; chance is {{ProbeImgChance}} percent over {{ProbeImgNSources}} sources.
> Within one class the source is still identifiable: Normal {{ProbeImgWithinNormal}} percent (chance {{ProbeImgWithinNormalChance}}), Tuberculosis {{ProbeImgWithinTB}}, Pneumonia {{ProbeImgWithinPneumonia}}, COVID-19 {{ProbeImgWithinCovid}}. Fine-tuning unfroze the last 30 backbone layers, 17 of which carry weights, so only those changed.
> So source and class are confounded and source information is linearly available, which makes a shortcut possible. This is indirect evidence, not proof that the model uses source cues.
> Grad-CAM, one image per class, illustrative only: COVID-19 and Normal heatmaps concentrate on neck, shoulders and image borders; Nodule/Mass and Tuberculosis fall inside the lung fields; Pneumonia (a paediatric image) is diffuse. Together with the external result this supports the concern.
---
layout: bullets
title: Robustness: seeds and ablation
- [PENDING: seed runs] Variability across seeds (runs 3 and 4)
- [PENDING: ablation] Effect of the preprocessing chain (run 5)
- Until then: single seed 42 per model; no variance estimate and no claim that the preprocessing helps
> These experiments are still running on the laptop CPU. I will fill this slide when they finish.
> Until then every number in this talk is from a single seed, and I make no claim about run-to-run variance or about the benefit of the preprocessing.
---
layout: figs2
title: Application demo (model {{ModelVersion}})
figure: screenshots/07_result_card, screenshots/08_gradcam_card
caption: Test-split Shenzhen radiograph (true label Tuberculosis). Left: prediction and scores. Right: Grad-CAM overlay.
> These are screenshots of the running application with the final model, not mock-ups. The image comes from the held-out test split of the final dataset, and its true label is Tuberculosis.
> The left panel shows the predicted class with its top-class score and all six softmax scores; the same values are returned by the API and stored in the history.
> The right panel shows the Grad-CAM overlay. It shows image regions that influenced the score; it is not radiological reasoning, and on this image part of the attention falls outside the lung fields.
> One correctly classified example is not evidence of accuracy; the test and external results on the previous slides are.
> Full-page captures of the upload, metrics and history pages are in docs/figures/screenshots/ if the live demo is not possible.
---
layout: bullets
title: Limitations
- Single run per model (seed 42), CPU, time-boxed: no variance estimate
- Classes assumed mutually exclusive; co-findings not modelled
- Patient identity unknown for TBX11K and COVID-19 Radiography rows
- Effusion and nodule data from NIH only; VinDr-CXR and JSRT not available
- External cohort is binary and small ({{NExternal}} scans)
- Heuristic CXR gate; Grad-CAM is not clinical reasoning; no clinical validity claimed
> I want to be clear about what this work does not show. There is one run per model, so no variance.
> Labels are single and exclusive, patient identity is unknown for two sources, and some classes rely on one source dataset.
> The external cohort is small, and the gate is a heuristic. Nothing here establishes clinical validity.
---
layout: bullets
title: Conclusion
- Rebuilt a six-class CXR benchmark from official sources with patient-grouped splits
- Delivered a reproducible pipeline from raw data to served metrics, and a working application
- Showed how leakage and proxy labels inflated the earlier V5 result
- Final model v6.1: {{TestAcc}}% internal accuracy, but external specificity {{ExtSpecK}}/{{ExtSpecN}} and TB recall {{ExtTBK}}/{{ExtTBN}}: internal metrics overstate behaviour
- Class-source confounding is strong; build 2 changed where external errors fall, not the external gap (indirect evidence of a source shortcut)
- Not shown: clinical utility, superiority over other methods, seed robustness, preprocessing benefit (pending)
> To conclude, the contribution is an auditable rebuild and a reproducible pipeline, together with evidence of how an earlier pipeline overstated its results.
> The final model is v6.1, run 2. Its internal accuracy is good, but on the external cohort specificity is low and no TB scan is recognised as TB, so internal metrics overstate generalisation.
> With build 2 the COVID-19 calls fell but the external gap remained; this is consistent with a source shortcut, not proof of one. Seed robustness and the preprocessing ablation remain pending.
> The claim I can defend is about method, not about clinical accuracy.
---
layout: bullets
title: Future scope
- Add VinDr-CXR and JSRT after access to strengthen effusion and nodule classes
- Multi-label formulation for co-findings
- Several seeds and a larger, multi-site external cohort
- Train a CXR gate on real radiographs and non-radiographs
- Clinician-reviewed prospective evaluation, reported per TRIPOD+AI and CLAIM [collins2024tripodai, mongan2020claim]
> Future work follows the limitations. The most useful steps are better data, a multi-label formulation, repeated runs for variance, and a larger multi-site external cohort.
> A trained gate would fix the weakest part of the application.
> Any clinical use would need prospective evaluation with clinicians and reporting following TRIPOD+AI and CLAIM.
---
layout: bullets
title: Publications plan
- Conference (IEEE) and journal (Elsevier, BSPC) drafts in preparation; not submitted [AUTHORS TO CONFIRM]
- Author order: K. Sathwik, N. Ramana, P. Tirupathi, K. Srinivas; corresponding author [TO BE CONFIRMED]
> I plan one conference paper and one journal paper from this work.
> The author order is confirmed by the project owner. The corresponding author is still to be confirmed, and all authors must verify contributions and approve the manuscripts before submission.
> Nothing has been submitted yet; the venue comparison is maintained separately.
---
layout: refs
title: Key references
- wang2017chestxray8: Wang et al., ChestX-ray8, CVPR 2017
- kermany2018identifying: Kermany et al., Cell 2018
- jaeger2014two: Jaeger et al., QIMS 2014
- liu2020rethinking: Liu et al., TBX11K, CVPR 2020
- cohen2020covid: Cohen et al., COVID-19 collection, 2020
- chowdhury2020can: Chowdhury et al., IEEE Access 2020
- huang2017densely: Huang et al., DenseNet, CVPR 2017
- deng2009imagenet: Deng et al., ImageNet, CVPR 2009
- selvaraju2017gradcam: Selvaraju et al., Grad-CAM, ICCV 2017
- zech2018variable: Zech et al., PLoS Med 2018
- degrave2021ai: DeGrave et al., Nat. Mach. Intell. 2021
- geirhos2020shortcut: Geirhos et al., Nat. Mach. Intell. 2020
- roberts2021common: Roberts et al., Nat. Mach. Intell. 2021
- oakdenrayner2020exploring: Oakden-Rayner, Acad. Radiol. 2020
- guo2017calibration: Guo et al., ICML 2017
- wilson1927probable: Wilson, JASA 1927
- kingma2015adam: Kingma and Ba, ICLR 2015
- collins2024tripodai: Collins et al., TRIPOD+AI, BMJ 2024
- mongan2020claim: Mongan et al., CLAIM, Radiol. AI 2020
> These are the main works cited in the talk, by BibTeX key. The complete list, with verification status, is in the shared references file of the repository.
> I can discuss any of them during questions.
---
layout: closing
title: Thank you
subtitle: Questions and discussion
> Thank you for your attention. I would be glad to answer questions about the data audit, the evaluation protocol, the application or the limitations.
> If asked about accuracy, I will point to the confidence intervals and to the external cohort, and note that clinical validity is not claimed.
