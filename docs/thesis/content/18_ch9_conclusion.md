# Conclusion and Future Scope

## Summary of the work

This project designed and developed LungAI, a deep learning system that predicts one of six lung conditions from a frontal chest radiograph. The system consists of a data pipeline, a classifier and a web application. The data pipeline downloads seven public collections from their official sources with logged provenance, applies label rules fixed in advance, hashes every image, splits by patient group with duplicates merged, and stops on any leakage check. The classifier is a DenseNet-121 trained with a two-phase transfer-learning schedule on a laptop CPU, after a memory problem was solved by running inference as a compiled graph. The web application, built with FastAPI and React, checks that an upload looks like a chest radiograph, returns the six class scores with a Grad-CAM overlay, marks low-confidence predictions as inconclusive, stores each prediction with the identity of the model, manages patient records and reports, and shows the metrics of the loaded checkpoint. It is tested at the unit, integration and browser level. The design was shaped by an audit of the first version of the system, whose recorded 82.93% internal accuracy rested on leaked and proxy-labelled data.

## Conclusions

For the final model, {{ModelVersion}}, the answers to the evaluation questions of Chapter 3 are as follows.

1. **Internal performance.** On a patient-grouped, leakage-controlled internal test split the system reaches {{TestAcc}}% accuracy (macro-F1 {{TestMacroFOne}}%, macro ROC-AUC {{TestMacroAUC}}). COVID-19, Tuberculosis and Pneumonia are recognised best, and Pulmonary Nodule / Mass and Normal are the weakest classes. Across three seeds of the final recipe, internal test accuracy is {{MSTestAccMean}} ± {{MSTestAccSD}}% (mean ± sample SD, a rough estimate); the served run is the highest of the three within a spread of under one percentage point, so a small favourable draw cannot be ruled out. In one ablation run at the same seed, the run without the blur, CLAHE and low-pass filter had an internal test accuracy of {{AblTestAcc}}%, so the chosen preprocessing scored higher internally in this single-seed comparison; it does not show that the filters improve generalisation.
2. **External behaviour.** On the external hospital the system flags most tuberculosis scans as abnormal but calls only {{ExtSpecK}} of {{ExtSpecN}} normal scans Normal and none of {{ExtTBN}} tuberculosis scans Tuberculosis, and this recurs in every seed (specificity {{MSExtSpecMin}}% to {{MSExtSpecMax}}%). A clean internal split therefore does not guarantee transfer to a new hospital. The source of an image is predictable from the features (balanced accuracy {{ProbeMdlBalAcc}}% against a chance of {{ProbeMdlChance}}%), which is consistent with source cues contributing to the gap; the evidence is indirect and does not prove that the model uses them.
3. **Cost.** The complete system, from acquisition to a served prediction, runs on a laptop CPU; training the final model (run 2) took about 5.8 hours, and every run is time-boxed.

Conclusions that do not depend on the final model are the following.

1. A prediction system in which every step is scripted, hashed and checked can be built and run on commodity hardware within a night, and the leakage checks (zero groups, files and external images spanning splits) can be made part of the build.
2. An application can be built so that it cannot show a number that it did not compute: it serves the generated metrics of the checkpoint it loaded, refuses to run with an unverified one, and returns an error instead of a placeholder prediction.
3. The audit of the first version shows that a high internal accuracy is no evidence of generalisation, and that external sensitivity without specificity is not a result.
4. An honest evaluation of a multi-source system requires per-source reporting and an external cohort, because in public collections the class label is partly a label of the source.

Supplementary measurements on frozen features (Appendix G) found that source information is spread across all spatial-frequency bands of the cached images and is entangled with disease information, so that neither removing one band nor a linear projection removed it cheaply; a rule fixed in advance therefore did not unlock a band-suppression retraining. These measurements concern frozen features and linear methods and do not describe the served model directly.

## Contributions revisited

Of the five contributions stated in Chapter 1, the first three concern the system itself: its end-to-end design with one shared configuration and preprocessing, the data pipeline that builds its dataset from official sources under recorded rules, and the web application with input checking, Grad-CAM, an inconclusive outcome, traceable predictions and served metrics. The fourth is the evaluation, which reports an internal accuracy of {{TestAcc}}% together with an external specificity of {{ExtSpecK}} of {{ExtSpecN}}, so that the strengths and the limits of the system are both on record. The fifth is the audit of the first version, which explains the design of the data pipeline. The project does not claim a new architecture or a clinically useful classifier.

## Future scope

The extensions below follow from the limitations.

- **Better external performance.** The most important next step is to make the system transfer to new hospitals: add sources with real patient identifiers for COVID-19 and TB, evaluate on a second external cohort, and test lung masking and masking of borders and annotations during training.
- **Better data for the weak classes.** Obtain VinDr-CXR and JSRT through credentialing and registration, to give Pleural Effusion and Nodule / Mass radiologist-read labels and a second source.
- **More seeds and ablation seeds.** Train at least five seeds per configuration (about six hours of CPU time each for the final build), also for the ablation and for other splits, and report mean and spread, so that differences between configurations can be judged.
- **Source-shortcut controls.** Retrain the network itself with a source held out, so that the dependence of the trained model on the source can be measured directly instead of inferred from frozen features (Appendix G).
- **Preprocessing redesign.** Resize to a canonical size before blur, CLAHE and low-pass filtering, remove the dependence on image resolution, and choose between the variants on validation. Reconsider the horizontal flip, which reverses cardiac laterality.
- **Calibration.** Fit temperature scaling on the validation split and validate the inconclusive threshold.
- **Multi-label formulation.** Replace the single-label assumption by a multi-label output that can represent co-existing findings.
- **Input gate.** Train or calibrate a gate on real radiographs and a real set of non-radiographs, and report its false-reject and false-accept rates.
- **Explanations.** Compare Grad-CAM with other attribution methods and test their localisation against bounding boxes (available in VinDr-CXR and TBX11K), given the doubts about saliency maps.
- **Fairness.** Evaluate performance by age, sex and site where such data exist.
- **Deployment.** Build and test the Docker image, add user accounts and an audit log, and retention rules for uploads.
- **Prospective clinical evaluation.** Any claim of clinical utility needs ethics approval, representative local data, comparison with radiologists and prospective evaluation, following reporting guidelines such as CLAIM and TRIPOD+AI [@mongan2020claim; @collins2024tripodai].
