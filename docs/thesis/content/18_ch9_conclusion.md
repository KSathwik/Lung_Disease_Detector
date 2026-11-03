# Conclusion and Future Scope

## Summary of the work

This project rebuilt a six-class chest X-ray classification system around a leakage-controlled dataset. It started from an earlier model whose recorded 82.93% internal accuracy did not survive an audit, found the causes (patient leakage, proxy labels, unverifiable identity and an external specificity of zero), and replaced the data pipeline with one that downloads from official sources with logged provenance, applies label rules fixed in advance, hashes every image, splits by patient group with duplicates merged, and stops on any leakage check. A DenseNet-121 was trained with a two-phase transfer-learning schedule on a laptop CPU, after a memory problem was solved by running inference as a compiled graph. The model, its Grad-CAM explanations, a heuristic input gate whose threshold was recalibrated on validation radiographs, and the metrics of the loaded checkpoint are served by a FastAPI and React application that is tested at the unit, integration and browser level.

## Conclusions

For the final model, {{ModelVersion}}, the answers to the three research questions of Chapter 3 are as follows.

1. **Internal versus external.** With a patient-grouped, leakage-controlled split the internal test accuracy is {{TestAcc}}% (macro-F1 {{TestMacroFOne}}%), but on the external hospital the model calls only {{ExtSpecK}} of {{ExtSpecN}} normal scans Normal and none of {{ExtTBN}} tuberculosis scans Tuberculosis. The gap is large and a clean internal split does not close it.
2. **Source shortcuts.** The source of an image is predictable from the features (balanced accuracy {{ProbeMdlBalAcc}}% against a chance of {{ProbeMdlChance}}%), also within a class. The first run's COVID-19 assignments on Montgomery were removed in the final run by adding a second source, but the external errors moved to other classes. The evidence is consistent with source shortcuts and is indirect; it does not prove them.
3. **Cost.** The pipeline, from acquisition to served metrics, runs on a laptop CPU; training the final model took several hours, and each run is a single time-boxed seed.

Seed variability and the preprocessing ablation will be added when those runs finish: [PENDING: seed runs] [PENDING: ablation].

Conclusions that do not depend on the final model are the following.

1. A pipeline in which every step is scripted, hashed and checked can be built and run on commodity hardware within a night, and the checks (zero groups, files and external images spanning splits) can be made part of the build.
2. Audit findings of the earlier system show that a high internal accuracy is not evidence of generalisation, and that external sensitivity without specificity is not a result.
3. The honest evaluation of a multi-source benchmark requires per-source reporting and an external cohort, because the class label is partly a label of the source.
4. A research prototype can be built so that it cannot show a number that it did not compute: the application serves the generated metrics of the checkpoint it loaded and refuses to run with an unverified one.

## Contributions revisited

The four contributions stated in Chapter 1 stand as pipeline and audit contributions: the auditable rebuild, the reproducible end-to-end pipeline, the evidence about internal versus external behaviour internal metrics of {{TestAcc}}% accuracy against an external specificity of {{ExtSpecK}} of {{ExtSpecN}}, and the audit of the earlier pipeline. The project does not claim a new architecture or a clinically useful classifier.

## Future scope

The most valuable extensions follow from the limitations.

- **More seeds.** Train at least five seeds per configuration (about five hours of CPU time each) and report mean and spread, so that differences between configurations can be judged.
- **Better data for the weak classes.** Obtain VinDr-CXR and JSRT through credentialing and registration, to give Pleural Effusion and Nodule / Mass radiologist-read labels and a second source. Add a source with real patient identifiers for COVID-19 and TB.
- **Source-shortcut controls.** A source-prediction probe on frozen features was run in this work. Report leave-one-source-out results for multi-source classes, and test whether performance holds when the border and annotation regions of the image are masked.
- **Preprocessing redesign.** Resize to a canonical size before blur, CLAHE and low-pass filtering, remove the dependence on image resolution, and choose between the variants on validation. Reconsider the horizontal flip, which reverses cardiac laterality.
- **Calibration.** Fit temperature scaling on the validation split and validate the inconclusive threshold.
- **Multi-label formulation.** Replace the single-label assumption by a multi-label output that can represent co-existing findings.
- **Input gate.** Train or calibrate a gate on real radiographs and a real set of non-radiographs, and report its false-reject and false-accept rates.
- **Explanations.** Compare Grad-CAM with other attribution methods and test their localisation against bounding boxes (available in VinDr-CXR and TBX11K), given the doubts about saliency maps.
- **Fairness.** Evaluate performance by age, sex and site where such data exist.
- **Deployment.** Build and test the Docker image, run continuous integration, add user accounts and an audit log, and retention rules for uploads.
- **Prospective clinical evaluation.** Any claim of clinical utility needs ethics approval, representative local data, comparison with radiologists and prospective evaluation, following reporting guidelines such as CLAIM and TRIPOD+AI [@mongan2020claim; @collins2024tripodai].
