# Experimental Setup and Results

## Experimental setup

### Hardware and software

All training and evaluation were run on one laptop with 8 GB of memory and no GPU, under Windows 11, with Python 3.12, TensorFlow 2.16.1, Keras 3.15.1, NumPy 1.26.4 and OpenCV 4.10. TensorFlow 2.16 has no GPU support on native Windows, so no accelerator was used. Library versions, the code commit and the checkpoint hash are recorded in the metadata of each run.

### Data

The model evaluated in this chapter is {{ModelVersion}}, trained on dataset build {{DatasetBuild}} with the split described in Chapter 5. The training set has {{NTrain}} images, the validation set {{NVal}} and the internal test set {{NTest}}. The external cohort has {{NExternal}} images (58 TB and 80 normal scans from Montgomery County). The checkpoint has the short hash {{CheckpointShort}}. All numbers in this chapter refer to this final model, which was selected over the first V6 run by the rule described in Chapter 6; results of the first run appear only where they are labelled as run 1.

### Training configuration

[[tab:hyper]] lists the training configuration.

Table: Training configuration of the evaluated model. {#tab:hyper}

| Item | Value |
|---|---|
| Backbone | DenseNet-121, ImageNet initialisation |
| Head | GAP, batch normalisation, Dense 256 (ReLU), Dropout 0.3, Dense 6 (softmax) |
| Input | 224 × 224 × 3, ImageNet mean and standard deviation |
| Loss | Sparse categorical cross-entropy with balanced class weights |
| Phase 1 | 2 epochs, backbone frozen, Adam, learning rate 10⁻³ |
| Phase 2 | Last 30 backbone layers trainable, Adam, learning rate 10⁻⁵ |
| Batch size | 32 |
| Augmentation | Horizontal flip (p = 0.5), rotation ±10° with reflected borders |
| Early stopping | Three fine-tuning epochs without validation macro-F1 gain |
| Checkpoint rule | Best validation macro-F1 (tie: validation macro ROC-AUC) |
| Seed | {{Seed}} for the final model; seeds {{MSSeeds}} in the seed study |
| Epochs trained | {{EpochsTrained}}; selected epoch {{SelectedEpoch}} |

### Evaluation protocol

The validation split is evaluated after every epoch and is used to choose the checkpoint. The test split and the external cohort are evaluated once, with the final flag, after the checkpoint is chosen. Confidence intervals for accuracy and recall are Wilson intervals; the bootstrap interval for accuracy uses 2,000 resamples with seed 42. Calibration uses 15 bins. The definitions are in Chapter 4.

### What is not done

No hyper-parameter search was run. Run-to-run variation is measured only by the follow-up seed runs reported in the section on seed variability, and the preprocessing is examined by one ablation run (section on the preprocessing ablation). No threshold was tuned on the test split or the external cohort. No calibration method was fitted. No comparison with other published systems is made.

## Training behaviour

[[fig:curves]] shows the training loss and the validation accuracy, macro-F1 and macro ROC-AUC per epoch. The checkpoint was selected at epoch {{SelectedEpoch}} out of {{EpochsTrained}}.

![fig_training_curves](Training loss and validation accuracy, macro-F1 and macro ROC-AUC per epoch.) {#fig:curves}

The run completed its full schedule of {{EpochsTrained}} epochs (the head epochs followed by fine-tuning) and was not stopped early. Validation macro-F1 rose slowly during fine-tuning and was highest at the last epoch ({{SelectedEpoch}}), where it was {{ValMacroFOne}}%. The validation curve was therefore still rising slightly when training ended, and a longer schedule might change the result; this was not tested.

## Internal results

### Overall performance

[[tab:overall]] gives the overall metrics on the validation and test splits.

Table: Overall performance of the final model on the validation and the internal test split. {#tab:overall}

| Split | N | Accuracy (%) | Macro F1 (%) | Macro ROC-AUC |
|---|---|---|---|---|
| Validation | {{NVal}} | {{ValAcc}} | {{ValMacroFOne}} | {{ValMacroAUC}} |
| Test | {{NTest}} | {{TestAcc}} (95% CI {{TestAccLo}} to {{TestAccHi}}) | {{TestMacroFOne}} | {{TestMacroAUC}} |

On the test split the weighted F1 is {{TestWeightedFOne}}%, the macro recall {{TestMacroRecall}}%, the macro precision {{TestMacroPrecision}}%, the macro PR-AUC {{TestMacroPRAUC}} and the expected calibration error {{TestECE}}.

### Per-class performance

[[tab:perclass]] gives the per-class results on the test split and [[fig:perclass]] shows precision, recall and F1 with the recall intervals.

Table: Per-class results on the internal test split (percentages except ROC-AUC). {#tab:perclass}

| Class | Support | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| COVID-19 | {{SupportCovid}} | {{PrecisionCovid}} | {{RecallCovid}} | {{SpecificityCovid}} | {{FOneCovid}} | {{AUCCovid}} |
| Normal | {{SupportNormal}} | {{PrecisionNormal}} | {{RecallNormal}} | {{SpecificityNormal}} | {{FOneNormal}} | {{AUCNormal}} |
| Pleural Effusion | {{SupportEffusion}} | {{PrecisionEffusion}} | {{RecallEffusion}} | {{SpecificityEffusion}} | {{FOneEffusion}} | {{AUCEffusion}} |
| Pneumonia | {{SupportPneumonia}} | {{PrecisionPneumonia}} | {{RecallPneumonia}} | {{SpecificityPneumonia}} | {{FOnePneumonia}} | {{AUCPneumonia}} |
| Pulmonary Nodule / Mass | {{SupportNodule}} | {{PrecisionNodule}} | {{RecallNodule}} | {{SpecificityNodule}} | {{FOneNodule}} | {{AUCNodule}} |
| Tuberculosis | {{SupportTB}} | {{PrecisionTB}} | {{RecallTB}} | {{SpecificityTB}} | {{FOneTB}} | {{AUCTB}} |

![fig_per_class_metrics](Precision, recall and F1 per class on the internal test split, with Wilson 95% intervals for recall.) {#fig:perclass}

### Confusion matrix and ROC curves

[[fig:confusion]] shows the confusion matrix of the test split, as counts and as row-normalised proportions, and [[fig:roc]] the one-versus-rest ROC curves.

![fig_confusion_matrix](Confusion matrix on the internal test split. Rows are true classes and columns are predicted classes.) {#fig:confusion}

![fig_roc_curves](One-versus-rest ROC curves per class on the internal test split.) {#fig:roc}

### Results by source

The internal test split mixes sources whose difficulty differs greatly, so overall accuracy depends on the source mix. [[fig:persource]] shows accuracy by source.

![fig_per_source](Accuracy on the internal test split by source.) {#fig:persource}

### Calibration

[[fig:calibration]] is the reliability diagram on the test split. The expected calibration error is {{TestECE}}. No calibration was fitted, so the scores shown by the application are uncalibrated softmax outputs.

![fig_calibration](Reliability diagram of the top-class score on the internal test split.) {#fig:calibration}

## External results

On the Montgomery County cohort the task is binary. [[tab:external]] reports the results. Sensitivity is reported only together with specificity.

Table: External results on the Montgomery County cohort, evaluated once. {#tab:external}

| Quantity | Result |
|---|---|
| TB scans flagged abnormal (sensitivity) | {{ExtSensK}} / {{ExtSensN}} ({{ExtSens}}%; 95% CI {{ExtSensLo}} to {{ExtSensHi}}) |
| Normal scans called Normal (specificity) | {{ExtSpecK}} / {{ExtSpecN}} ({{ExtSpec}}%; 95% CI {{ExtSpecLo}} to {{ExtSpecHi}}) |
| TB scans called Tuberculosis (exact recall) | {{ExtTBK}} / {{ExtTBN}} ({{ExtTB}}%) |
| ROC-AUC of P(abnormal) | {{ExtAUC}} |
| Scans assigned to COVID-19 | {{ExtCovidCalls}} |

[[fig:external]] shows the class assigned to the normal and TB scans. Montgomery contains no COVID-19 cases, so any scan assigned to COVID-19 is a wrong class.

![fig_external_breakdown](Predicted class for the normal and the tuberculosis scans of the Montgomery County cohort.) {#fig:external}

## Source-identifiability probe

Because class and source are confounded in the benchmark, a control was run on the final build. A linear classifier was fitted on frozen features of {{ProbeNTrain}} training images and evaluated on {{ProbeNTest}} test images to predict the source of an image rather than its class. [[tab:probe]] gives the balanced accuracy.

Table: Balanced accuracy (%) of a linear probe that predicts the source of a test image. Chance is the accuracy of a random guess among the sources considered. {#tab:probe}

| Features | All sources ({{ProbeImgNSources}}; chance {{ProbeImgChance}}) | Within Normal ({{ProbeImgWithinNormalNSources}} sources; chance {{ProbeImgWithinNormalChance}}) | Within Tuberculosis ({{ProbeImgWithinTBNSources}}; chance {{ProbeImgWithinTBChance}}) | Within Pneumonia ({{ProbeImgWithinPneumoniaNSources}}; chance {{ProbeImgWithinPneumoniaChance}}) | Within COVID-19 ({{ProbeImgWithinCovidNSources}}; chance {{ProbeImgWithinCovidChance}}) |
|---|---|---|---|---|---|
| ImageNet features | {{ProbeImgBalAcc}} | {{ProbeImgWithinNormal}} | {{ProbeImgWithinTB}} | {{ProbeImgWithinPneumonia}} | {{ProbeImgWithinCovid}} |
| Backbone of the trained model | {{ProbeMdlBalAcc}} | {{ProbeMdlWithinNormal}} | {{ProbeMdlWithinTB}} | {{ProbeMdlWithinPneumonia}} | {{ProbeMdlWithinCovid}} |

The source is predicted far above chance, and also within a single class, where the label is the same and only the source differs. Fine-tuning changed only the last 17 weighted layers of the backbone (verified by comparing weights), so the features of the trained model stay close to those of ImageNet, which is why the two rows are similar. The probe shows that source information is linearly available in the features the classifier uses. It does not show that the classifier uses it.

## Grad-CAM examples

[[fig:gradcam]] shows Grad-CAM overlays for a few test images. They illustrate the output of the application and are not evidence of localisation. The images were chosen by a rule fixed before the heatmaps were viewed: one correctly classified test image per class, at the median confidence of that class. In these examples the COVID-19 and Normal heatmaps concentrate on the neck and trachea, the shoulders and the image borders, the Nodule / Mass and Tuberculosis heatmaps fall inside the lung fields, and the Pneumonia heatmap, on a paediatric Kermany image, is diffuse. With one image per class this is qualitative and illustrative only.

![fig_gradcam_examples](Grad-CAM overlays on a few images of the internal test split, for the predicted class.) {#fig:gradcam}

## Comparison of the two V6 runs

The two runs were compared on the {{CmpN}} validation images that neither build used for training or testing, by the rule fixed in advance (Chapter 6). [[tab:runcmp]] gives the result.

Table: Run 1 and run 2 on the common validation set, overall and restricted to the images from the four sources both runs trained on. {#tab:runcmp}

| Set | Run | N | Accuracy (%) | Macro F1 (%) | Macro ROC-AUC |
|---|---|---|---|---|---|
| Common set | Run 1 | {{CmpRunOneN}} | {{CmpRunOneAcc}} | {{CmpRunOneMacroFOne}} | {{CmpRunOneMacroAUC}} |
| Common set | Run 2 (final) | {{CmpRunTwoN}} | {{CmpRunTwoAcc}} | {{CmpRunTwoMacroFOne}} | {{CmpRunTwoMacroAUC}} |
| Shared sources only | Run 1 | {{CmpRunOneSharedN}} | {{CmpRunOneSharedAcc}} | {{CmpRunOneSharedMacroFOne}} | {{CmpRunOneSharedMacroAUC}} |
| Shared sources only | Run 2 (final) | {{CmpRunTwoSharedN}} | {{CmpRunTwoSharedAcc}} | {{CmpRunTwoSharedMacroFOne}} | {{CmpRunTwoSharedMacroAUC}} |

Run 2 has the higher macro-F1 and was selected. On the paired images, run 1 was right and run 2 wrong for {{CmpMcNemarB}} images and the reverse held for {{CmpMcNemarC}} (exact McNemar p = {{CmpMcNemarP}}; descriptive only). The caveat is the shared-sources rows: on the images from the four sources both runs trained on, accuracy is almost equal and the macro-F1 gap is smaller, so much of the advantage of run 2 lies on TBX11K and COVID-19 Radiography images, which only run 2 saw in training. Each run is a single seed, so the comparison does not show that the recipe of run 2 is better in general. Run 2 was started after the external results of run 1 were known; only the selection rule was fixed before run 2's results.

[[tab:extcmp]] puts the external behaviour of the two runs side by side. The run 1 values are historical and come from the external evaluation file of run 1 (`artifacts/evaluation_reports/v6_external/metrics_external.json`); each run was evaluated on Montgomery once.

Table: External behaviour of run 1 (not selected; historical values) and run 2 (final) on the Montgomery scans, none of which show COVID-19. {#tab:extcmp}

| Quantity | Run 1 | Run 2 (final) |
|---|---|---|
| Scans assigned to COVID-19 | {{RunOneExtCovidCalls}} of {{NExternal}} | {{ExtCovidCalls}} of {{NExternal}} |
| Normal scans called Normal | {{RunOneExtSpecK}} / {{ExtSpecN}} | {{ExtSpecK}} / {{ExtSpecN}} |
| TB scans called Tuberculosis | {{RunOneExtTBK}} / {{ExtTBN}} | {{ExtTBK}} / {{ExtTBN}} |
| ROC-AUC of P(abnormal) | {{RunOneExtAUC}} | {{ExtAUC}} |

## Seed variability

The recipe of the final model was trained with {{MSNRuns}} seeds ({{MSSeeds}}) on the same build, the same split (identical fingerprint), the same schedule and the same checkpoint rule; only the seed changed. Run 2 is the served model. Runs 3 and 4 were stopped early by the rule of three fine-tuning epochs without validation macro-F1 gain (best epochs 6 and 7), whereas run 2 completed all 16 epochs; the same rule applied to all runs, so this is part of the variability of the recipe. The test split and the Montgomery cohort were evaluated once per run. [[tab:seeds]] gives the results.

Table: Seed variability of the final recipe (runs 2 to 4; same build, split and schedule). Test metrics in percent except ROC-AUC; Montgomery columns in percent. The last row gives the mean and the sample standard deviation over the three runs. {#tab:seeds}

| Run | Seed | Test accuracy | Test macro-F1 | Test macro ROC-AUC | Ext. sensitivity | Ext. specificity | Exact TB recall |
|---|---|---|---|---|---|---|---|
| 2 (served) | {{MSRunOneSeed}} | {{MSRunOneTestAcc}} | {{MSRunOneTestMacroFOne}} | {{MSRunOneTestMacroAUC}} | {{MSRunOneExtSens}} | {{MSRunOneExtSpec}} | {{MSRunOneExtTB}} |
| 3 | {{MSRunTwoSeed}} | {{MSRunTwoTestAcc}} | {{MSRunTwoTestMacroFOne}} | {{MSRunTwoTestMacroAUC}} | {{MSRunTwoExtSens}} | {{MSRunTwoExtSpec}} | {{MSRunTwoExtTB}} |
| 4 | {{MSRunThreeSeed}} | {{MSRunThreeTestAcc}} | {{MSRunThreeTestMacroFOne}} | {{MSRunThreeTestMacroAUC}} | {{MSRunThreeExtSens}} | {{MSRunThreeExtSpec}} | {{MSRunThreeExtTB}} |
| Mean ± SD | | {{MSTestAccMean}} ± {{MSTestAccSD}} | {{MSTestMacroFOneMean}} ± {{MSTestMacroFOneSD}} | {{MSTestMacroAUCMean}} ± {{MSTestMacroAUCSD}} | {{MSExtSensMean}} ± {{MSExtSensSD}} | {{MSExtSpecMean}} ± {{MSExtSpecSD}} | {{MSExtTBMean}} ± {{MSExtTBSD}} |

The internal results vary little with the seed. Test accuracy ranges from {{MSTestAccMin}}% to {{MSTestAccMax}}% (standard deviation {{MSTestAccSD}} points), macro-F1 from {{MSTestMacroFOneMin}}% to {{MSTestMacroFOneMax}}% and macro ROC-AUC from {{MSTestMacroAUCMin}} to {{MSTestMacroAUCMax}}. The served run 2 has the highest internal values of the three seeds, but all three lie within a spread of under one percentage point, so with this recipe and split it is not an outlier; three seeds cannot rule out a small favourable draw. The external behaviour is consistently poor: specificity ranges from {{MSExtSpecMin}}% to {{MSExtSpecMax}}% (mean {{MSExtSpecMean}}%), so it varies more between seeds than the internal metrics, and no seed names any Montgomery tuberculosis scan Tuberculosis. The external failure is thus not an artefact of one seed. The standard deviation is the sample standard deviation over three runs, a rough estimate, and it describes the seed only: the split, the data and the architecture were fixed. The served model stays run 2, which was chosen before seeds 43 and 44 existed; no seed was selected on test or external results.

## Preprocessing ablation

One further run (run 5) used the same build, the same split fingerprint, the same seed ({{MSRunOneSeed}}) and the same schedule as run 2, but images cached without the Gaussian blur, CLAHE and low-pass filter; the resize and the normalisation were unchanged. It completed all 16 epochs and selected its last epoch (16 of 16), with the validation score still rising. [[tab:ablation]] compares it with run 2.

Table: Preprocessing ablation. Run 2 (with the filters) and run 5 (without them), same seed. Percentages except ROC-AUC. {#tab:ablation}

| Quantity | Run 2 (filters) | Run 5 (no filters) |
|---|---|---|
| Test accuracy | {{MSRunOneTestAcc}} | {{AblTestAcc}} |
| Test macro-F1 | {{MSRunOneTestMacroFOne}} | {{AblTestMacroFOne}} |
| Test macro ROC-AUC | {{MSRunOneTestMacroAUC}} | {{AblTestMacroAUC}} |
| Ext. sensitivity | {{MSRunOneExtSens}} | {{AblExtSens}} |
| Ext. specificity | {{MSRunOneExtSpec}} | {{AblExtSpec}} |
| Exact TB recall | {{MSRunOneExtTB}} | {{AblExtTB}} |

Without the filters, internal accuracy and macro-F1 are lower, by more than the seed-to-seed spread of the previous section, and the run-5 internal values lie below the whole range of runs 2 to 4. In this comparison the run with the filters therefore scored higher internally. Montgomery specificity is unchanged, so the main external failure is present with and without the filters. The run without filters named a small number of Montgomery tuberculosis scans Tuberculosis ({{AblExtTB}}% of the scans) where no filtered run did; this is a small count from one run and is reported as an observation, not as an effect. Run 5 also selected its last epoch, so a difference in convergence speed cannot be excluded. The comparison is a single ablation seed of one alternative: it does not show that the preprocessing improves generalisation, and it does not identify which filter matters.

## Source-information analyses

The probe of the previous section gives one number for the whole image. This section reports four further analyses on frozen features whose procedures are given in Chapter 4: a map of source and disease information over spatial-frequency bands, linear source erasure, leave-one-source-out linear heads, and a band-stop screening with a go/no-go decision. The ImageNet features are the primary extractor. The backbone of the final model is secondary, because it was trained on all six sources. The no-filter cache is a sensitivity check. All results describe the cached 224 × 224 images, one run and one fixed probe sample, and all are subject to the patient-identity caveat of Chapter 5.

### Spectral map

[[fig:spectral]] shows the band-pass and band-stop results on the test sample of {{SpecMainTestN}} images; [[tab:specsrc]] and [[tab:specdis]] give the values of panel (a), and [[tab:specstop]] the values of panel (b).

![fig_spectral_map](Spectral map of source and disease information on the test sample. Panel (a), band-pass images A to F: source balanced accuracy (BA, blue) and within-source disease macro-F1 (orange). Solid lines with circles are ImageNet features (primary), dashed lines with squares the run-2 backbone (secondary; it saw all six sources), dotted lines with triangles ImageNet features of the no-filter cache (sensitivity check). Dash-dot horizontal lines are the full-image levels of the primary extractor and the grey dotted line is chance for the source task (16.7%). The vertical axis mixes balanced accuracy (blue) and macro-F1 (orange). Panel (b), band-stop differences for ImageNet features: full image minus band-stop image, so a positive value is information lost by removing the band (ΔS for source BA, ΔD for within-source macro-F1). Error bars are 95% bootstrap intervals over scored images only; each probe was fitted once.) {#fig:spectral}

Table: Source balanced accuracy (BA, %) of a linear probe on band-pass images, test sample (chance 16.7%). The within-Normal column is the source accuracy among Normal images of four sources (nominal chance 25%). Intervals are 95% bootstrap intervals over test images. {#tab:specsrc}

| Variant | Source BA, ImageNet (95% interval) | Source BA, run-2 backbone | Source BA, no-filter cache | Within-Normal source accuracy, ImageNet |
|---|---|---|---|---|
| Full image | {{SpecMainTestFullSrcBA}} ({{SpecMainTestFullSrcBALo}} to {{SpecMainTestFullSrcBAHi}}) | {{SpecServedTestFullSrcBA}} | {{SpecNofilterTestFullSrcBA}} | {{SpecMainTestFullNormBA}} |
| Band-pass A | {{SpecMainTestPassASrcBA}} ({{SpecMainTestPassASrcBALo}} to {{SpecMainTestPassASrcBAHi}}) | {{SpecServedTestPassASrcBA}} | {{SpecNofilterTestPassASrcBA}} | {{SpecMainTestPassANormBA}} |
| Band-pass B | {{SpecMainTestPassBSrcBA}} ({{SpecMainTestPassBSrcBALo}} to {{SpecMainTestPassBSrcBAHi}}) | {{SpecServedTestPassBSrcBA}} | {{SpecNofilterTestPassBSrcBA}} | {{SpecMainTestPassBNormBA}} |
| Band-pass C | {{SpecMainTestPassCSrcBA}} ({{SpecMainTestPassCSrcBALo}} to {{SpecMainTestPassCSrcBAHi}}) | {{SpecServedTestPassCSrcBA}} | {{SpecNofilterTestPassCSrcBA}} | {{SpecMainTestPassCNormBA}} |
| Band-pass D | {{SpecMainTestPassDSrcBA}} ({{SpecMainTestPassDSrcBALo}} to {{SpecMainTestPassDSrcBAHi}}) | {{SpecServedTestPassDSrcBA}} | {{SpecNofilterTestPassDSrcBA}} | {{SpecMainTestPassDNormBA}} |
| Band-pass E | {{SpecMainTestPassESrcBA}} ({{SpecMainTestPassESrcBALo}} to {{SpecMainTestPassESrcBAHi}}) | {{SpecServedTestPassESrcBA}} | {{SpecNofilterTestPassESrcBA}} | {{SpecMainTestPassENormBA}} |
| Band-pass F | {{SpecMainTestPassFSrcBA}} ({{SpecMainTestPassFSrcBALo}} to {{SpecMainTestPassFSrcBAHi}}) | {{SpecServedTestPassFSrcBA}} | {{SpecNofilterTestPassFSrcBA}} | {{SpecMainTestPassFNormBA}} |

Table: Within-source disease macro-F1 (%) of a linear probe on band-pass images, test sample. Same probe sample and intervals as in [[tab:specsrc]]. {#tab:specdis}

| Variant | Within-source F1, ImageNet (95% interval) | Within-source F1, run-2 backbone | Within-source F1, no-filter cache |
|---|---|---|---|
| Full image | {{SpecMainTestFullFWs}} ({{SpecMainTestFullFWsLo}} to {{SpecMainTestFullFWsHi}}) | {{SpecServedTestFullFWs}} | {{SpecNofilterTestFullFWs}} |
| Band-pass A | {{SpecMainTestPassAFWs}} ({{SpecMainTestPassAFWsLo}} to {{SpecMainTestPassAFWsHi}}) | {{SpecServedTestPassAFWs}} | {{SpecNofilterTestPassAFWs}} |
| Band-pass B | {{SpecMainTestPassBFWs}} ({{SpecMainTestPassBFWsLo}} to {{SpecMainTestPassBFWsHi}}) | {{SpecServedTestPassBFWs}} | {{SpecNofilterTestPassBFWs}} |
| Band-pass C | {{SpecMainTestPassCFWs}} ({{SpecMainTestPassCFWsLo}} to {{SpecMainTestPassCFWsHi}}) | {{SpecServedTestPassCFWs}} | {{SpecNofilterTestPassCFWs}} |
| Band-pass D | {{SpecMainTestPassDFWs}} ({{SpecMainTestPassDFWsLo}} to {{SpecMainTestPassDFWsHi}}) | {{SpecServedTestPassDFWs}} | {{SpecNofilterTestPassDFWs}} |
| Band-pass E | {{SpecMainTestPassEFWs}} ({{SpecMainTestPassEFWsLo}} to {{SpecMainTestPassEFWsHi}}) | {{SpecServedTestPassEFWs}} | {{SpecNofilterTestPassEFWs}} |
| Band-pass F | {{SpecMainTestPassFFWs}} ({{SpecMainTestPassFFWsLo}} to {{SpecMainTestPassFFWsHi}}) | {{SpecServedTestPassFFWs}} | {{SpecNofilterTestPassFFWs}} |

Source information is present in every band-pass image. With ImageNet features the source balanced accuracy is {{SpecMainTestPassASrcBA}}% in band A, dips to {{SpecMainTestPassBSrcBA}}% in band B and rises to {{SpecMainTestPassESrcBA}}% in band E and {{SpecMainTestPassFSrcBA}}% in band F, against {{SpecMainTestFullSrcBA}}% for the full image and a chance level of 16.7%. Within Normal images alone, where the class is the same, the source accuracy is above the nominal 25% in every band, from {{SpecMainTestPassANormBA}}% in band A to {{SpecMainTestPassFNormBA}}% in band F. Disease information, measured as the within-source macro-F1, is also spread over the bands ({{SpecMainTestPassBFWs}}% in band B, {{SpecMainTestPassFFWs}}% in band F, {{SpecMainTestFullFWs}}% for the full image). Both curves are highest in the two highest bands. The run-2 backbone and the no-filter cache give profiles of the same shape, with a dip at band B and a rise toward the high bands. In the no-filter cache the source accuracy in band E ({{SpecNofilterTestPassESrcBA}}%, against {{SpecMainTestPassESrcBA}}% for the filtered cache) and band F ({{SpecNofilterTestPassFSrcBA}}% against {{SpecMainTestPassFSrcBA}}%) is lower and the intervals for band E do not overlap, while bands A to C are slightly higher. This is consistent with the pre-filters and the resampling contributing to the high-band source signal. It is a sensitivity result and not a test.

Table: Band-stop differences on the test sample for ImageNet features (percentage points; full image minus band-stop image, so a positive value is information lost by removing the band). ΔS is the change in source balanced accuracy and ΔD the change in within-source disease macro-F1. The last column is the validation value of ΔS used for selection. {#tab:specstop}

| Band removed | ΔS, test (95% interval) | ΔD, test (95% interval) | ΔS, validation |
|---|---|---|---|
| A | {{SpecMainTestStopADeltaS}} ({{SpecMainTestStopADeltaSLo}} to {{SpecMainTestStopADeltaSHi}}) | {{SpecMainTestStopADeltaD}} ({{SpecMainTestStopADeltaDLo}} to {{SpecMainTestStopADeltaDHi}}) | {{SpecMainValStopADeltaS}} |
| B | {{SpecMainTestStopBDeltaS}} ({{SpecMainTestStopBDeltaSLo}} to {{SpecMainTestStopBDeltaSHi}}) | {{SpecMainTestStopBDeltaD}} ({{SpecMainTestStopBDeltaDLo}} to {{SpecMainTestStopBDeltaDHi}}) | {{SpecMainValStopBDeltaS}} |
| C | {{SpecMainTestStopCDeltaS}} ({{SpecMainTestStopCDeltaSLo}} to {{SpecMainTestStopCDeltaSHi}}) | {{SpecMainTestStopCDeltaD}} ({{SpecMainTestStopCDeltaDLo}} to {{SpecMainTestStopCDeltaDHi}}) | {{SpecMainValStopCDeltaS}} |
| D | {{SpecMainTestStopDDeltaS}} ({{SpecMainTestStopDDeltaSLo}} to {{SpecMainTestStopDDeltaSHi}}) | {{SpecMainTestStopDDeltaD}} ({{SpecMainTestStopDDeltaDLo}} to {{SpecMainTestStopDDeltaDHi}}) | {{SpecMainValStopDDeltaS}} |
| E | {{SpecMainTestStopEDeltaS}} ({{SpecMainTestStopEDeltaSLo}} to {{SpecMainTestStopEDeltaSHi}}) | {{SpecMainTestStopEDeltaD}} ({{SpecMainTestStopEDeltaDLo}} to {{SpecMainTestStopEDeltaDHi}}) | {{SpecMainValStopEDeltaS}} |
| F | {{SpecMainTestStopFDeltaS}} ({{SpecMainTestStopFDeltaSLo}} to {{SpecMainTestStopFDeltaSHi}}) | {{SpecMainTestStopFDeltaD}} ({{SpecMainTestStopFDeltaDLo}} to {{SpecMainTestStopFDeltaDHi}}) | {{SpecMainValStopFDeltaS}} |

Removing a single band leaves the source information nearly intact. The largest loss is {{SpecMainTestStopFDeltaS}} points, for band F, with an interval from {{SpecMainTestStopFDeltaSLo}} to {{SpecMainTestStopFDeltaSHi}}. Band D shows a small change in the opposite direction ({{SpecMainTestStopDDeltaS}} points, interval {{SpecMainTestStopDDeltaSLo}} to {{SpecMainTestStopDDeltaSHi}}), and the other four intervals include zero. No band-stop change reaches the 5-point threshold fixed in advance. The intervals of ΔD all include zero; the largest point estimate is {{SpecMainTestStopEDeltaD}} points, for band E. The plain finding is a null result: no band stands out as the carrier of source information, and no band is necessary for it in these features.

### Linear source erasure

[[fig:inlp]] shows how the source accuracy that a fresh linear probe can reach and the within-source disease macro-F1 change as INLP [@ravfogel2020null] removes source directions; [[tab:inlp]] gives the test values at the start and at the end, and the values after a one-step removal with LEACE [@belrose2023leace].

![fig_inlp](Source balanced accuracy (BA, blue) and within-source disease macro-F1 (orange) against the INLP iteration k, on the validation split (solid) and the test split (dashed, lighter). Left, ImageNet features (primary); right, run-2 backbone (secondary; trained on all six sources, so not held-out). The grey dotted line is chance for the source task (16.7%), the vertical dashed line is the iteration cap (k = {{InlpImgKStar}}), and the values at k = 0 are the levels of the full image. The vertical axis mixes balanced accuracy and macro-F1. The figure has no error bars; intervals for the change at k = {{InlpImgKStar}} are in the table. One run, one fit per iteration.) {#fig:inlp}

Table: Linear source erasure on frozen features of the full image, internal test split ({{InlpImgNTest}} images). BA is balanced accuracy of a fresh source probe (chance 16.7%); the F1 columns are macro-F1 (%); k is the INLP iteration. {#tab:inlp}

| Features and step | Source BA | Within-source F1 | Six-class macro-F1 | Accuracy | Macro ROC-AUC |
|---|---|---|---|---|---|
| ImageNet, unprojected (k = 0) | {{InlpImgTestSrcBAKZero}} | {{InlpImgTestFWsKZero}} | {{InlpImgTestFAllKZero}} | {{InlpImgTestAccKZero}} | {{InlpImgTestAUCKZero}} |
| ImageNet, INLP at k = {{InlpImgKStar}} | {{InlpImgTestSrcBAKStar}} | {{InlpImgTestFWsKStar}} | {{InlpImgTestFAllKStar}} | {{InlpImgTestAccKStar}} | {{InlpImgTestAUCKStar}} |
| ImageNet, LEACE | {{InlpImgLeaceTestSrcBA}} | {{InlpImgLeaceTestFWs}} | {{InlpImgLeaceTestFAll}} | {{InlpImgLeaceTestAcc}} | {{InlpImgLeaceTestAUC}} |
| Run-2 backbone, unprojected (k = 0) | {{InlpMdlTestSrcBAKZero}} | {{InlpMdlTestFWsKZero}} | {{InlpMdlTestFAllKZero}} | {{InlpMdlTestAccKZero}} | {{InlpMdlTestAUCKZero}} |
| Run-2 backbone, INLP at k = {{InlpMdlKStar}} | {{InlpMdlTestSrcBAKStar}} | {{InlpMdlTestFWsKStar}} | {{InlpMdlTestFAllKStar}} | {{InlpMdlTestAccKStar}} | {{InlpMdlTestAUCKStar}} |
| Run-2 backbone, LEACE | {{InlpMdlLeaceTestSrcBA}} | {{InlpMdlLeaceTestFWs}} | {{InlpMdlLeaceTestFAll}} | {{InlpMdlLeaceTestAcc}} | {{InlpMdlLeaceTestAUC}} |

Both extractors reached the iteration cap of {{InlpImgKStar}} and were not stopped by the rule on source accuracy (stopped by: {{InlpImgStoppedBy}}). With ImageNet features, {{InlpImgKStar}} iterations lowered the source accuracy that a fresh probe reaches from {{InlpImgTestSrcBAKZero}}% to {{InlpImgTestSrcBAKStar}}%. It remained well above chance, so this linear route did not remove the source information within the cap. The within-source disease macro-F1 fell from {{InlpImgTestFWsKZero}}% to {{InlpImgTestFWsKStar}}% (change {{InlpImgTestChangeFWsKStar}} points, 95% interval {{InlpImgTestChangeFWsKStarLo}} to {{InlpImgTestChangeFWsKStarHi}}) and the six-class macro-F1 from {{InlpImgTestFAllKZero}}% to {{InlpImgTestFAllKStar}}% (change {{InlpImgTestChangeFAllKStar}} points, interval {{InlpImgTestChangeFAllKStarLo}} to {{InlpImgTestChangeFAllKStarHi}}). The six-class loss is the larger of the two. LEACE removes all linearly decodable source information in one step, with source accuracy at chance in the fit and at {{InlpImgLeaceTestSrcBA}}% on the test split. It changed the six-class macro-F1 by {{InlpImgTestChangeFAllLeace}} points and the within-source macro-F1 by {{InlpImgTestChangeFWsLeace}} points. The trained backbone behaves in the same way (source accuracy from {{InlpMdlTestSrcBAKZero}}% to {{InlpMdlTestSrcBAKStar}}%, within-source macro-F1 change {{InlpMdlTestChangeFWsKStar}} points with interval {{InlpMdlTestChangeFWsKStarLo}} to {{InlpMdlTestChangeFWsKStarHi}}, six-class change {{InlpMdlTestChangeFAllKStar}}).

[[tab:inlpext]] gives the Montgomery results of the same heads. They were computed once, Montgomery is not untouched, and the cohort has only 138 images.

Table: Montgomery results of the linear heads before and after INLP (Normal against Tuberculosis; one evaluation per head). Sensitivity and specificity are reported together. The same 138 scans are scored at both steps, so the two rows of one extractor are not independent. {#tab:inlpext}

| Features and step | Sensitivity | Specificity (Wilson 95% interval) | ROC-AUC of P(abnormal) |
|---|---|---|---|
| ImageNet, unprojected (k = 0) | {{InlpImgExtSensKZeroK}} / {{InlpImgExtSensKZeroN}} ({{InlpImgExtSensKZero}}%) | {{InlpImgExtSpecKZeroK}} / {{InlpImgExtSpecKZeroN}} ({{InlpImgExtSpecKZero}}%; {{InlpImgExtSpecKZeroLo}} to {{InlpImgExtSpecKZeroHi}}) | {{InlpImgExtAUCKZero}} |
| ImageNet, INLP, k = {{InlpImgKStar}} | {{InlpImgExtSensKStarK}} / {{InlpImgExtSensKStarN}} ({{InlpImgExtSensKStar}}%) | {{InlpImgExtSpecKStarK}} / {{InlpImgExtSpecKStarN}} ({{InlpImgExtSpecKStar}}%; {{InlpImgExtSpecKStarLo}} to {{InlpImgExtSpecKStarHi}}) | {{InlpImgExtAUCKStar}} |
| Run-2 backbone, unprojected (k = 0) | {{InlpMdlExtSensKZeroK}} / {{InlpMdlExtSensKZeroN}} ({{InlpMdlExtSensKZero}}%) | {{InlpMdlExtSpecKZeroK}} / {{InlpMdlExtSpecKZeroN}} ({{InlpMdlExtSpecKZero}}%; {{InlpMdlExtSpecKZeroLo}} to {{InlpMdlExtSpecKZeroHi}}) | {{InlpMdlExtAUCKZero}} |
| Run-2 backbone, INLP, k = {{InlpMdlKStar}} | {{InlpMdlExtSensKStarK}} / {{InlpMdlExtSensKStarN}} ({{InlpMdlExtSensKStar}}%) | {{InlpMdlExtSpecKStarK}} / {{InlpMdlExtSpecKStarN}} ({{InlpMdlExtSpecKStar}}%; {{InlpMdlExtSpecKStarLo}} to {{InlpMdlExtSpecKStarHi}}) | {{InlpMdlExtAUCKStar}} |

For ImageNet features the specificity is {{InlpImgExtSpecKStar}}% at the last iteration against {{InlpImgExtSpecKZero}}% before projection, with overlapping intervals; for the trained backbone it is {{InlpMdlExtSpecKStar}}% at both steps. No conclusion about a change in specificity is drawn from 138 scans.

### Leave-one-source-out

The held-out design follows Yao et al. [@yao2019strong]. [[fig:loso]] and [[tab:losoimg]] give the results for ImageNet features, and [[tab:losomdl]] those for the trained backbone.

![fig_loso_matrix](Linear-head leave-one-source-out on frozen features. Rows are the held-out sources; columns are the macro-F1 on the held-out test split (LOSO), the macro-F1 of the in-distribution reference trained on all six sources, and the gap (reference minus LOSO), all as fractions. Macro-F1 is computed on the evaluated classes of the source, and predictions of any other class count as errors. Left, ImageNet features (primary); right, run-2 backbone (secondary; it saw all sources, so these are not held-out results). The colour scale is shared by the three columns and is not comparable between F1 and gap; the printed values carry the information. There are no intervals in the figure; they are in the tables. Test images per source: {{LosoImgCohenN}} (Cohen), {{LosoImgKermanyN}} (Kermany), {{LosoImgNIHN}} (NIH), {{LosoImgShenzhenN}} (Shenzhen) and {{LosoImgTBXkN}} (TBX11K).) {#fig:loso}

Table: Linear-head leave-one-source-out, ImageNet features (primary). Macro-F1 is in percent over the evaluated classes; the gap is the reference minus the leave-one-source-out value, in percentage points. The binary column is the ROC-AUC of 1 − P(Normal) for abnormal against normal and is not computed for Cohen (fewer than 20 normal test images). {#tab:losoimg}

| Held-out source | Test images | Evaluated classes | LOSO macro-F1 (95% interval) | Reference macro-F1 | Gap (95% interval) | Binary ROC-AUC (95% interval) |
|---|---|---|---|---|---|---|
| Cohen | {{LosoImgCohenN}} | COVID-19, Pneumonia | {{LosoImgCohenMacroFOne}} ({{LosoImgCohenMacroFOneLo}} to {{LosoImgCohenMacroFOneHi}}) | {{LosoImgCohenRefMacroFOne}} | {{LosoImgCohenGap}} ({{LosoImgCohenGapLo}} to {{LosoImgCohenGapHi}}) | n/a |
| Kermany | {{LosoImgKermanyN}} | Normal, Pneumonia | {{LosoImgKermanyMacroFOne}} ({{LosoImgKermanyMacroFOneLo}} to {{LosoImgKermanyMacroFOneHi}}) | {{LosoImgKermanyRefMacroFOne}} | {{LosoImgKermanyGap}} ({{LosoImgKermanyGapLo}} to {{LosoImgKermanyGapHi}}) | {{LosoImgKermanyBinAUC}} ({{LosoImgKermanyBinAUCLo}} to {{LosoImgKermanyBinAUCHi}}) |
| NIH ChestX-ray14 | {{LosoImgNIHN}} | Normal, Pneumonia | {{LosoImgNIHMacroFOne}} ({{LosoImgNIHMacroFOneLo}} to {{LosoImgNIHMacroFOneHi}}) | {{LosoImgNIHRefMacroFOne}} | {{LosoImgNIHGap}} ({{LosoImgNIHGapLo}} to {{LosoImgNIHGapHi}}) | {{LosoImgNIHBinAUC}} ({{LosoImgNIHBinAUCLo}} to {{LosoImgNIHBinAUCHi}}) |
| Shenzhen | {{LosoImgShenzhenN}} | Normal, Tuberculosis | {{LosoImgShenzhenMacroFOne}} ({{LosoImgShenzhenMacroFOneLo}} to {{LosoImgShenzhenMacroFOneHi}}) | {{LosoImgShenzhenRefMacroFOne}} | {{LosoImgShenzhenGap}} ({{LosoImgShenzhenGapLo}} to {{LosoImgShenzhenGapHi}}) | {{LosoImgShenzhenBinAUC}} ({{LosoImgShenzhenBinAUCLo}} to {{LosoImgShenzhenBinAUCHi}}) |
| TBX11K | {{LosoImgTBXkN}} | Normal, Tuberculosis | {{LosoImgTBXkMacroFOne}} ({{LosoImgTBXkMacroFOneLo}} to {{LosoImgTBXkMacroFOneHi}}) | {{LosoImgTBXkRefMacroFOne}} | {{LosoImgTBXkGap}} ({{LosoImgTBXkGapLo}} to {{LosoImgTBXkGapHi}}) | {{LosoImgTBXkBinAUC}} ({{LosoImgTBXkBinAUCLo}} to {{LosoImgTBXkBinAUCHi}}) |

Table: Linear-head leave-one-source-out on frozen features of the run-2 backbone, which was trained on all six sources (secondary; not held-out and not external). Columns as in [[tab:losoimg]]. {#tab:losomdl}

| Held-out source | Test images | Evaluated classes | LOSO macro-F1 (95% interval) | Reference macro-F1 | Gap (95% interval) | Binary ROC-AUC (95% interval) |
|---|---|---|---|---|---|---|
| Cohen | {{LosoMdlCohenN}} | COVID-19, Pneumonia | {{LosoMdlCohenMacroFOne}} ({{LosoMdlCohenMacroFOneLo}} to {{LosoMdlCohenMacroFOneHi}}) | {{LosoMdlCohenRefMacroFOne}} | {{LosoMdlCohenGap}} ({{LosoMdlCohenGapLo}} to {{LosoMdlCohenGapHi}}) | n/a |
| Kermany | {{LosoMdlKermanyN}} | Normal, Pneumonia | {{LosoMdlKermanyMacroFOne}} ({{LosoMdlKermanyMacroFOneLo}} to {{LosoMdlKermanyMacroFOneHi}}) | {{LosoMdlKermanyRefMacroFOne}} | {{LosoMdlKermanyGap}} ({{LosoMdlKermanyGapLo}} to {{LosoMdlKermanyGapHi}}) | {{LosoMdlKermanyBinAUC}} ({{LosoMdlKermanyBinAUCLo}} to {{LosoMdlKermanyBinAUCHi}}) |
| NIH ChestX-ray14 | {{LosoMdlNIHN}} | Normal, Pneumonia | {{LosoMdlNIHMacroFOne}} ({{LosoMdlNIHMacroFOneLo}} to {{LosoMdlNIHMacroFOneHi}}) | {{LosoMdlNIHRefMacroFOne}} | {{LosoMdlNIHGap}} ({{LosoMdlNIHGapLo}} to {{LosoMdlNIHGapHi}}) | {{LosoMdlNIHBinAUC}} ({{LosoMdlNIHBinAUCLo}} to {{LosoMdlNIHBinAUCHi}}) |
| Shenzhen | {{LosoMdlShenzhenN}} | Normal, Tuberculosis | {{LosoMdlShenzhenMacroFOne}} ({{LosoMdlShenzhenMacroFOneLo}} to {{LosoMdlShenzhenMacroFOneHi}}) | {{LosoMdlShenzhenRefMacroFOne}} | {{LosoMdlShenzhenGap}} ({{LosoMdlShenzhenGapLo}} to {{LosoMdlShenzhenGapHi}}) | {{LosoMdlShenzhenBinAUC}} ({{LosoMdlShenzhenBinAUCLo}} to {{LosoMdlShenzhenBinAUCHi}}) |
| TBX11K | {{LosoMdlTBXkN}} | Normal, Tuberculosis | {{LosoMdlTBXkMacroFOne}} ({{LosoMdlTBXkMacroFOneLo}} to {{LosoMdlTBXkMacroFOneHi}}) | {{LosoMdlTBXkRefMacroFOne}} | {{LosoMdlTBXkGap}} ({{LosoMdlTBXkGapLo}} to {{LosoMdlTBXkGapHi}}) | {{LosoMdlTBXkBinAUC}} ({{LosoMdlTBXkBinAUCLo}} to {{LosoMdlTBXkBinAUCHi}}) |

With ImageNet features the head trained on five sources scores far below its in-distribution reference on four of the five held-out sources: the gap is {{LosoImgTBXkGap}} points for TBX11K, {{LosoImgShenzhenGap}} for Shenzhen, {{LosoImgKermanyGap}} for Kermany and {{LosoImgCohenGap}} for Cohen. The two tuberculosis sources have binary ROC-AUCs of only {{LosoImgShenzhenBinAUC}} (Shenzhen) and {{LosoImgTBXkBinAUC}} (TBX11K), whereas the Kermany value is {{LosoImgKermanyBinAUC}}. The trained backbone gives gaps of the same order ({{LosoMdlTBXkGap}} points for TBX11K, {{LosoMdlShenzhenGap}} for Shenzhen). NIH shows a negative gap ({{LosoImgNIHGap}} points, interval {{LosoImgNIHGapLo}} to {{LosoImgNIHGapHi}}). This is a feature of the metric and not better generalisation: only two NIH classes can be evaluated, the reference is low ({{LosoImgNIHRefMacroFOne}}%) because the six-class head also predicts Pleural Effusion and Nodule / Mass, which count as errors for the two evaluated classes, and the absolute held-out value is low as well ({{LosoImgNIHMacroFOne}}%). Cohen has only {{LosoImgCohenN}} test images in its evaluated classes, and its reference ({{LosoImgCohenRefMacroFOne}}%) is also low, so it is very uncertain. The balanced accuracy of a two-class source can fall below 50% (TBX11K {{LosoImgTBXkBalAcc}}%) because predictions of the other classes are counted as errors; it is not a result worse than chance on a binary question, which the binary AUC addresses.

The head trained on all six sources and scored once on Montgomery gives a sensitivity of {{LosoImgExtSens}}% and a specificity of {{LosoImgExtSpec}}% for ImageNet features, and {{LosoMdlExtSens}}% and {{LosoMdlExtSpec}}% for the trained backbone.

### Band-stop screening and the go/no-go decision

The decision rule of Chapter 4 was applied to the validation numbers. No band satisfied criterion 1 (the band with the largest source drop is band {{BsBand}}, and the number of bands satisfying the criterion is {{BsNBandsSatisfying}}), so the random control was not run. [[tab:c2crit]] gives the outcome and [[tab:c2lite]] the screening results for band {{BsBand}}, with a training sample of {{BsLiteValNTrain}} images.

Table: Go/no-go criteria for a band-suppression retraining (analysis C2) and their outcome. {#tab:c2crit}

| Criterion | Outcome |
|---|---|
| 1. A band with ΔS of at least 5 points (lower interval bound above zero) and ΔD of at most 2 points (upper bound below 3) | {{BsConditionOne}} |
| 2. A matched-energy random control removes less source information than the band | {{BsConditionTwo}} (control not run, because no band met criterion 1) |
| 3. Screening: source accuracy falls by at least 5 points with a loss of at most 2 points of within-source F1 (validation) | {{BsConditionThree}} |
| 4. Student approves one night of CPU time | {{BsConditionFour}} (moot while criteria 1 to 3 fail) |
| Decision | {{BsDecision}} |

Table: Band-stop screening for band {{BsBand}} (exploratory; ImageNet features; linear heads; percentage points, full image minus band-stop image; 95% bootstrap intervals over scored images). The validation row decides; the test row was computed once. {#tab:c2lite}

| Split | ΔS (source BA) | ΔD (within-source F1) |
|---|---|---|
| Validation | {{BsLiteValDeltaS}} ({{BsLiteValDeltaSLo}} to {{BsLiteValDeltaSHi}}) | {{BsLiteValDeltaD}} ({{BsLiteValDeltaDLo}} to {{BsLiteValDeltaDHi}}) |
| Test (exploratory) | {{BsLiteTestDeltaS}} ({{BsLiteTestDeltaSLo}} to {{BsLiteTestDeltaSHi}}) | {{BsLiteTestDeltaD}} ({{BsLiteTestDeltaDLo}} to {{BsLiteTestDeltaDHi}}) |

Removing the best single band lowered the source accuracy of a linear head by {{BsLiteValDeltaS}} points on validation, below the 5-point threshold. The validation disease change was not distinguishable from zero, but the upper bound of the selection-step disease difference ({{SpecMainValStopFDeltaDHi}} points) also exceeded the 3-point limit of criterion 1. The decision is {{BsDecision}}. The band-suppression retraining was therefore not started, no network was trained on band-stopped images, and no number for it exists. This is a negative result of the band-stop idea within this protocol, for frozen-feature linear heads only. It is not evidence that source information is unimportant.

## Interpretation

### Internal results

The internal test accuracy of {{TestAcc}}% hides large differences between classes. COVID-19 (F1 {{FOneCovid}}%), Tuberculosis ({{FOneTB}}%) and Pneumonia ({{FOnePneumonia}}%) are the strongest classes. Pulmonary Nodule / Mass (F1 {{FOneNodule}}%) is the weakest, followed by Normal ({{FOneNormal}}%). The confusion matrix shows that the main confusions involve Nodule / Mass, Normal and Pleural Effusion with one another. Nodule / Mass and Pleural Effusion both come from NIH only and have labels mined from reports, so part of the error may be label noise rather than model error; this was not tested. Accuracy varies a great deal between sources ([[fig:persource]]), from {{SrcAccCohen}}% for the Cohen collection and {{SrcAccNIH}}% for NIH to {{SrcAccCovidRad}}% for the COVID-19 Radiography Database. The sources differ in the classes they contain and in their image style, so these numbers must not be read as a ranking of data quality. The expected calibration error of {{TestECE}} is small, so on the internal test split the top-class score is close to the observed frequency of being right; this does not make it a probability of disease, and it was not checked at another hospital.

### External results

On the unseen hospital the results are much weaker. The model flags {{ExtSensK}} of {{ExtSensN}} tuberculosis scans as abnormal, but it calls only {{ExtSpecK}} of {{ExtSpecN}} normal scans Normal, so most healthy patients of this cohort would be flagged. The exact Tuberculosis recall is {{ExtTBK}} of {{ExtTBN}}, so the abnormal sensitivity does not mean that tuberculosis is recognised. Most scans, normal and tuberculosis alike, were assigned to Pneumonia or Pulmonary Nodule / Mass ([[fig:external]]), and only {{ExtCovidCalls}} scans were called COVID-19. The internal Tuberculosis F1 of {{FOneTB}}% therefore does not transfer to this cohort.

Run 1 called {{RunOneExtCovidCalls}} of {{NExternal}} Montgomery scans COVID-19. With build 2, which adds several sources at once (TBX11K, the COVID-19 Radiography Database and two more NIH archives), run 2 called only {{ExtCovidCalls}} scans COVID-19, but the external errors moved to the other abnormal classes: the internal gains did not translate into better external specificity or into any recognition of Tuberculosis. This is consistent with, but does not prove, a source shortcut in run 1; without an ablation the change cannot be attributed to any single added source. The external gap was not closed.

### Evidence on source shortcuts

Three observations point the same way. The source of an image is predictable from the features, also within a single class (the probe). The external behaviour shows low specificity and no Tuberculosis recognition. The Grad-CAM examples for COVID-19 and Normal fall outside the lung fields. Each piece of evidence is indirect: the probe shows availability, the external result shows failure to generalise, and the heatmaps are qualitative and few. Together they are consistent with a source shortcut, but this report does not state as proven that the model uses source cues.

### Failure analysis

The failures of the final model fall into four groups. First, confusions among Nodule / Mass, Normal and Pleural Effusion on the internal test split, in classes whose labels are mined from reports. Second, low accuracy for some sources inside the test split, for example the Cohen collection. Third, on the external cohort, normal scans assigned to disease classes, which is the failure that matters most for use. Fourth, no tuberculosis scan of the external cohort assigned to Tuberculosis. No failure case was reviewed by a radiologist.
