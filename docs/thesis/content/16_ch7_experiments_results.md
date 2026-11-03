# Experimental Setup and Results

## Experimental setup

### Hardware and software

All training and evaluation were run on one laptop with an 8 GB memory and a CPU only, under Windows 11, with Python 3.12, TensorFlow 2.16.1, Keras 3.15.1, NumPy 1.26.4 and OpenCV 4.10. TensorFlow 2.16 has no GPU support on native Windows, so no accelerator was used. Library versions, the code commit and the checkpoint hash are recorded in the metadata of each run.

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
| Seed | {{Seed}} (one run) |
| Epochs trained | {{EpochsTrained}}; selected epoch {{SelectedEpoch}} |

### Evaluation protocol

The validation split is evaluated after every epoch and is used to choose the checkpoint. The test split and the external cohort are evaluated once, with the final flag, after the checkpoint is chosen. Confidence intervals for accuracy and recall are Wilson intervals; the bootstrap interval for accuracy uses 2,000 resamples with seed 42. Calibration uses 15 bins. The definitions are in Chapter 4.

### What is not done

No hyper-parameter search was run. Run-to-run variation is measured only by the follow-up seed runs reported in the section on seed variability [PENDING: seed runs]. No threshold was tuned on the test split or the external cohort. No calibration method was fitted. No comparison with other published systems is made.

## Training behaviour

[[fig:curves]] shows the training loss and the validation accuracy, macro-F1 and macro ROC-AUC per epoch. The checkpoint was selected at epoch {{SelectedEpoch}} out of {{EpochsTrained}}.

![fig_training_curves](Training loss and validation accuracy, macro-F1 and macro ROC-AUC per epoch.) {#fig:curves}

The run completed its full schedule of {{EpochsTrained}} epochs (the head epochs followed by fine-tuning) and was not stopped early. Validation macro-F1 rose slowly during fine-tuning and was highest at the last epoch ({{SelectedEpoch}}), where it was {{ValMacroFOne}}%. The validation curve was therefore still rising slightly when training ended, and a longer schedule might change the result; this was not tested

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

[PENDING: seed runs] Two further seeds of the selected recipe are being trained. The mean, sample standard deviation, minimum and maximum of the main metrics over the three seeds will be reported here.

## Preprocessing ablation

[PENDING: ablation] One run on images without the Gaussian blur, CLAHE and low-pass filter, with the same manifest, split and seed, will be compared here with the final model. Until it is complete, this report makes no claim that the preprocessing helps.

## Interpretation

### Internal results

The internal test accuracy of {{TestAcc}}% hides large differences between classes. COVID-19 (F1 {{FOneCovid}}%), Tuberculosis ({{FOneTB}}%) and Pneumonia ({{FOnePneumonia}}%) are the strongest classes. Pulmonary Nodule / Mass (F1 {{FOneNodule}}%) is the weakest, followed by Normal ({{FOneNormal}}%). The confusion matrix shows that the main confusions involve Nodule / Mass, Normal and Pleural Effusion with one another. Nodule / Mass and Pleural Effusion both come from NIH only and have labels mined from reports, so part of the error may be label noise and not model error; this was not tested. Accuracy varies a great deal between sources ([[fig:persource]]), from {{SrcAccCohen}}% for the Cohen collection and {{SrcAccNIH}}% for NIH to {{SrcAccCovidRad}}% for the COVID-19 Radiography Database. The sources differ in the classes they contain and in their image style, so these numbers must not be read as a ranking of data quality. The expected calibration error of {{TestECE}} is small, so on the internal test split the top-class score is close to the observed frequency of being right; this does not make it a probability of disease, and it was not checked at another hospital.

### External results

On the unseen hospital the picture is different. The model flags {{ExtSensK}} of {{ExtSensN}} tuberculosis scans as abnormal, but it calls only {{ExtSpecK}} of {{ExtSpecN}} normal scans Normal, so most healthy patients of this cohort would be flagged. The exact Tuberculosis recall is {{ExtTBK}} of {{ExtTBN}}: the abnormal sensitivity does not mean that tuberculosis is recognised. Most scans, normal and tuberculosis alike, were assigned to Pneumonia or Pulmonary Nodule / Mass ([[fig:external]]), and only {{ExtCovidCalls}} scans were called COVID-19. The internal Tuberculosis F1 of {{FOneTB}}% therefore does not transfer to this cohort.

Run 1 called {{RunOneExtCovidCalls}} of {{NExternal}} Montgomery scans COVID-19. Adding a second large COVID-19 source in run 2 removed that behaviour, but the external errors moved to the other abnormal classes: the internal gains did not translate into better external specificity or into any recognition of Tuberculosis. Adding sources changed which error appears; it did not close the external gap.

### Evidence on source shortcuts

Three observations point the same way. The source of an image is predictable from the features, also within a single class (the probe). The external behaviour shows low specificity and no Tuberculosis recognition. The Grad-CAM examples for COVID-19 and Normal fall outside the lung fields. Each piece of evidence is indirect: the probe shows availability, the external result shows failure to generalise, and the heatmaps are qualitative and few. Together they are consistent with a source shortcut, but this report does not state as proven that the model uses source cues.

### Failure analysis

The failures of the final model fall into four groups. First, confusions among Nodule / Mass, Normal and Pleural Effusion on the internal test split, in classes whose labels are mined from reports. Second, low accuracy for some sources inside the test split, for example the Cohen collection. Third, on the external cohort, normal scans assigned to disease classes, which is the failure that matters most for use. Fourth, no tuberculosis scan of the external cohort assigned to Tuberculosis. No failure case was reviewed by a radiologist.
