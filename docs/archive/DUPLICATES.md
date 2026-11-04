# Removed byte-identical duplicates (2026-11-04)

Each path below was an exact byte-for-byte copy (same git blob) of the kept path and was removed with `git rm` as step 2 of `docs/REPO_ORGANISATION_PLAN.md`. Recover any of them with `git show <commit>:<path>` from history (everything before the commit that applies this change). Kept copies are the canonical ones.

Caveat: archived documents and archived scripts (for example `docs/archive/THESIS.md` text, `thesis_tooling/*.py`, `historical_audits/*.md`) still mention old image paths in plain text; most of those paths were already stale (they point to an older layout such as `docs/screenshots/...`). Use this table to find the surviving copy. Nothing outside `docs/archive` and `experiments/archive` reads any removed path.

Total: 70 files removed, 17.5 MB.

| Removed path | Kept path |
|---|---|
| `docs/archive/THESIS.md` | `docs/archive/thesis_drafts/LungAI_MTech_Project_Report.md` |
| `docs/archive/figures_and_screenshots/confusion_matrix_ResNet.png` | `experiments/archive/legacy_reports/confusion_matrix.png` |
| `docs/archive/figures_and_screenshots/diagrams/rendered/dataset_training_pipeline.png` | `docs/archive/submission_package/figures/Figure_2_Dataset_Pipeline.png` |
| `docs/archive/figures_and_screenshots/diagrams/rendered/system_architecture.png` | `docs/archive/submission_package/figures/Figure_1_System_Workflow.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_internal_digital_normal.png` | `experiments/results/film_normalization/panel_internal_digital_normal.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_internal_digital_pneumonia.png` | `experiments/results/film_normalization/panel_internal_digital_pneumonia.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_internal_pneumonia.png` | `experiments/results/cropping_samples/panel_internal_pneumonia.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_montgomery_abnormal_0104.png` | `experiments/results/cropping_samples/panel_montgomery_abnormal_0104.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_montgomery_abnormal_0140.png` | `experiments/results/cropping_samples/panel_montgomery_abnormal_0140.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_montgomery_normal_0001.png` | `experiments/results/cropping_samples/panel_montgomery_normal_0001.png` |
| `docs/archive/figures_and_screenshots/figures/dataset/panel_montgomery_normal_0020.png` | `experiments/results/cropping_samples/panel_montgomery_normal_0020.png` |
| `docs/archive/figures_and_screenshots/figures/failure_analysis/phase4a_confidence_analysis.png` | `experiments/results/phase4a_confidence_analysis.png` |
| `docs/archive/figures_and_screenshots/figures/failure_analysis/phase4a_feature_space.png` | `experiments/results/phase4a_feature_space.png` |
| `docs/archive/figures_and_screenshots/figures/failure_analysis/phase4a_image_distribution_analysis.png` | `experiments/results/phase4a_image_distribution_analysis.png` |
| `docs/archive/figures_and_screenshots/figures/failure_analysis/phase4a_montgomery_confusion_matrix.png` | `experiments/results/phase4a_montgomery_confusion_matrix.png` |
| `docs/archive/figures_and_screenshots/figures/model/coral_tsne_feature_space.png` | `experiments/densenet_coral_v5/coral_feature_space.png` |
| `docs/archive/figures_and_screenshots/figures/model/dann_tsne_feature_space.png` | `experiments/densenet_dann_v5/dann_feature_space.png` |
| `docs/archive/figures_and_screenshots/figures/model/hybrid_tsne_feature_space.png` | `experiments/densenet_hybrid_v5/hybrid_feature_space.png` |
| `docs/archive/figures_and_screenshots/figures/model/model_d_tsne_feature_space.png` | `experiments/densenet_frequency_v5/frequency_feature_space.png` |
| `docs/archive/figures_and_screenshots/figures/performance/coral_confusion_matrix.png` | `experiments/densenet_coral_v5/coral_confusion_matrix.png` |
| `docs/archive/figures_and_screenshots/figures/performance/coral_roc_curves.png` | `experiments/densenet_coral_v5/coral_roc_curves.png` |
| `docs/archive/figures_and_screenshots/figures/performance/dann_confusion_matrix.png` | `experiments/densenet_dann_v5/dann_confusion_matrix.png` |
| `docs/archive/figures_and_screenshots/figures/performance/dann_roc_curves.png` | `experiments/densenet_dann_v5/dann_roc_curves.png` |
| `docs/archive/figures_and_screenshots/figures/performance/hybrid_confusion_matrix.png` | `experiments/densenet_hybrid_v5/hybrid_confusion_matrix.png` |
| `docs/archive/figures_and_screenshots/figures/performance/hybrid_roc_curves.png` | `experiments/densenet_hybrid_v5/hybrid_roc_curves.png` |
| `docs/archive/figures_and_screenshots/figures/performance/model_d_confusion_matrix.png` | `experiments/densenet_frequency_v5/frequency_confusion_matrix.png` |
| `docs/archive/figures_and_screenshots/figures/performance/model_d_pr_curves.png` | `experiments/densenet_frequency_v5/frequency_pr_curves.png` |
| `docs/archive/figures_and_screenshots/figures/performance/model_d_roc_curves.png` | `experiments/densenet_frequency_v5/frequency_roc_curves.png` |
| `docs/archive/figures_and_screenshots/images/01_landing_analyze_page.png` | `docs/archive/figures_and_screenshots/screenshots/application/02_analyze.png` |
| `docs/archive/figures_and_screenshots/images/02_patient_management.png` | `docs/archive/figures_and_screenshots/screenshots/application/07_patients.png` |
| `docs/archive/figures_and_screenshots/images/03_model_metrics.png` | `docs/archive/figures_and_screenshots/screenshots/application/05_metrics.png` |
| `docs/archive/figures_and_screenshots/images/04_history_records.png` | `docs/archive/figures_and_screenshots/screenshots/application/06_history.png` |
| `docs/archive/figures_and_screenshots/screenshots/analysis/prediction_result_tb.png` | `docs/archive/figures_and_screenshots/screenshots/application/03_prediction.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/01_home.png` | `docs/archive/figures_and_screenshots/screenshots/application/01_home.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/02_analyze_empty.png` | `docs/archive/figures_and_screenshots/screenshots/application/02_analyze.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/03_prediction.png` | `docs/archive/figures_and_screenshots/screenshots/application/03_prediction.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/04_gradcam.png` | `docs/archive/figures_and_screenshots/screenshots/application/04_gradcam.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/05_metrics.png` | `docs/archive/figures_and_screenshots/screenshots/application/05_metrics.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/06_patients.png` | `docs/archive/figures_and_screenshots/screenshots/application/07_patients.png` |
| `docs/archive/figures_and_screenshots/screenshots/e2e/07_history.png` | `docs/archive/figures_and_screenshots/screenshots/application/06_history.png` |
| `docs/archive/figures_and_screenshots/screenshots/explanation/gradcam_overlay_tb.png` | `docs/archive/figures_and_screenshots/screenshots/application/04_gradcam.png` |
| `docs/archive/figures_and_screenshots/training_curves.png` | `experiments/archive/legacy_reports/training_history.png` |
| `docs/archive/historical_audits/CXR_INPUT_VALIDATION_AUDIT.md` | `docs/archive/historical_audits/CXR_GATE_FINAL_FORENSIC_AUDIT.md` |
| `docs/archive/submission_package/figures/Figure_10A_Montgomery_CM.png` | `experiments/results/phase4a_montgomery_confusion_matrix.png` |
| `docs/archive/submission_package/figures/Figure_10B_Feature_Space.png` | `experiments/results/phase4a_feature_space.png` |
| `docs/archive/submission_package/figures/Figure_4_Training_Curves.png` | `experiments/results/densenet_v5_training_curves.png` |
| `docs/archive/submission_package/figures/Figure_6_Confusion_Matrix.png` | `experiments/results/densenet_v5_confusion_matrix.png` |
| `docs/archive/submission_package/figures/Figure_7_ROC_Curves.png` | `experiments/results/densenet_v5_roc_curves.png` |
| `docs/archive/submission_package/figures/Figure_8_PR_Curves.png` | `experiments/densenet_frequency_v5/frequency_pr_curves.png` |
| `docs/archive/thesis_drafts/LungAI_Updated_Professional_Presentation_root_copy.pptx` | `docs/archive/thesis_drafts/LungAI_Updated_Professional_Presentation.pptx` |
| `docs/archive/thesis_drafts/papers/CRediT_Author_Statement.md` | `docs/archive/submission_package/CRediT_Author_Statement.md` |
| `docs/archive/thesis_drafts/papers/Cover_Letter.md` | `docs/archive/submission_package/Cover_Letter.md` |
| `docs/archive/thesis_drafts/papers/Cover_Letter_Elsevier.docx` | `docs/archive/submission_package/Cover_Letter_Elsevier.docx` |
| `docs/archive/thesis_drafts/papers/Data_and_Code_Availability_Statement.md` | `docs/archive/submission_package/Data_and_Code_Availability_Statement.md` |
| `docs/archive/thesis_drafts/papers/Declaration_of_Competing_Interests.md` | `docs/archive/submission_package/Declaration_of_Competing_Interests.md` |
| `docs/archive/thesis_drafts/papers/Highlights.txt` | `docs/archive/submission_package/Highlights.txt` |
| `docs/archive/thesis_drafts/papers/JOURNAL_PAPER_BSPC.md` | `docs/archive/thesis_drafts/MANUSCRIPT_BSPC_LUNGAI.md` |
| `docs/archive/thesis_drafts/papers/Manuscript_BSPC_Elsevier.docx` | `docs/archive/submission_package/Manuscript_BSPC_Elsevier.docx` |
| `docs/archive/thesis_drafts/papers/Suggested_Reviewers.md` | `docs/archive/submission_package/Suggested_Reviewers.md` |
| `docs/archive/thesis_drafts/papers/figures/Figure_10A_Montgomery_CM.png` | `experiments/results/phase4a_montgomery_confusion_matrix.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_10B_Feature_Space.png` | `experiments/results/phase4a_feature_space.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_11_V3_vs_V5_Delta.png` | `docs/archive/submission_package/figures/Figure_11_V3_vs_V5_Delta.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_1_System_Workflow.png` | `docs/archive/submission_package/figures/Figure_1_System_Workflow.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_2_Dataset_Pipeline.png` | `docs/archive/submission_package/figures/Figure_2_Dataset_Pipeline.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_3_Source_Distribution.png` | `docs/archive/submission_package/figures/Figure_3_Source_Distribution.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_4_Training_Curves.png` | `experiments/results/densenet_v5_training_curves.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_6_Confusion_Matrix.png` | `experiments/results/densenet_v5_confusion_matrix.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_7_ROC_Curves.png` | `experiments/results/densenet_v5_roc_curves.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_8_PR_Curves.png` | `experiments/densenet_frequency_v5/frequency_pr_curves.png` |
| `docs/archive/thesis_drafts/papers/figures/Figure_9_Per_Class_Performance.png` | `docs/archive/submission_package/figures/Figure_9_Per_Class_Performance.png` |
