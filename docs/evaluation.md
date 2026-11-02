# Evaluation

There are two entry points with different evidence levels:

| Entry point | Input | Evidence level |
|---|---|---|
| `python -m lungai.evaluation.evaluate` | A checkpoint plus images, or a saved predictions CSV | **Reproduced**: metrics computed from per-image outputs |
| `python -m lungai.evaluation.historical` | Aggregate result files in `experiments/` | **Recorded**: internal-consistency check only |

At present only the second can run, because there is no checkpoint and no image.

## Canonical evaluation

```bash
python -m lungai.evaluation.evaluate --split val      --checkpoint models/densenet121_frequency_v5.h5
python -m lungai.evaluation.evaluate --split test     --checkpoint models/densenet121_frequency_v5.h5 --final
python -m lungai.evaluation.evaluate --split external --checkpoint models/densenet121_frequency_v5.h5 --final
python -m lungai.evaluation.evaluate --predictions artifacts/evaluation_reports/<run>/predictions_test.csv
```

- `test` and `external` require `--final`. They are reported once and must never drive tuning, thresholds or model choice.
- A checkpoint whose SHA-256 is not canonical is refused, unless `--allow-unverified-checkpoint` is passed. The report then records `checkpoint_is_canonical: false`.
- Any missing image stops the run with an error. Nothing is skipped silently; `n_excluded` is always 0.

**Outputs** go to `artifacts/evaluation_reports/<model_id>_<split>_<UTC>/` (this folder is gitignored because it holds per-image data):

- `predictions_<split>.csv`: `image_id`, `split`, `source_dataset`, `true_label`, `prob_<class_id>` × 6, `pred_index`, `pred_label`, `checkpoint_sha256`.
- `metrics_<split>.json`: the report header (model ID and name, checkpoint file and SHA-256, dataset version, manifest SHA-256, split, timestamp, library versions, class order) plus the results.
- `summary_<split>.md`: a readable summary.

**Input validation.** Before computing any metric, the run fails when:

- columns are missing;
- the file contains more than one split or more than one checkpoint;
- an `image_id` is duplicated;
- `pred_index` differs from the argmax of the stored probabilities;
- the set of `image_id`s differs from the manifest split;
- a true label disagrees with the manifest;
- a label falls outside the class mapping;
- probabilities have the wrong shape, are non-finite, or rows do not sum to 1.

## Metric definitions

Implemented in [`lungai/evaluation/metrics.py`](../lungai/evaluation/metrics.py) and tested in `tests/ml/test_metrics.py`.

| Metric | Definition |
|---|---|
| Prediction | argmax of the 6 softmax outputs |
| Accuracy | correct / N, with a Wilson 95% CI and a percentile-bootstrap CI (2,000 resamples, seed 42) |
| Per-class precision, recall, specificity, F1 | One-vs-rest from the confusion matrix (rows = true, columns = predicted). Recall has a Wilson 95% CI |
| Macro / weighted | Unweighted mean over the 6 classes / mean weighted by true support |
| ROC-AUC, PR-AUC | Per class, one-vs-rest on that class's probability. Macro = unweighted mean (sklearn `multi_class="ovr"`, `average="macro"`). Undefined (null) if a class is absent |
| ECE | 15 equal-width bins on the top-class score: Σ (n_b/N)·\|acc_b − conf_b\| |
| Per source | Accuracy and per-class recall within each `source_dataset` |

### External cohort (Montgomery)

This is a **binary** assessment, separate from the six-class task, and six-class metrics are not reported for it.

| Quantity | Definition |
|---|---|
| Positive | Any predicted class other than Normal |
| Reference positive | A TB-confirmed scan |
| Reported | Sensitivity and specificity, each as numerator/denominator with a Wilson 95% CI; ROC-AUC of P(abnormal) = 1 − P(Normal); exact Tuberculosis recall |

A sensitivity claim is never reported without its specificity.

## Recorded results check

`python -m lungai.evaluation.historical` writes `artifacts/evaluation_reports/historical_record_verification.{json,md}`. For each of the five recorded V5 runs it does the following:

- **Recomputes and compares:** accuracy, macro precision, recall and F1, weighted F1, per-class precision, recall, F1, specificity and support, all from the stored confusion matrix, compared with the recorded values (tolerance 6e-5, since the recorded values are rounded to 4 d.p.).
- **Recomputes the external binary result** from the stored prediction breakdown.
- **Lists the validation evidence** that was available for model selection.
- **Marks as unverifiable:** ROC-AUC, PR-AUC and calibration.

Result on this branch: 5 of 5 runs consistent, 0 mismatches (29 checks per run).

## Limitations

- No per-image predictions exist for any recorded run, so ROC-AUC, PR-AUC, calibration and bootstrap CIs cannot be checked.
- The internal test split is patient-leaked (154 images), so the internal metrics are optimistic.
- Each run is a single seed.
