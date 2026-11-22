# Run selection protocol (fixed before run 2's results were seen)

Written 2026-11-03 ~12:00, while run 2 (`v6r2_densenet121_seed42`, build `v6_build_002`) was still training and before any of its validation, test or external results existed. Run 1 (`v6_densenet121_seed42`, build `v6_build_001`) is the model currently served.

## Why a separate rule is needed

Each run selected its own epoch on its own build's validation split. The two validation splits differ (build 2 adds TBX11K and the COVID-19 Radiography Database and was split independently), so the two runs' validation scores are not comparable. Test and external results may not be used to choose between runs (they are reported once, after selection).

## Common selection set

An image belongs to the common selection set if all of the following hold:

1. it is in the `val` split of build 1 or of build 2;
2. it is not in the `train` or `test` split of either build, by exact file (MD5) and by patient: its patient key (`source_id` + `source_patient_id` where `patient_id_status` is `genuine` or `derived_from_filename`, otherwise the file's SHA-256) does not occur in the `train` or `test` split of either build;
3. its label is the same in both builds where it appears in both.

Montgomery (external) is never included.

## Decision rule

- **Primary:** macro-F1 over the six classes on the common selection set. The run with the higher value is served.
- **Tie or difference < 0.01:** keep run 1 (the model already integrated and verified), unless the macro ROC-AUC difference favours run 2 by ≥ 0.01.
- Reported alongside (not used for the decision): accuracy, macro ROC-AUC, per-class recall, and the same metrics restricted to the sources present in both builds (NIH, Kermany, Shenzhen, Cohen), because run 1 never saw TBX11K or COVID-19 Radiography images and run 2 has an in-distribution advantage on them.
- McNemar's test (paired, exact) on the common set is reported as a descriptive statistic. With one seed per run, no claim is made that one training recipe is better in general.

## What happens to the run not selected

It stays documented (run metadata, evaluation reports) as a second V6 experiment. Its test and external results are reported in the experiment record, labelled as not selected.

Implementation: `lungai/evaluation/compare_runs.py`, output `artifacts/evaluation_reports/run_comparison.json`.

## Follow-up runs (fixed 2026-11-03 ~15:00, before run 2's evaluation and before the comparison)

Approved by the project owner: two more seeds and one preprocessing ablation, all on the build of the selected run, run one after another (8 GB RAM; parallel training is not safe).

| Run | Seed | Change from the selected recipe | Purpose |
|---|---|---|---|
| Run 3 | 43 | none | Seed-to-seed variability |
| Run 4 | 44 | none | Seed-to-seed variability |
| Run 5 | 42 (as run 2) | cached images without Gaussian blur, CLAHE and low-pass (`scripts/data/make_ablation_build.py --variant no_filters`), same manifest and split | Preprocessing ablation |

Same architecture, class weights, optimiser schedule and early-stopping patience (3) as the selected run. The selected run's training may have been ended by its wall-clock limit rather than by its epoch limit; the follow-up runs therefore get a fine-tune epoch limit equal to the number of fine-tune epochs the selected run actually completed, and a wall-clock limit well above the expected duration, so that every run receives the same schedule. Each run is evaluated once on val, test and external. Seed variability is reported as mean, sample SD, min and max over the selected run and runs 3–4 (`lungai/evaluation/aggregate_seeds.py`). The served model remains the one chosen by the rule above; the extra runs do not replace it (choosing the best of several seeds on test or external data would be selection on the evaluation sets).
