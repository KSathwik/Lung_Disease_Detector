# experiments/ (historical, V1 to V5)

This folder holds the scripts and result files of the earlier V1 to V5 experiments. It is **not** the current pipeline: the current system is the `lungai/` package, and its results are in `docs/results/current_model.md` and `docs/experiment_results.md`.

The scripts and result files are kept because `python -m lungai.evaluation.historical` (run in CI) reads the V5 result files to verify the recorded history, and `artifacts/` cites them. The written reports of these experiments were superseded and removed on 2026-11-22; they remain available in the git history.

The V5 results must not be read as valid: the V5 dataset had patient leakage between training and test, proxy labels, and an external specificity of 0/80, and its checkpoint is lost. What was wrong and what replaced it is summarised in `docs/experiment_results.md` and in Appendix E of the M.Tech report.
