# models/

Place the canonical checkpoint here: `densenet121_v6r2_seed42.h5` (model v6.1, SHA-256 `7992b8da10ff97f90207dc730c3163ba968a4d6094e1bff263377612ee0597c6`),
produced by `python -m lungai.training.train_v6` (run `v6r2_densenet121_seed42`).

```bash
python scripts/install_checkpoint.py --from runs/v6r2_densenet121_seed42/model.h5
```

`*.h5` files are gitignored and never committed. See docs/inference.md.
The superseded run-1 checkpoint (v6.0, `densenet121_v6_seed42.h5`) is kept only as history. The previous v5.0 checkpoint (`densenet121_frequency_v5.h5`) is lost; see `historical_checkpoints` in lungai/configs/lungai_densenet121.json.
