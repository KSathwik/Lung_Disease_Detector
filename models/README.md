# models/

Place the canonical checkpoint here: `densenet121_v6_seed42.h5` (SHA-256 `c51ed228455c2aab93b4b0220ee721443c3c0486eea961a242829ce68ee6c0e6`),
produced by `python -m lungai.training.train_v6` (run `v6_densenet121_seed42`).

```bash
python scripts/install_checkpoint.py --from runs/v6_densenet121_seed42/model.h5
```

`*.h5` files are gitignored and never committed. See docs/inference.md.
The previous v5.0 checkpoint (`densenet121_frequency_v5.h5`) is lost; see `historical_checkpoints` in lungai/configs/lungai_densenet121.json.
