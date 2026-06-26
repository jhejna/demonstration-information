# Training & Inference

## 1. Preprocess episodes

Convert raw MCAP episodes to `.npz` cache files (run once before training):

```bash
conda run -n openx python scripts/preprocess_episodes.py \
    --root episode_data \
    --splits train test \
    --workers 8
```

## 2. Train

```bash
conda run -n openx python scripts/train.py \
    --config configs/bc/manav.py:default \
    --path /tmp/train_out \
    --name manav_run
```

Checkpoints are saved every `save_freq` steps under `<path>/<name>/<step>/`.

## 3. Estimate quality (L2 score)

Run after training to score the dataset against the saved checkpoint:

```bash
conda run -n openx python scripts/quality/estimate_quality.py \
    --estimator=l2 \
    --obs_ckpt=/tmp/train_out/manav_run/500 \
    --batch_size=1024 \
    --path=./scores/
```

`--obs_ckpt` takes either the run directory (uses latest checkpoint) or `<run_dir>/<step>` for a specific step. Scores are saved as `./scores/<dataset_name>.pkl`.
