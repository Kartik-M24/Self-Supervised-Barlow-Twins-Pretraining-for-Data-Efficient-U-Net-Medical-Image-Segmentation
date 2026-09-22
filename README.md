# Semi-Supervised Segmentation: BT-UNet vs SimSiam-UNet vs Supervised UNet

Three notebooks that answer one question: does self-supervised pretraining help when you only have a few labelled masks? Each trains the same U-Net on 5%, 10%, 20% and 50% of the labels, under 5-fold cross-validation, and reports mean ± std on a held-out test set.

| Notebook | Pretraining | Encoder starts from |
|---|---|---|
| `BT-Unet_clean_kfold.ipynb` | Barlow Twins | SSL weights |
| `SimSiam-Unet_GPU_Multiple_datasets_kfold_ensemble.ipynb` | SimSiam | SSL weights |
| `UNet_baseline_Multiple_datasets_kfold.ipynb` | none | random init |

The architecture, data loading, metrics, losses, K-fold split and ensembling code are identical across all three. Only the encoder initialisation differs, so any gap in the results comes from pretraining rather than from a change in the model.

## Requirements

```
tensorflow>=2.16   # Keras 3
scikit-learn
scikit-image
pandas
numpy
matplotlib
tqdm
hausdorff
```

A GPU is assumed. A full sweep is 4 fractions × 5 folds = 20 fine-tuning runs per notebook, each up to 200 epochs with early stopping.

## Directory layout

Create this before the first run:

```
project/
├── datasets/
│   ├── ISIC2018/
│   │   ├── train/{org,gt}/
│   │   ├── val/{org,gt}/
│   │   └── test/{org,gt}/
│   └── KDSB/
│       ├── train/{org,gt}/
│       └── test/{org,gt}/
├── saved_models/     # created automatically
├── saved_np/         # created automatically
├── logs/             # created automatically
└── results/          # created automatically
```

`org` holds images, `gt` holds masks. Mask filenames are derived from image filenames by a lambda in `DATASET_CONFIGS`:

- ISIC2018: `ISIC_0000000.jpg` to `ISIC_0000000_segmentation.png`
- KDSB: `image.png` to `image_GT.png`

To add a dataset, add an entry to `DATASET_CONFIGS` in the config cell of all three notebooks.

### Validation splits

`HAS_VAL` is set by checking whether `datasets/{NAME}/val/org/` exists. ISIC2018 has one and it gets used directly. KDSB does not, so validation is carved out of the labelled subset using `val_split = 0.3`. All three notebooks handle both cases the same way, so the two datasets stay comparable within a notebook but validation numbers are not comparable across datasets.

## Configuration

Everything lives in the first code cell.

| Setting | Default | Notes |
|---|---|---|
| `DATASET_NAME` | `"ISIC2018"` | must match a `DATASET_CONFIGS` key |
| `LABEL_FRACTIONS` | `[0.05, 0.1, 0.2, 0.5]` | fractions of labelled training data |
| `USE_KFOLD` | `True` | see below |
| `N_FOLDS` | `5` | folds per fraction |
| `CV_RANDOM_STATE` | `42` | keep fixed across notebooks so folds match |
| `PRETRAIN_FLG` | `False` | `True` runs SSL, `False` loads saved encoder |
| `TRAIN_FLG` | `True` | `False` loads saved fine-tuned weights |
| `TEST_MODE` | `False` | smoke test: 1 fraction, 1 epoch, 10 samples, 2 folds |
| `FINETUNE_EPOCHS` | `200` | early stopping patience 20 |

`USE_KFOLD` picks one of two mutually exclusive paths. With `True` the single-split section is skipped and cross-validation plus ensembling runs. With `False` the reverse. Nothing trains twice either way.

Set `CV_RANDOM_STATE` and `LABEL_FRACTIONS` identically in all three notebooks or the comparison is meaningless.

### GPU assignment

Each notebook pins itself to one device, and the pin must execute before TensorFlow is imported, so it sits at the top of the imports cell:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

Current assignment: BT-UNet on `1`, SimSiam on `0`, baseline on `0`. Change the baseline before running it alongside SimSiam, or the two will contend for the same card.

## Running

Smoke test first. Set `TEST_MODE = True`, run all cells, confirm the pipeline completes in a few minutes, then set it back to `False`.

### Order

1. **Pretrain the encoders.** In BT-UNet and SimSiam, set `PRETRAIN_FLG = True` and run through the pretraining section. Each saves an encoder to `saved_models/`:
   - `{DATASET_NAME}_barlow_twins_encoder.keras`
   - `{DATASET_NAME}_simsiam_encoder.keras`

   Pretraining uses only images, no masks, so it runs once per dataset and is reused by every label fraction. Set `PRETRAIN_FLG = False` afterwards to avoid repeating it.

2. **Fine-tune.** With `TRAIN_FLG = True` and `USE_KFOLD = True`, run the rest of the notebook. This trains 20 models, saves each fold, then averages the folds for a given fraction into one ensemble model.

3. **Run the baseline** the same way. It has no pretraining section, so it goes straight from data loading to training.

First run of a dataset caches resized arrays to `saved_np/` as `.npy`. Later runs load from cache. Delete those files if you change `IMG_HEIGHT` or `IMG_WIDTH`.

### Outputs

Models in `saved_models/`:

```
model_{DATASET}_BT-UNet_{pct}pct_fold{i}.keras
model_{DATASET}_BT-UNet_{pct}pct_ensemble.keras
model_{DATASET}_SimSiam-UNET_{pct}pct_fold{i}.keras
model_{DATASET}_UNet_{pct}pct_fold{i}.keras
```

CSVs in `results/`, one namespace per notebook so nothing gets overwritten:

| | BT-UNet | SimSiam | Baseline |
|---|---|---|---|
| per-fold | `{DS}_cv_results.csv` | `{DS}_simsiam_cv_results.csv` | `{DS}_baseline_cv_results.csv` |
| mean ± std | `{DS}_cv_summary.csv` | `{DS}_simsiam_cv_summary.csv` | `{DS}_baseline_cv_summary.csv` |
| ensemble | `{DS}_ensemble_results.csv` | `{DS}_simsiam_ensemble_results.csv` | `{DS}_baseline_ensemble_results.csv` |
| single-split | `{DS}_results.csv` | `{DS}_simsiam_results.csv` | `{DS}_baseline_results.csv` |

BT-UNet is the unprefixed default. Every row carries a `Dataset` column, so you can concatenate across datasets and group by it.

## Comparing results

The number to compare is `Test Dice` from the `_cv_summary.csv` files, which gives mean and std across the 5 folds for each fraction. Test metrics come from the external test set, untouched by any fold split.

```python
import pandas as pd
DS = "ISIC2018"
frames = {
    "BT-UNet":  f"results/{DS}_cv_results.csv",
    "SimSiam":  f"results/{DS}_simsiam_cv_results.csv",
    "Baseline": f"results/{DS}_baseline_cv_results.csv",
}
df = pd.concat([pd.read_csv(p).assign(Model=m) for m, p in frames.items()])
print(df.groupby(["Model", "Label Fraction (%)"])["Test Dice"].agg(["mean", "std"]))
```

Expect the gap between pretrained and baseline to be widest at 5% and to shrink as the fraction grows. That shrinking gap is the result the experiment is designed to show.

The ensemble CSVs report one row per fraction, averaging the 5 fold models into a single predictor. Compare those against the fold mean in the same notebook to see what ensembling buys you.

## Reusing trained models

Set `TRAIN_FLG = False` to skip training and load saved weights. The filenames must match the current naming scheme exactly. If you have models from an earlier version saved as `model_SimSiam-UNET_5_fold0.keras`, rename them:

```python
import os, re
SAVE_DIR, DATASET_NAME = "saved_models", "ISIC2018"
for f in os.listdir(SAVE_DIR):
    m = re.fullmatch(r"model_SimSiam-UNET_(\d+)_(fold\d+|ensemble)\.keras", f)
    if m:
        new = f"model_{DATASET_NAME}_SimSiam-UNET_{m.group(1)}pct_{m.group(2)}.keras"
        os.rename(os.path.join(SAVE_DIR, f), os.path.join(SAVE_DIR, new))
```

If `model.load_weights()` complains about layer names after renaming, swap that call for `tf.keras.models.load_model(path, compile=False)`, which reads structure from the file instead of matching against a freshly built model.

## Troubleshooting

**`FileNotFoundError` on the encoder.** Pretraining has not run for this dataset, or `DATASET_NAME` changed. Set `PRETRAIN_FLG = True` once.

**Results table is empty.** `USE_KFOLD = True` fills `cv_results`, not `all_results`. Read the K-fold summary cell instead of the single-split table.

**OOM during K-fold.** Each fold calls `tf.keras.backend.clear_session()` and deletes the model, but the ensemble step loads all 5 fold models at once. Lower `FINETUNE_BATCH` or `N_FOLDS`.

**Fewer folds than requested.** `n_folds_this = min(N_FOLDS, X_subset.shape[0])`. At 5% of a small dataset there may be fewer samples than folds.

**Two notebooks fighting for a GPU.** Check `CUDA_VISIBLE_DEVICES` at the top of the imports cell in each.

**Visualisation cell loads nothing.** It looks for the 50% ensemble and falls back to the 50% single-split model. Neither exists until a full run finishes.
