# Self-Supervised Barlow Twins Pretraining for Data-Efficient U-Net Segmentation

Does self-supervised pretraining help a U-Net segment medical images when only a few masks are labelled? This repository compares four model families on two datasets, ISIC2018 (skin lesions) and KDSB (Kaggle Data Science Bowl 2018 nuclei), across several label fractions under 5-fold cross-validation.

| Family | Notebook | Pretraining | Pretrained part |
|---|---|---|---|
| UNet (baseline) | `UNet-Baseline.ipynb` | none | nothing (random init) |
| BT-UNet | `BT-UNet.ipynb` | Barlow Twins | encoder + bottleneck |
| SimSiam-UNet | `SimSiam-UNet.ipynb` | SimSiam | encoder + bottleneck |
| DBT-UNet | `Dense-BT-UNet.ipynb` | Barlow Twins + dense decoder term | encoder + bottleneck + decoder |

All four share the same U-Net (separable-conv residual blocks, filters 16 to 256), data loading, metrics, BCE+Dice loss, fold split and ensembling code. Only the initialisation differs, so a gap in the results can be attributed to pretraining.

## Repository layout

```
BT Unet/
├── UNet-Baseline.ipynb        # supervised baseline, no pretraining
├── BT-UNet.ipynb              # Barlow Twins encoder pretraining + fine-tuning
├── SimSiam-UNet.ipynb         # SimSiam encoder pretraining + fine-tuning
├── Dense-BT-UNet.ipynb        # DBT-UNet: global + dense Barlow Twins, whole U-Net pretrained
├── Evaluation_Pipeline.ipynb  # re-scores saved models on train / val / test
├── smoke_test.py              # CPU end-to-end check on synthetic data
├── Ablations/                 # hyperparameter ablation scripts and driver (own README)
│   └── Arch/                  # architecture ablation overlay
├── saved_models/              # legacy single-split result CSVs
├── saved_np/                  # cached KDSB arrays (Git LFS)
├── evaluation_results.csv     # legacy evaluation output
└── model_BT-Unet.keras        # legacy BT-UNet checkpoint
```

The `.npy` arrays are stored with Git LFS. After cloning, run `git lfs pull` to fetch them. Otherwise the notebooks rebuild them from `datasets/` on first run.

## Requirements

```
tensorflow>=2.16   # Keras 3
scikit-learn
scikit-image
pandas>=2.2       # tier0_tables.py uses groupby.apply(include_groups=...)
numpy
matplotlib
tqdm
hausdorff
```

A GPU is assumed. A full sweep is 4 fractions × 5 folds = 20 fine-tuning runs per notebook, each up to 200 epochs with early stopping (patience 20 on validation Dice).

## Data

Every notebook uses paths relative to its working directory, so run them from `BT Unet/` and put the data there:

```
BT Unet/
├── datasets/
│   ├── ISIC2018/
│   │   ├── train/{org,gt}/
│   │   ├── val/{org,gt}/
│   │   └── test/{org,gt}/
│   └── KDSB/
│       ├── train/{org,gt}/
│       └── test/{org,gt}/
├── saved_models/   # created automatically
├── saved_np/       # created automatically
├── logs/           # created automatically
└── results/        # created automatically
```

`org` holds images and `gt` holds masks. `DATASET_CONFIGS` in each notebook's config cell maps an image filename to its mask filename:

- ISIC2018: `ISIC_0000000.jpg` → `ISIC_0000000_segmentation.png`
- KDSB: mask has the same filename as the image

To add a dataset, add an entry to `DATASET_CONFIGS` in every notebook you plan to run.

The first run for a dataset caches resized arrays to `saved_np/{DATASET}_{X|Y}_{split}_256x256.npy`, and later runs load the cache. If you change `IMG_HEIGHT` or `IMG_WIDTH`, delete the cache.

**Validation.** `HAS_VAL` is true when `datasets/{NAME}/val/org/` exists. ISIC2018 has a validation folder. KDSB has none, so the single-split path carves validation out of the labelled subset (`val_split = 0.3`). Under K-fold, each fold validates on its own held-out fold, and the external test set is never touched by any split.

**Labelled subset.** Each fraction takes the first `int(N * fraction)` training images (`X_train[:n]`). All results are conditional on that single draw.

## Configuration

Everything lives in the first code cell of each notebook.

| Setting | Default | Notes |
|---|---|---|
| `DATASET_NAME` | `"ISIC2018"` (`"KDSB"` in DBT-UNet) | must match a `DATASET_CONFIGS` key |
| `LABEL_FRACTIONS` | `[0.05, 0.10, 0.20, 0.50]` | add `0.01` for 1% |
| `USE_KFOLD` | `True` | `True` runs K-fold + ensembling, `False` runs one train/val split. Never both. |
| `N_FOLDS` | `5` | capped at the number of labelled samples |
| `CV_RANDOM_STATE` | `42` | keep identical across notebooks so folds match |
| `PRETRAIN_FLG` | `False` (`True` in DBT-UNet) | `True` runs SSL pretraining, `False` loads the saved encoder |
| `TRAIN_FLG` | `True` | `False` loads saved fine-tuned weights |
| `TEST_MODE` | `False` | smoke test: 1 fraction, 1 epoch, ≤10 samples, 2 folds |
| `PRETRAIN_EPOCHS` / `PRETRAIN_BATCH` | `100` / `8` | SSL notebooks only |
| `FINETUNE_EPOCHS` / `FINETUNE_BATCH` | `200` / `8` | |
| `RECALL_FLOOR` | `0.05` | folds whose held-out recall falls below this are left out of the ensemble (still logged) |

Keep `CV_RANDOM_STATE` and `LABEL_FRACTIONS` identical across notebooks, or the comparison is meaningless.

**GPU pinning.** Each notebook sets `CUDA_VISIBLE_DEVICES` before TensorFlow is imported (setting it afterwards has no effect). Current pins: BT-UNet `1`, SimSiam `0`, baseline `0`, DBT-UNet `2`, evaluation `0`. SimSiam and the baseline share card 0, so change one before running them together.

## Smoke test (CPU, no data needed)

`BT Unet/smoke_test.py` checks the whole pipeline end to end on the CPU. It generates a small synthetic ISIC2018- and KDSB-shaped dataset in a temporary folder and runs everything there, so your real `datasets/`, `saved_models/` and `results/` are never touched. It covers:

- the five notebooks, executed from copies at 32×32 with `TEST_MODE = True`
- the ablation scripts, including resume and the analysis scripts
- the `Arch/` overlay (`verify_arch.py` plus three variants)
- path resolution in both driver notebooks

The GPU is hidden throughout. It needs `nbclient` and `ipykernel` on top of the requirements above.

```bash
cd "BT Unet"
python smoke_test.py                       # everything
python smoke_test.py --only notebooks      # or: scripts, arch, drivers
python smoke_test.py --datasets KDSB --keep
```

The script exits non-zero if any stage fails. On failure it keeps the workspace, with executed notebooks in `executed/` and script logs in `logs/`. The results measure nothing (one epoch on random blobs): the test only shows that every stage runs.

## Running

For a quick in-notebook check, set `TEST_MODE = True`, run all cells, confirm the pipeline finishes in a few minutes, then set it back.

1. **Pretrain encoders.** In `BT-UNet.ipynb` and `SimSiam-UNet.ipynb`, set `PRETRAIN_FLG = True` and run through the pretraining section. This saves `saved_models/{DATASET}_barlow_twins_encoder.keras` and `saved_models/{DATASET}_simsiam_encoder.keras`. Pretraining uses images only, so it runs once per dataset and every fraction reuses it. Set `PRETRAIN_FLG = False` afterwards.
2. **Fine-tune.** With `TRAIN_FLG = True` and `USE_KFOLD = True`, run the rest of each notebook. Each fraction trains 5 fold models, then averages the folds that pass the recall floor into one ensemble model.
3. **Baseline.** `UNet-Baseline.ipynb` has no pretraining section, so it goes straight from data loading to training.
4. **DBT-UNet.** See below. Run it once per arm.
5. **Evaluate** (optional). `Evaluation_Pipeline.ipynb` re-scores saved models in one pass.

## DBT-UNet (Dense Barlow Twins)

In BT-UNet, only the encoder is pretrained: `build_encoder` stops at the bottleneck and global-average-pools, so the decoder starts from random weights at fine-tuning time. DBT-UNet pretrains the whole U-Net. Its trunk runs the decoder too and has two heads:

- the **global head**, identical to BT-UNet's down to layer names, on the pooled bottleneck
- a **dense head** of 1×1 convolutions on the decoder output, which lifts its 16 channels to `DENSE_DIM` (128)

The objective is `L = L_global + BETA * L_dense`. The dense term is the same Barlow Twins loss, with every spatial position treated as a sample. With batch 8, the global cross-correlation matrix is estimated from 8 samples. At 256×256 and stride 4, the dense matrix gets 32,768.

For positions to correspond between the two views, both views share one random-resized-crop. Flips and rot90 are sampled independently per view and undone on the decoder feature map (`undo_geometry`) before the loss. Colour jitter stays independent.

At fine-tuning time, weights move **by layer name** into a freshly built U-Net: encoder, bottleneck and decoder, but not the output layer. If any expected layer fails to copy, `transfer_trunk_weights` raises an error rather than half-loading.

### Arms

Run the notebook twice, changing only `BETA` and `VARIANT`:

| Arm | `BETA` | `VARIANT` | What it is |
|---|---|---|---|
| Control | `0.0` | `base` | BT-UNet's objective under DBT's shared-crop augmentation. The decoder stays random. |
| Treatment | `1.0` | `b1` | adds the dense term, so the decoder is pretrained |

The control must be named `base`: `Ablations/ablation_report.py` computes deltas only against a variant with that exact name.

### DBT-specific settings

| Setting | Default | Notes |
|---|---|---|
| `PRETRAIN_SIZE` | `256` | set `128` to cut activation memory by about 4× and afford a bigger batch. The trunk transfers across resolutions. |
| `DENSE_DIM` | `128` | dense head width |
| `DENSE_STRIDE` | `4` | spatial subsampling before the dense loss |
| `DENSE_LAMBD` | `None` | off-diagonal weight for the dense term. `None` uses `LAMBD` (5e-3). |
| `FREEZE_ENCODER` / `FREEZE_DECODER` | `False` | both `True` trains the output layer alone (linear probe) |

The **Sanity Checks** cell runs before pretraining and takes seconds on CPU. It checks that `undo_geometry` inverts all eight flip/rot90 combinations (including mixed within one batch), that the dense loss separates matched from mismatched views and stays finite with a dead channel, that the global head matches BT-UNet's layer names, and that `BETA = 0` leaves decoder weights untouched while `BETA > 0` moves them.

## Outputs

Models go to `saved_models/`:

```
model_{DS}_UNet_{pct}pct_fold{i}.keras                   model_{DS}_UNet_{pct}pct_ensemble.keras
model_{DS}_BT-UNet_{pct}pct_fold{i}.keras                model_{DS}_BT-UNet_{pct}pct_ensemble.keras
model_{DS}_SimSiam-UNET_{pct}pct_fold{i}.keras           model_{DS}_SimSiam-UNET_{pct}pct_ensemble.keras
model_{DS}_DBT-UNet_{pct}pct_{VARIANT}_fold{i}.keras     model_{DS}_DBT-UNet_{pct}pct_{VARIANT}_ensemble.keras
{DS}_dense_barlow_twins_{VARIANT}_trunk.keras            # DBT pretrained trunk
```

CSVs go to `results/`, with one namespace per notebook so nothing is overwritten:

| | Baseline | BT-UNet | SimSiam | DBT-UNet |
|---|---|---|---|---|
| per-fold | `{DS}_baseline_cv_results.csv` | `{DS}_cv_results.csv` | `{DS}_simsiam_cv_results.csv` | `{DS}_densebarlow_{VARIANT}_cv_results.csv` |
| mean ± std | `{DS}_baseline_cv_summary.csv` | `{DS}_cv_summary.csv` | `{DS}_simsiam_cv_summary.csv` | `{DS}_densebarlow_{VARIANT}_cv_summary.csv` |
| ensemble | `{DS}_baseline_ensemble_results.csv` | `{DS}_ensemble_results.csv` | `{DS}_simsiam_ensemble_results.csv` | `{DS}_densebarlow_{VARIANT}_ensemble_results.csv` |
| single-split | `{DS}_baseline_results.csv` | `{DS}_results.csv` | `{DS}_simsiam_results.csv` | — |

DBT-UNet also writes `{DS}_densebarlow_{VARIANT}_pretrain_loss.csv` with the total, global and dense loss per epoch. Its rows carry explicit `Model`, `Variant`, `Beta` and `Kind` columns, and its ensemble rows record `Folds Combined` and `Folds Total`. The other notebooks' rows carry `Dataset`, `Label Fraction (%)` and `Fold`, so the family has to come from the filename.

## Comparing results

Compare `Test Dice`. Test metrics come from the external test set, which no fold split touches.

```python
import pandas as pd
DS = "ISIC2018"
frames = {
    "Baseline": f"results/{DS}_baseline_cv_results.csv",
    "BT-UNet":  f"results/{DS}_cv_results.csv",
    "SimSiam":  f"results/{DS}_simsiam_cv_results.csv",
    "DBT-UNet": f"results/{DS}_densebarlow_b1_cv_results.csv",
}
df = pd.concat([pd.read_csv(p).assign(Model=m) for m, p in frames.items()])
print(df.groupby(["Model", "Label Fraction (%)"])["Test Dice"]
        .agg(["mean", "median", "std", "min", "max"]))
```

At low fractions, report the median and per-fold spread next to the mean. Folds often split into working and collapsed groups, and the mean then lands between them and describes no real run. When comparing ensembles, check `Folds Combined`: if the recall floor dropped folds, the ensemble averages fewer models than its neighbours in the table.

## Evaluation pipeline

`Evaluation_Pipeline.ipynb` loads saved `.keras` files, scores them on train, validation and test, and writes `results/evaluation_results.csv` plus per-dataset and fold-summary CSVs. It handles several datasets in one pass (`DATASETS_TO_EVALUATE`) and chooses a validation set for each artefact:

- **fold models:** their exact held-out fold, rebuilt from `N_FOLDS` and `CV_RANDOM_STATE`
- **ensembles:** the `val/` folder when there is one. Otherwise the result is flagged `CONTAMINATED`, because every labelled sample was in most members' training data.

Its `MODEL_PATTERN` already includes the ablation-variant segment from `Ablations/eval_pattern_patch.py`, and results are grouped by `Variant`. Running `patch_eval_pipeline.py` on it applies no edits. The family list is `BT-UNet`, `SimSiam-UNET` and `UNet` only, and non-matching files are skipped without an error, so **DBT-UNet models are not picked up**. Use the DBT notebook's own result CSVs for those.

## Ablations

`BT Unet/Ablations/` holds a script-based version of the pipeline with every SSL hyperparameter exposed, resumable K-fold fine-tuning, collapse diagnosis (`tier0_sweep.py`), and a variant-vs-base report (`ablation_report.py`). `Ablations/Arch/` extends it with architecture flags (depth, width, activation, output head, attention, no skips). Drive both from their notebooks. See [`BT Unet/Ablations/README.md`](BT%20Unet/Ablations/README.md). The drivers find their own folder from the working directory and use `BT Unet/` as `PROJECT_DIR`. Set the `PROJECT_DIR` environment variable to point them somewhere else.

## Findings to date

All numbers are mean test Dice over 5 folds from one labelled-subset draw. With about 110 variant-vs-base comparisons in the ablations, roughly 5 would clear d/SE 2 by chance. Only trust effects that replicate across fractions or methods.

- **DBT-UNet (`b1`) on ISIC2018:** 0.632 at 1% and 0.676 at 5%, against BT-UNet's 0.551 and 0.569. The control arm (`base`) has not been run, so this gain cannot yet be split between the dense term and the shared-crop augmentation.
- **Pretraining batch size** is the only hyperparameter effect that replicated on both datasets: batch 64 improved BT-UNet by +0.097 Dice on ISIC2018 and +0.073 on KDSB.
- **Removing the output BatchNorm** (`--no-output-bn`, "outbn0") is the only architecture change with a large, replicated effect. On KDSB it raised BT-UNet from 0.582 to 0.830 and SimSiam from 0.492 to 0.840 at 20%, and cut 50% fold std from 0.15–0.18 to 0.01 or less. It did not help the scratch baseline (−0.012 at 50%). The threshold sweep points to a calibration effect: threshold-sensitive folds fell by 11 for each SSL method and by 3 for the baseline. Every notebook in this repository still uses the output BatchNorm.
- **Frozen encoder** helped at 1% and hurt at 10% on ISIC2018 for both SSL methods. This did not replicate on KDSB.
- **Depth 3 and width 8** matched the base model (94k vs 353k parameters). Depth 5 and width 32 trended worse.

### Open questions

- the DBT control arm, and DBT-UNet on KDSB
- whether the dense term and removing the output BatchNorm are additive
- more labelled-subset seeds, which would also decide whether the baseline's 50% cost from removing the BatchNorm is real
- `DENSE_STRIDE = 8` vs 4, to see how much spatial autocorrelation limits the dense term
- recalibrating the output BatchNorm statistics after training, to test the train/inference-mismatch explanation for the outbn0 gain

## Reusing trained models

Set `TRAIN_FLG = False` to skip training and load saved weights. Filenames must match the scheme above exactly. If `model.load_weights()` complains about layer names, use `tf.keras.models.load_model(path, compile=False)` instead, which reads the structure from the file.

## Troubleshooting

**`FileNotFoundError` on the encoder or trunk.** Pretraining has not run for this dataset (or this DBT `VARIANT`), or `DATASET_NAME` changed. Set `PRETRAIN_FLG = True` once.

**`RuntimeError: Trunk weight transfer is incomplete`** (DBT-UNet). The saved trunk was built with different filter counts or layer prefixes from the current architecture. Re-pretrain it.

**Results table is empty.** `USE_KFOLD = True` fills `cv_results`, not `all_results`. Read the K-fold summary cell instead of the single-split table.

**OOM during K-fold.** Each fold clears the session, but the ensemble step loads every surviving fold model at once. Lower `FINETUNE_BATCH` or `N_FOLDS`.

**Fewer folds than requested.** `n_folds_this = min(N_FOLDS, n_labelled)`. At small fractions there may be fewer samples than folds. DBT-UNet skips fractions with fewer than 2 labelled images.

**Visualisation cell loads nothing.** It looks for the 50% ensemble and falls back to the 50% single-split model (fold 0 in DBT-UNet). Neither exists until a full run has finished.
