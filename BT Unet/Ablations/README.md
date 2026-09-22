# Ablation runners

Companion code for the BT-UNet / SimSiam-UNet / supervised-UNet comparison. The
architecture, losses, metrics and data loading are copied from your three
notebooks so results stay comparable with the ISIC2018 runs you already have.
Everything new sits around that core: configurable SSL hyperparameters, resume,
variant naming, and collapse bookkeeping.

The seed sweep is not included, per your call. Every result these scripts produce
is therefore conditional on one labelled subset, `X_train[:n]`, exactly as in the
notebooks. `common.labelled_subset()` keeps that behaviour deliberately.

## Files

| File | What it does | Needs a GPU |
|---|---|---|
| `Ablation_Driver.ipynb` | Notebook front end — run everything from cells | via subprocess |
| `common.py` | Architecture, data, losses, metrics, model construction | — |
| `ssl_models.py` | Barlow Twins and SimSiam with every knob exposed | — |
| `pretrain.py` | Runs one SSL pretraining variant, saves the encoder | yes |
| `finetune.py` | K-fold fine-tuning with resume and both ensemble variants | yes |
| `tier0_tables.py` | Re-analysis of CSVs you already have | no |
| `tier0_sweep.py` | Threshold sweep and collapse diagnosis from saved models | prediction only |
| `ablation_report.py` | Each variant vs `base`, with the delta next to fold-level noise | no |
| `eval_pattern_patch.py` | Replacement regex for the evaluation pipeline, with tests | — |
| `patch_eval_pipeline.py` | Writes `Evaluation_Pipeline_PATCHED.ipynb` using that regex | — |
| `Arch/` | Architecture-ablation overlay (see below) | yes |

The scripts resolve `datasets/`, `saved_models/`, `saved_np/` and `results/`
relative to the working directory, which the driver sets to `PROJECT_DIR`. The
setup cell finds `ABLATION_DIR` (this folder) from the notebook's working
directory and takes `PROJECT_DIR` as its parent, i.e. the folder holding the
training notebooks. To put the data somewhere else, set the `PROJECT_DIR`
environment variable (and `ABLATION_DIR` if the scripts live elsewhere) before
starting Jupyter. The `.npy` cache filenames match the notebooks', so arrays
already built get reused rather than rebuilt.

### `Arch/`: architecture ablations

`Arch/` overlays this folder. It holds `arch.py` (a parameterised U-Net whose
defaults reproduce `common.py` exactly), `verify_arch.py` (asserts that), and
copies of `common.py`, `ssl_models.py`, `pretrain.py`, `finetune.py` and
`tier0_sweep.py` extended with `--depth`, `--base-filters`, `--activation`,
`--output-activation`, `--no-output-bn`, `--attention` and `--no-skips`.
`tier0_sweep.py` is copied unchanged because a script's own directory comes
first on the import path, so it has to sit in `Arch/` to pick up the new
`common.py`. Drive it from `Arch/Ablation_Arch_Driver.ipynb`, and run
`verify_arch.py` before anything else.

## Two things these scripts fix relative to the notebooks

**The evaluation pipeline used to silently drop every ablation model.** Its
original `MODEL_PATTERN` allowed nothing between `{pct}pct` and the `_fold{i}` or
`_ensemble` suffix, so `model_ISIC2018_BT-UNet_5pct_frozen_fold0.keras` failed to
match, and `discover_models` skips non-matching files without raising. The
committed `Evaluation_Pipeline.ipynb` already carries the replacement pattern from
`eval_pattern_patch.py` and groups by `Variant`, so `patch_eval_pipeline.py`
applies 0 edits to it. It is kept for older copies of the notebook. Run
`python eval_pattern_patch.py` to see the pattern's tests pass.

**Backbone surgery by index breaks the moment you change a head.** Your BT
notebook cuts the pretrained encoder at `layers[-9]` and the SimSiam one at
`layers[-10]`. The two numbers differ only because the SimSiam projection head
carries one extra `BatchNormalization`. Change `proj_dim` depth, add a predictor,
and the index points somewhere else while the code still runs. `common.py` cuts
at `bneck_add` and reads skips from `enc_stage{0..3}_add` instead. Names do not
move.

## Run order

If you work in Jupyter, open `Ablation_Driver.ipynb` and work down it. Every
training cell shells out with `subprocess` rather than `%run` or an import, which
matters for VRAM: a subprocess releases all its GPU memory on exit, whereas
importing `common` loads TensorFlow into your kernel and holds the card until you
restart it. The notebook also renders the Tier 0 tables and plots inline, since
those touch only pandas.

From a terminal, the same steps run directly. Stage 0 first, and read it before
booking the GPU.

```bash
python tier0_tables.py --results-dir results --out-dir results/tier0
python tier0_sweep.py --dataset ISIC2018 --gpu 0 --tune-on-foldval
```

`tier0_tables.py` needs nothing but the CSVs. On your ISIC2018 results it
reports, at 5%:

| Model | Fold mean | Std | Median | Collapsed | Ensemble | Folds combined |
|---|---|---|---|---|---|---|
| BT-UNet | 0.569 | 0.047 | 0.557 | 0/5 | 0.566 | 5 |
| SimSiam | 0.631 | 0.100 | 0.595 | 0/5 | 0.623 | 5 |
| Baseline | 0.234 | 0.293 | 0.063 | 3/5 | 0.612 | 2 |

The last two columns are the point. `RECALL_FLOOR` dropped three baseline folds
at 5% and nothing anywhere else, so that 0.612 is a best-of-2 average sitting in
a table beside best-of-5 numbers. Report `Folds Combined` next to every ensemble.

`tier0_sweep.py` then asks whether those three folds collapsed or merely
miscalibrated. A fold scores zero because nothing crosses 0.5, and that has two
causes with different meanings: the model learned nothing, or it learned
something that sits under the threshold. The `Diagnosis` column says which. If it
comes back `calibration`, your headline claim changes from "SSL prevents
collapse" to "SSL produces better-calibrated outputs at a fixed threshold", and
the rest of the queue is worth re-ordering around that.

Then work through the stages in `Ablation_Driver.ipynb`. Stages 1 and 2 carry
the main result; 3 through 6 explain it; 7 and 8 are for leftover time.

| Stage | What it varies |
|---|---|
| 1 | Main sweep: scratch, BT and SimSiam at 1, 5, 10, 20, 50% |
| 2 | Frozen encoder (`--freeze-encoder`) at 1, 5, 10% |
| 3 | Pretraining length: 0, 25, 50, 200 epochs |
| 4 | SSL data budget: pretrain on 5% or 20% of images |
| 5 | Projector width, pretraining batch size, Barlow lambda |
| 6 | SimSiam predictor width, no stop-gradient |
| 7 | Augmentation strength, crop+flip only |
| 8 | Loss: Dice only, BCE only |

Everything runs on one GPU, in sequence. Pin it with `--gpu`, which sets
`CUDA_VISIBLE_DEVICES` before the TensorFlow import in each entrypoint; setting it
afterwards does nothing, which is what the comment in your BT notebook's cell 5
refers to. Each script owns the card for its own process and releases it on exit,
so no two runs ever hold VRAM at once.

## Budget

Per-epoch wall times summed from the notebook outputs, for a full 4-fraction,
5-fold sweep:

| | BT-UNet | SimSiam | Baseline |
|---|---|---|---|
| Total | 1.81 h | 2.05 h | 2.01 h |

The per-stage estimates in the driver's markdown cells come to about 32 GPU hours
for ISIC2018's full queue plus KDSB's main sweep, serialised on one card, which is
44% of a 72-hour booking.

By fraction, averaged across the three: 5% 0.10 h, 10% 0.20 h, 20% 0.65 h,
50% 1.00 h. Extrapolating on training-set size puts 1% near 0.03 h and 100% near
2.1 h. Pretraining cost was estimated rather than measured, since `PRETRAIN_FLG`
was False in all three notebook runs.

Metric computation is not in those numbers. `evalResult` runs Hausdorff distance
and the connected-component IoU on CPU across all 1000 test images, for every
fold. Neither shows up in the per-epoch timings, and on the 1% and 5% fractions
they can exceed the training time. `--fast-metrics` skips both per fold;
ensembles are always scored with the full set.

## Fractions

The notebooks define four fractions and you asked for five, so `finetune.py`
defaults to `0.01 0.05 0.10 0.20 0.50`. Adding 1% rather than 100% is a choice
worth revisiting: at 20% the three families span 0.006 Dice and at 50% they span
0.003, so the upper end is saturated and buys little, while 1% costs about two
minutes per model and sits where the gap should be widest. If you want the
full-supervision anchor as well, pass `--fractions 0.01 0.05 0.10 0.20 0.50 1.0`
and budget an extra 2 h per family.

## Naming

Base-variant names are identical to your notebooks', so `pretrain.py` finds the
encoders you already trained and `finetune.py` finds your completed fold models.

```
model_{DATASET}_{FAMILY}_{pct}pct[_{variant}][_fold{i}|_ensemble].keras
{DATASET}_barlow_twins[_{variant}]_encoder.keras
{DATASET}_simsiam[_{variant}]_encoder.keras
results/{DATASET}_{method}_{variant}_cv_results.csv
results/{DATASET}_{method}_{variant}_ensemble_results.csv
```

The CSVs are the one exception: rows now carry Variant, Keyname, Collapsed and
Prob Max columns your originals lack, and appending different columns to an
existing CSV misaligns it silently. `tier0_tables.py` reads both old and new.

Variants may not contain underscores and may not be `ensemble` or start with
`fold`. `common.model_keyname()` asserts both, since the parser depends on them.
Encoders are written with a sidecar `.json` recording every hyperparameter used,
so a run six weeks old still explains itself.

## Resume

A fold counts as finished only when its CSV row exists, and rows append as each
fold completes. A `.keras` alone proves nothing: `ModelCheckpoint(save_best_only)`
writes every time val Dice improves, so an interrupted fold leaves a real file at
whatever epoch it died on.

Since base-variant filenames match your notebooks', an existing model may also be
finished work from your completed runs. A `.partial` sentinel, written before
`fit` and removed after, tells the two apart. Only a file with a live sentinel is
ever deleted. Anything else is **adopted**: loaded, scored, and given a row
without retraining, which backfills collapse bookkeeping and both ensemble
variants onto your existing ISIC2018 results for the cost of a prediction pass.
Pass `--retrain-foreign` to retrain instead, or `--force` to retrain everything.

Test-set probabilities are cached per fold under `saved_np/test_probs/`, which is
what lets `tier0_sweep.py` sweep thresholds and rebuild ensembles without
touching a GPU. Budget about 130 MB per fold at 1000 test images; pass
`--no-save-probs` if disk is tight.

## Caveats

These scripts have not been run against your data. I do not have the datasets or
a GPU here, so the numpy and pandas paths are tested and the TensorFlow paths are
not. Smoke-test with one cheap configuration before launching the queue:

```bash
python pretrain.py --method barlow --dataset ISIC2018 --gpu 0 --epochs 1 --variant smoke
python finetune.py --dataset ISIC2018 --gpu 0 --method barlow \
    --encoder-variant smoke --variant smoke --fractions 0.01 --epochs 2 --fast-metrics
```

That exercises the encoder save/load, the name-based backbone cut, the fold loop,
resume, and both ensemble paths in a few minutes.

Two behaviours worth naming in a writeup. `--freeze-encoder` sets
`backbone.trainable = False`, which also puts the encoder's BatchNormalization
layers into inference mode for the whole run, so it is "frozen encoder incl. BN"
rather than "frozen weights". And the ensemble metrics in `finetune.py` come from
averaging cached sigmoid outputs and then thresholding, which is arithmetically
identical to the Keras `Average` layer over fold models that your notebooks
build, without holding five models in memory at once.
