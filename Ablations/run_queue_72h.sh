#!/usr/bin/env bash
# ============================================================
# 72-hour ablation queue
# ============================================================
#
# Time estimates come from your completed ISIC2018 runs. Summing the per-epoch
# wall times printed in the three notebooks:
#
#     BT-UNet   1.81 h    SimSiam   2.05 h    Baseline   2.01 h
#
# for a full 4-fraction x 5-fold sweep, which breaks down as
#
#     5%  0.10 h    10%  0.20 h    20%  0.65 h    50%  1.00 h
#
# Extrapolating linearly in training-set size: 1% costs about 0.03 h and 100%
# about 2.1 h. Metric computation is on top of that and is not in these numbers;
# Hausdorff and connected-component IoU run on CPU over all 1000 test images per
# fold, so pass --fast-metrics on the low fractions and let the ensembles carry
# the full metric set.
#
# Everything below runs on ONE GPU, strictly in sequence. Each script owns the
# card for its own process and releases it on exit, so nothing overlaps. Roughly
# 33 GPU-hours for ISIC2018, which leaves most of a 72-hour booking as slack.
#
# Usage:
#     bash run_queue_72h.sh ISIC2018 0        # dataset, GPU index
#
# Run cost_model.py first for the current budget.
#
# Every step is resumable. Re-running the script after a crash skips folds whose
# .keras already exists and picks up where it stopped.

set -euo pipefail

DS="${1:-ISIC2018}"
GPU="${2:-0}"
FRAC_ALL="0.01 0.05 0.10 0.20 0.50"
FRAC_LOW="0.01 0.05 0.10"
FRAC_CHEAP="0.05"

log() { echo -e "\n\033[1m### $*\033[0m\n"; }

# ------------------------------------------------------------
# STAGE 0 — free. No GPU, no training. Run this before booking.
# ------------------------------------------------------------
# python tier0_tables.py --results-dir results --out-dir results/tier0
# python tier0_sweep.py --dataset "$DS" --gpu "$GPU" --tune-on-foldval
#
# tier0_sweep needs a GPU for prediction but no training; it finishes in minutes
# and tells you whether the 5% baseline folds collapsed or merely miscalibrated.
# If it says calibration, the headline claim changes and so does the rest of this
# queue, which is why it comes first.

# ------------------------------------------------------------
# STAGE 1 — extend the main sweep to five fractions      ~6.5 h
# ------------------------------------------------------------
# Adds 1%, which is where the SSL gap should be largest, and re-runs the existing
# fractions under the new runner so every result carries collapse bookkeeping and
# both ensemble variants.

log "STAGE 1: baseline"
python finetune.py --dataset "$DS" --gpu "$GPU" --scratch \
    --fractions $FRAC_ALL --fast-metrics

log "STAGE 1: BT-UNet"
python pretrain.py --method barlow --dataset "$DS" --gpu "$GPU" --variant base
python finetune.py --dataset "$DS" --gpu "$GPU" --method barlow \
    --encoder-variant base --fractions $FRAC_ALL --fast-metrics

log "STAGE 1: SimSiam"
python pretrain.py --method simsiam --dataset "$DS" --gpu "$GPU" --variant base
python finetune.py --dataset "$DS" --gpu "$GPU" --method simsiam \
    --encoder-variant base --fractions $FRAC_ALL --fast-metrics

# ------------------------------------------------------------
# STAGE 2 — frozen encoder                                ~0.7 h
# ------------------------------------------------------------
# Separates "the SSL weights are a good representation" from "the SSL weights are
# a good initialisation". Frozen also trains faster. Note this freezes encoder
# BatchNormalization into inference mode as well, so describe it as
# "frozen encoder incl. BN".

log "STAGE 2: frozen encoders"
for M in barlow simsiam; do
    python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
        --encoder-variant base --freeze-encoder --variant frozen \
        --fractions $FRAC_LOW --fast-metrics
done

# ------------------------------------------------------------
# STAGE 3 — pretraining length curve                      ~4 h
# ------------------------------------------------------------
# The 0-epoch point should reproduce the from-scratch baseline. If it does not,
# something other than the encoder weights differs between your pipelines, and
# you want to know that before interpreting any of the gaps.

log "STAGE 3: pretraining length"
for E in 0 25 50 200; do
    for M in barlow simsiam; do
        python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
            --epochs $E --variant "e${E}"
        python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
            --encoder-variant "e${E}" --variant "e${E}" \
            --fractions $FRAC_CHEAP --fast-metrics
    done
done
# e100 is the base encoder from Stage 1; no need to retrain it.

# ------------------------------------------------------------
# STAGE 4 — SSL data budget                               ~3 h
# ------------------------------------------------------------
# Both SSL methods currently pretrain on all 2594 ISIC2018 training images,
# including the 95% whose labels you withhold. This asks how much of the gain at
# 5% comes from the SSL objective and how much from simply seeing 20x more images.

log "STAGE 4: SSL data budget"
for M in barlow simsiam; do
    python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
        --ssl-fraction 0.05 --variant sslbudget5
    python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
        --encoder-variant sslbudget5 --variant sslbudget5 \
        --fractions 0.05 --fast-metrics

    python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
        --ssl-fraction 0.20 --variant sslbudget20
    python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
        --encoder-variant sslbudget20 --variant sslbudget20 \
        --fractions 0.20 --fast-metrics
done

# ------------------------------------------------------------
# STAGE 5 — projector, batch (both methods) + BT lambda   ~10 h
# ------------------------------------------------------------
# PROJECT_DIM is IMG_HEIGHT // 2 = 128 and PRETRAIN_BATCH is 8. Barlow Twins is
# sensitive to both, and the reference implementation uses projectors in the
# thousands with large batches. BT currently trails SimSiam at 5% on ISIC2018
# (0.569 vs 0.631 fold mean), so this sweep can change the result rather than
# merely characterise it. Both methods are swept on the shared parameters, so
# neither is compared at a disadvantage.

log "STAGE 5a: shared hyperparameters, both methods"
# proj-dim and batch exist in both methods, so both get swept. Only lambda
# (Barlow) and the predictor/stop-gradient (SimSiam) are method-specific.
for M in barlow simsiam; do
    for P in 256 512 2048; do
        python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
            --proj-dim $P --variant "proj${P}"
        python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
            --encoder-variant "proj${P}" --variant "proj${P}" \
            --fractions $FRAC_CHEAP --fast-metrics
    done
    for B in 32 64; do
        python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
            --batch $B --variant "bs${B}"
        python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
            --encoder-variant "bs${B}" --variant "bs${B}" \
            --fractions $FRAC_CHEAP --fast-metrics
    done
done

log "STAGE 5b: Barlow-only lambda"
for L in 0.001 0.02; do
    TAG="lambda$(echo $L | tr -d '.')"
    python pretrain.py --method barlow --dataset "$DS" --gpu "$GPU" \
        --lambd $L --variant "$TAG"
    python finetune.py --dataset "$DS" --gpu "$GPU" --method barlow \
        --encoder-variant "$TAG" --variant "$TAG" \
        --fractions $FRAC_CHEAP --fast-metrics
done

# ------------------------------------------------------------
# STAGE 6 — SimSiam-only hyperparameters                  ~2 h
# ------------------------------------------------------------
# The predictor bottleneck is 128 -> 64 -> 128, a ratio of 1/2; the paper uses
# 1/4. The no-stop-gradient run is the collapse demonstration: pretraining loss
# should dive toward -1 while downstream Dice falls apart.

log "STAGE 6: SimSiam predictor + stop-gradient"
for PD in 32 128; do
    python pretrain.py --method simsiam --dataset "$DS" --gpu "$GPU" \
        --pred-dim $PD --variant "pred${PD}"
    python finetune.py --dataset "$DS" --gpu "$GPU" --method simsiam \
        --encoder-variant "pred${PD}" --variant "pred${PD}" \
        --fractions $FRAC_CHEAP --fast-metrics
done

python pretrain.py --method simsiam --dataset "$DS" --gpu "$GPU" \
    --no-stopgrad --variant nostopgrad
python finetune.py --dataset "$DS" --gpu "$GPU" --method simsiam \
    --encoder-variant nostopgrad --variant nostopgrad \
    --fractions $FRAC_CHEAP --fast-metrics

# ------------------------------------------------------------
# STAGE 7 — augmentation                                  ~4 h
# ------------------------------------------------------------
# You already flagged the colour-jitter probability as a deliberate departure
# from the reference (0.8 rather than 0.9, for consistency across the two
# methods). This measures what that choice and the rest of the pipeline buy.
# The no-augmentation run is a floor check: SSL with an identity augmentation
# should collapse, and if it does not, pretraining is not doing what you think.

log "STAGE 7: augmentation"
for M in barlow simsiam; do
    python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
        --aug-strength 0.25 --variant aug025
    python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
        --encoder-variant aug025 --variant aug025 --fractions $FRAC_CHEAP --fast-metrics

    python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
        --aug-strength 1.0 --variant aug100
    python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
        --encoder-variant aug100 --variant aug100 --fractions $FRAC_CHEAP --fast-metrics

    python pretrain.py --method $M --dataset "$DS" --gpu "$GPU" \
        --jitter-p 0 --no-rot90 --variant cropflip
    python finetune.py --dataset "$DS" --gpu "$GPU" --method $M \
        --encoder-variant cropflip --variant cropflip --fractions $FRAC_CHEAP --fast-metrics
done

# ------------------------------------------------------------
# STAGE 8 — loss weighting, if time remains               ~2 h
# ------------------------------------------------------------
log "STAGE 8: loss weighting"
for L in dice bce; do
    python finetune.py --dataset "$DS" --gpu "$GPU" --scratch \
        --loss $L --variant "loss${L}" --fractions $FRAC_CHEAP --fast-metrics
    python finetune.py --dataset "$DS" --gpu "$GPU" --method barlow \
        --encoder-variant base --loss $L --variant "loss${L}" \
        --fractions $FRAC_CHEAP --fast-metrics
done

# ------------------------------------------------------------
log "Collating"
python tier0_tables.py --results-dir results --out-dir results/tier0
python tier0_sweep.py --dataset "$DS" --gpu "$GPU"
echo "Queue complete."
