#!/usr/bin/env python3
"""
Threshold sweep and collapse diagnosis over models you have already trained.
Predicts once per fold, caches the probabilities, then does every sweep in numpy.

  python tier0_sweep.py --dataset ISIC2018 --gpu 0

What it answers.

Is a collapsed fold actually collapsed? A fold scores ~0 because no sigmoid output
crosses 0.5. That has two causes with different meanings. Either the model learned
nothing, or it learned something and sits below the threshold. Three of your five
baseline folds at 5% on ISIC2018 hit exactly this, and the two readings support
different claims: SSL initialisation prevents collapse, or SSL initialisation
produces better-calibrated outputs at a fixed 0.5. This tells you which.

The output column `Diagnosis` reads:
  collapsed     nothing crosses any threshold, Dice stays near zero throughout
  calibration   near-zero at 0.5, but a lower threshold recovers real Dice
  healthy       Dice at 0.5 is already within 0.02 of the best threshold

What the best threshold costs you. Picking a threshold on the test set is
test-set tuning and cannot be reported as a headline number. Use the sweep two
ways: as the collapse diagnosis above, and as a robustness statement (dice at the
fixed 0.5 you preregistered, plus how much it moves across 0.3 to 0.7). If you do
want a tuned threshold in the paper, tune it on the held-out fold and apply it to
test, which --tune-on-foldval does.

Unfiltered ensembles are rebuilt here too, so this doubles as the re-ensembling
step for the runs that predate finetune.py.
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["ISIC2018", "KDSB"])
    p.add_argument("--gpu", default="0")
    p.add_argument("--model-dir", default="saved_models")
    p.add_argument("--out-dir", default="results/tier0")
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                   help="default: 0.05 to 0.95 step 0.05")
    p.add_argument("--recall-floor", type=float, default=0.05)
    p.add_argument("--tune-on-foldval", action="store_true",
                   help="also report test Dice at the threshold that maximises "
                        "held-out fold Dice, which is honest tuning")
    p.add_argument("--families", nargs="*", default=None,
                   help="restrict to e.g. BT-UNet SimSiam Baseline")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--force", action="store_true", help="ignore cached probabilities")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import glob
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    import common
    from eval_pattern_patch import parse_model_name

    common.ensure_dirs()
    common.configure_gpu()
    os.makedirs(args.out_dir, exist_ok=True)

    thresholds = args.thresholds or [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    data = common.load_dataset(args.dataset)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    Y_test_f = Y_test.astype(np.float32)

    prob_dir = os.path.join(common.SAVE_NP, "test_probs")
    val_prob_dir = os.path.join(common.SAVE_NP, "foldval_probs")
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(val_prob_dir, exist_ok=True)

    # ---- discover fold models for this dataset ----
    entries, undated = [], []
    for path in sorted(glob.glob(os.path.join(args.model_dir, "*.keras"))):
        info = parse_model_name(path)
        if info is None or info["kind"] != "fold":
            continue
        # A filename with no dataset segment (model_UNet_10pct_fold0.keras) cannot
        # be attributed. Treating it as "might be this dataset" pulls models from
        # another dataset into the run, and the mismatch only surfaces later as a
        # broadcast error against the wrong test set. It also collides in the
        # probability cache, since the cache key comes from the filename.
        if info["dataset"] is None:
            undated.append(os.path.basename(path))
            continue
        if info["dataset"] != args.dataset:
            continue
        if args.families and info["family"] not in args.families:
            continue
        variant = info["variant"] or "base"
        if args.variants and variant not in args.variants:
            continue
        info["variant"] = variant
        info["path"] = path
        entries.append(info)

    if undated:
        print(f"\nSkipped {len(undated)} model(s) with no dataset in the filename; "
              f"they cannot be attributed to {args.dataset}:")
        for n in undated[:8]:
            print(f"    {n}")
        if len(undated) > 8:
            print(f"    ... and {len(undated) - 8} more")
        print("  These are almost certainly from an earlier run. Rename them with a "
              "dataset prefix or move them aside.\n")

    if not entries:
        sys.exit(f"No fold models for {args.dataset} in {args.model_dir}")
    print(f"Found {len(entries)} fold model(s)")

    def dice_np(gt, pred, smooth=1.0):
        """Per-image Dice averaged over the batch, matching common.dice_coeff."""
        gt = gt.reshape(gt.shape[0], -1)
        pred = pred.reshape(pred.shape[0], -1)
        inter = (gt * pred).sum(axis=1)
        return float(np.mean((2. * inter + smooth) / (gt.sum(axis=1) + pred.sum(axis=1) + smooth)))

    def recall_np(gt, pred):
        gt = gt.reshape(-1)
        pred = pred.reshape(-1)
        tp = float((gt * pred).sum())
        fn = float((gt * (1 - pred)).sum())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    rows = []
    probs_by_cell = {}

    for info in entries:
        key = os.path.splitext(os.path.basename(info["path"]))[0].replace("model_", "")
        test_cache = os.path.join(prob_dir, f"{key}_testprob.npy")
        val_cache = os.path.join(val_prob_dir, f"{key}_valprob.npy")

        need_val = args.tune_on_foldval
        have = os.path.exists(test_cache) and (not need_val or os.path.exists(val_cache))

        if have and not args.force:
            print(f"  cached  {key}")
            prob_test = np.load(test_cache).astype(np.float32)
            prob_val = np.load(val_cache).astype(np.float32) if need_val else None
        else:
            print(f"  predict {key}")
            model = tf.keras.models.load_model(info["path"], compile=False)
            prob_test = model.predict(X_test, verbose=0).astype(np.float32)
            np.save(test_cache, prob_test.astype(np.float16))

            prob_val = None
            if need_val:
                from sklearn.model_selection import KFold
                _, _, n_lab = common.labelled_subset(X_train, Y_train, info["fraction"])
                n_folds = min(5, n_lab)
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                splits = list(kf.split(np.arange(n_lab)))
                _, val_idx = splits[info["fold"]]
                prob_val = model.predict(X_train[:n_lab][val_idx], verbose=0).astype(np.float32)
                np.save(val_cache, prob_val.astype(np.float16))
                np.save(val_cache.replace("_valprob", "_valgt"),
                        Y_train[:n_lab][val_idx].astype(np.float16))

            del model
            tf.keras.backend.clear_session()

        cell = (info["family"], info["variant"], info["pct"])

        # A stale cache from a different dataset would otherwise surface much
        # later as an opaque broadcast error inside dice_np.
        if prob_test.shape[0] != Y_test_f.shape[0]:
            sys.exit(
                f"\nShape mismatch for {key}:\n"
                f"  predictions: {prob_test.shape[0]} images\n"
                f"  {args.dataset} test set: {Y_test_f.shape[0]} images\n"
                f"This usually means a cached probability file belongs to a "
                f"different dataset. Delete it and rerun:\n"
                f"  rm {test_cache}\n"
            )

        probs_by_cell.setdefault(cell, {})[info["fold"]] = prob_test

        dices = {t: dice_np(Y_test_f, (prob_test > t).astype(np.float32)) for t in thresholds}
        best_t = max(dices, key=dices.get)
        d_half = dices.get(0.5, dice_np(Y_test_f, (prob_test > 0.5).astype(np.float32)))

        if dices[best_t] < 0.10:
            diagnosis = "collapsed"
        elif d_half < 0.10 <= dices[best_t]:
            diagnosis = "calibration"
        elif dices[best_t] - d_half > 0.02:
            diagnosis = "threshold-sensitive"
        else:
            diagnosis = "healthy"

        row = {
            "Dataset": args.dataset, "Model": info["family"], "Variant": info["variant"],
            "Label Fraction (%)": info["pct"], "Fold": info["fold"], "Kind": "fold",
            "Prob Max": float(prob_test.max()),
            "Prob Mean": float(prob_test.mean()),
            "Prob p99.9": float(np.quantile(prob_test, 0.999)),
            "Frac Pixels > 0.5": float((prob_test > 0.5).mean()),
            "Test Dice @0.5": d_half,
            "Best Threshold": best_t,
            "Test Dice @best": dices[best_t],
            "Dice Range 0.3-0.7": max(dices[t] for t in thresholds if 0.3 <= t <= 0.7)
                                  - min(dices[t] for t in thresholds if 0.3 <= t <= 0.7),
            "Diagnosis": diagnosis,
        }
        for t in thresholds:
            row[f"Dice @{t:.2f}"] = dices[t]

        if args.tune_on_foldval and prob_val is not None:
            gt_val = np.load(val_cache.replace("_valprob", "_valgt")).astype(np.float32)
            val_d = {t: dice_np(gt_val, (prob_val > t).astype(np.float32)) for t in thresholds}
            t_star = max(val_d, key=val_d.get)
            row["Tuned Threshold (foldval)"] = t_star
            row["Test Dice @tuned"] = dices[t_star]

        rows.append(row)

    fold_df = pd.DataFrame(rows)
    fold_path = os.path.join(args.out_dir, f"{args.dataset}_threshold_sweep_folds.csv")
    fold_df.to_csv(fold_path, index=False)

    # ---- ensembles, filtered and unfiltered, across the same thresholds ----
    ens_rows = []
    for (family, variant, pct), folds in sorted(probs_by_cell.items()):
        sub = fold_df[(fold_df["Model"] == family) & (fold_df["Variant"] == variant)
                      & (fold_df["Label Fraction (%)"] == pct)]
        # Filter on test recall at 0.5 here, because the held-out fold recall the
        # notebooks used is not recoverable from a saved model alone. It reproduces
        # the same exclusions on your ISIC2018 run; if it ever disagrees, prefer the
        # notebook's own Folds Combined.
        keep = []
        for f_idx, prob in folds.items():
            r = recall_np(Y_test_f, (prob > 0.5).astype(np.float32))
            if r >= args.recall_floor:
                keep.append(f_idx)
        allf = sorted(folds)

        for label_, members in (("filtered", sorted(keep)), ("unfiltered", allf)):
            if not members:
                continue
            mean_prob = np.mean([folds[i] for i in members], axis=0)
            dices = {t: dice_np(Y_test_f, (mean_prob > t).astype(np.float32))
                     for t in thresholds}
            best_t = max(dices, key=dices.get)
            erow = {
                "Dataset": args.dataset, "Model": family, "Variant": variant,
                "Label Fraction (%)": pct, "Kind": "ensemble", "Filter": label_,
                "Folds Combined": len(members), "Folds Total": len(allf),
                "Excluded Folds": ",".join(str(i) for i in allf if i not in members),
                "Test Dice @0.5": dices.get(0.5),
                "Best Threshold": best_t,
                "Test Dice @best": dices[best_t],
            }
            for t in thresholds:
                erow[f"Dice @{t:.2f}"] = dices[t]
            ens_rows.append(erow)

    ens_df = pd.DataFrame(ens_rows)
    ens_path = os.path.join(args.out_dir, f"{args.dataset}_threshold_sweep_ensembles.csv")
    ens_df.to_csv(ens_path, index=False)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant, g in fold_df.groupby("Variant"):
        pcts = sorted(g["Label Fraction (%)"].unique())
        fig, axes = plt.subplots(1, len(pcts), figsize=(4.2 * len(pcts), 4),
                                 sharey=True, squeeze=False)
        for ax, pct in zip(axes[0], pcts):
            sub = g[g["Label Fraction (%)"] == pct]
            for family, gg in sub.groupby("Model"):
                ys = [gg[f"Dice @{t:.2f}"].mean() for t in thresholds]
                ax.plot(thresholds, ys, marker="o", markersize=3, label=family)
            ax.axvline(0.5, color="k", linestyle=":", linewidth=1)
            ax.set_title(f"{pct}% labels")
            ax.set_xlabel("Threshold")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("Mean fold test Dice")
        axes[0][-1].legend()
        fig.suptitle(f"{args.dataset} — threshold sensitivity ({variant})")
        path = os.path.join(args.out_dir, f"threshold_sweep_{args.dataset}_{variant}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")

    print(f"\nWrote:\n  {fold_path}\n  {ens_path}")

    counts = fold_df.groupby(["Model", "Label Fraction (%)", "Diagnosis"]).size()
    if not counts.empty:
        print("\nDiagnosis counts:")
        print(counts.to_string())


if __name__ == "__main__":
    sys.exit(main())
