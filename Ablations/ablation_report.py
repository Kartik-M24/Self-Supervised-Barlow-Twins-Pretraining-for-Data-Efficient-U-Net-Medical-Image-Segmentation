#!/usr/bin/env python3
"""
Ablation report that compares each variant against base using fold-level spread.

  python ablation_report.py --results-dir results

The delta table produced by the driver's summary cell subtracts fold MEANS. With
five folds and fold standard deviations between 0.02 and 0.15, most of those
deltas are smaller than the noise they sit in, and a single lucky fold can
manufacture one. This script puts the delta next to the uncertainty so you can
tell which is which.

Columns:

  n            folds contributing
  mean/median  both, because they disagree when a cell is driven by one outlier
  std          fold-to-fold spread within this variant
  d_mean       variant mean minus base mean
  d_median     variant median minus base median. If d_mean and d_median have
               opposite signs, the mean is being carried by one fold and the
               median is the honest number.
  SE           pooled standard error of the difference
  d/SE         effect size. Below 2 the difference is not separable from noise
               at this fold count.
  verdict      noise | weak | clear, plus a flag when mean and median disagree

Nothing here is a significance test in the publishable sense. Five folds drawn
from one labelled subset cannot support that, and folds share most of their
training data so they are not independent samples. Read d/SE as a triage tool:
it tells you which rows are worth a second look, not which are true.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_MAP = [
    ("_baseline_", "UNet"), ("_scratch_", "UNet"), ("_unet_", "UNet"),
    ("_simsiam_", "SimSiam-UNET"), ("_barlow_", "BT-UNet"), ("_bt_", "BT-UNet"),
]


def infer_model(path):
    name = os.path.basename(path).lower()
    for frag, label in DEFAULT_MAP:
        if frag in name:
            return label
    return "BT-UNet"          # the BT notebook wrote {DATASET}_cv_results.csv


def infer_variant(path):
    m = re.search(r"_(?:baseline|scratch|unet|simsiam|barlow|bt)_(.+?)_cv_results\.csv$",
                  os.path.basename(path), re.IGNORECASE)
    return m.group(1) if m else "base"


def load(results_dir, metric, exclude):
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*cv_results.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["Model"] = df["Model"] if "Model" in df.columns else infer_model(path)
        df["Variant"] = df["Variant"] if "Variant" in df.columns else infer_variant(path)
        df["Source File"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        sys.exit(f"no *cv_results.csv in {results_dir}")
    df = pd.concat(frames, ignore_index=True)

    if "Kind" in df.columns:
        df = df[df["Kind"].fillna("fold") == "fold"]

    for pat in exclude:
        df = df[~df["Variant"].str.contains(pat, case=False, na=False)]

    # Adopted models get re-scored under the new runner while the original
    # notebook CSV still holds the same fold. Identical numbers, counted twice.
    key = ["Dataset", "Model", "Variant", "Label Fraction (%)", "Fold"]
    before = len(df)
    df = df.sort_values("Source File").drop_duplicates(subset=key, keep="first")
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate fold rows "
              f"(same model, variant, fraction and fold in more than one CSV)\n")

    if metric not in df.columns:
        sys.exit(f"{metric!r} not found. Have: {[c for c in df.columns if 'Test' in c]}")
    return df


def report(df, metric, collapse_floor):
    recall_col = "FoldVal Recall" if "FoldVal Recall" in df.columns else "Test Recall"
    rows = []

    for (dataset, model, pct), g in df.groupby(["Dataset", "Model", "Label Fraction (%)"]):
        if "base" not in set(g["Variant"]):
            continue
        b = g[g["Variant"] == "base"][metric].astype(float)
        b_mean, b_med, b_std, b_n = b.mean(), b.median(), b.std(ddof=1), len(b)

        for variant, gv in g.groupby("Variant"):
            v = gv[variant == gv["Variant"]][metric].astype(float)
            v_mean, v_med = v.mean(), v.median()
            v_std = v.std(ddof=1) if len(v) > 1 else np.nan
            n = len(v)

            if variant == "base":
                d_mean = d_med = 0.0
                se, ratio, verdict = np.nan, np.nan, "reference"
            else:
                d_mean, d_med = v_mean - b_mean, v_med - b_med
                se = np.sqrt(b_std ** 2 / max(b_n, 1) + (v_std or 0) ** 2 / max(n, 1))
                ratio = abs(d_mean) / se if se and se > 0 else np.nan
                if np.isnan(ratio):
                    verdict = "?"
                elif ratio < 2:
                    verdict = "noise"
                elif ratio < 3:
                    verdict = "weak"
                else:
                    verdict = "clear"
                # A sign disagreement means one fold is carrying the mean.
                if verdict != "noise" and d_mean * d_med < 0:
                    verdict += " (outlier-driven)"

            collapsed = int((gv[recall_col].astype(float) < collapse_floor).sum())
            rows.append(dict(Dataset=dataset, Model=model, Fraction=pct,
                             Variant=variant, n=n, collapsed=collapsed,
                             mean=v_mean, median=v_med, std=v_std,
                             d_mean=d_mean, d_median=d_med, SE=se,
                             **{"d/SE": ratio}, verdict=verdict))

    out = pd.DataFrame(rows)
    return out.sort_values(["Dataset", "Model", "Fraction", "d_mean"],
                           ascending=[True, True, True, False])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--dataset", default=None,
                   help="restrict to one dataset, e.g. ISIC2018. Omit to include "
                        "every dataset found, which is what the default output "
                        "filename assumes.")
    p.add_argument("--out", default=None,
                   help="default: results/tier0/ablation_report[_DATASET].csv")
    p.add_argument("--metric", default="Test Dice")
    p.add_argument("--collapse-floor", type=float, default=0.05)
    p.add_argument("--exclude", nargs="*", default=["smoke"],
                   help="variant substrings to drop; smoke tests by default")
    args = p.parse_args()

    df = load(args.results_dir, args.metric, args.exclude)

    if args.dataset:
        found = sorted(df["Dataset"].dropna().unique())
        df = df[df["Dataset"] == args.dataset]
        if df.empty:
            sys.exit(f"No rows for dataset {args.dataset!r}. Found: {found}")

    if args.out is None:
        suffix = f"_{args.dataset}" if args.dataset else ""
        args.out = f"results/tier0/ablation_report{suffix}.csv"

    rep = report(df, args.metric, args.collapse_floor)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rep.to_csv(args.out, index=False)

    with pd.option_context("display.width", 220, "display.max_rows", 400,
                           "display.float_format", "{:.4f}".format):
        for (ds, model), g in rep.groupby(["Dataset", "Model"]):
            print(f"\n{'=' * 100}\n{ds} — {model}\n{'=' * 100}")
            print(g.drop(columns=["Dataset", "Model"]).to_string(index=False))

    print(f"\n\nWrote {args.out}")

    interesting = rep[rep["verdict"].str.startswith(("weak", "clear"))]
    if not interesting.empty:
        print("\nRows above the noise floor:")
        with pd.option_context("display.width", 220,
                               "display.float_format", "{:.4f}".format):
            print(interesting[["Model", "Fraction", "Variant", "mean", "median",
                               "d_mean", "d_median", "d/SE", "verdict"]]
                  .to_string(index=False))
    else:
        print("\nNothing clears the noise floor. With 5 folds from one labelled "
              "subset that is a real possibility, not a bug.")


if __name__ == "__main__":
    sys.exit(main())
