#!/usr/bin/env python3
"""
Re-analysis of results you already have. No GPU, no TensorFlow, no retraining.

  python tier0_tables.py --results-dir results --out-dir results/tier0

Reads every *_cv_results.csv and *_ensemble_results.csv it can find and writes:

  fold_summary.csv      mean, std, median, min, max, n per (dataset, model, fraction)
  collapse_rates.csv    how many folds failed, per family and fraction
  headline.csv          fold mean vs ensemble side by side, with folds combined
  fold_scatter_*.png    per-fold test Dice, which is the figure that shows the
                        5% baseline result is bimodal rather than merely noisy
  tier0_summary.md      the above as text you can paste into a writeup

Two things this exists to fix in your current tables.

The ensembles are not like for like. RECALL_FLOOR removed 3 of 5 baseline folds at
5% on ISIC2018 and nothing anywhere else, so one cell of the comparison is a
best-of-2 average and the rest are best-of-5. Folds Combined belongs next to every
ensemble number.

The mean is a poor summary at 5%. Baseline folds came in at 0.528, 0.0001, 0.577,
0.063, 0.0001; the mean of 0.234 describes no run that happened. The collapse rate
and the median carry the finding, and the scatter shows it at a glance.

Model identity is inferred from the filename because the notebooks' cv_results rows
carry Dataset, Label Fraction and Fold but no Model column. Override with --map if
your filenames do not follow the conventions below.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# filename fragment -> family label
DEFAULT_MAP = [
    ("_baseline_", "Baseline"),
    ("_scratch_", "Baseline"),
    ("_unet_", "Baseline"),
    ("_simsiam_", "SimSiam"),
    ("_barlow_", "BT-UNet"),
    ("_bt_", "BT-UNet"),
]


def infer_model(path, overrides):
    name = os.path.basename(path).lower()
    for frag, label in overrides + DEFAULT_MAP:
        if frag.lower() in name:
            return label
    # The BT notebook writes {DATASET}_cv_results.csv with no family fragment.
    return "BT-UNet"


def infer_variant(path):
    m = re.search(r"_(?:baseline|scratch|unet|simsiam|barlow|bt)_(.+?)_cv_results\.csv$",
                  os.path.basename(path), re.IGNORECASE)
    return m.group(1) if m else "base"


def load_fold_frames(results_dir, overrides):
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*cv_results.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "Model" not in df.columns:
            df["Model"] = infer_model(path, overrides)
        if "Variant" not in df.columns:
            df["Variant"] = infer_variant(path)
        df["Source File"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        sys.exit(f"No *cv_results.csv found in {results_dir}")
    df = canonicalise(pd.concat(frames, ignore_index=True))
    # An adopted model is re-scored by the new runner while the notebook CSV
    # still holds the same fold. Numbers agree; counting both double-weights it.
    return dedupe(df, ["Dataset", "Model", "Variant", "Label Fraction (%)", "Fold"],
                  "fold")


def load_ensemble_frames(results_dir, overrides):
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*ensemble_results.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "Model" not in df.columns:
            df["Model"] = infer_model(path, overrides)
        if "Variant" not in df.columns:
            df["Variant"] = "base"
        if "Filter" not in df.columns:
            df["Filter"] = "filtered"     # the notebooks only ever wrote filtered
        df["Source File"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = canonicalise(pd.concat(frames, ignore_index=True))
    return dedupe(df, ["Dataset", "Model", "Variant", "Label Fraction (%)", "Filter"],
                  "ensemble")


GROUP = ["Dataset", "Model", "Variant", "Label Fraction (%)"]

# The notebooks carry no Model column, so the family is inferred from the
# filename and comes out as Baseline / SimSiam / BT-UNet. The new runner writes
# Model explicitly using the on-disk family names, UNet / SimSiam-UNET / BT-UNet.
# Same models, two spellings, so nothing matched and every family appeared twice.
# Canonicalise both to one set before grouping.
CANONICAL_MODEL = {
    "unet": "Baseline",
    "baseline": "Baseline",
    "scratch": "Baseline",
    "simsiam-unet": "SimSiam",
    "simsiam": "SimSiam",
    "bt-unet": "BT-UNet",
    "barlow": "BT-UNet",
    "bt": "BT-UNet",
}


def canonicalise(df):
    df["Model"] = (df["Model"].astype(str).str.strip()
                   .map(lambda m: CANONICAL_MODEL.get(m.lower(), m)))
    df["Variant"] = df["Variant"].astype(str).str.strip().replace({"nan": "base"})
    return df


def dedupe(df, key_cols, what):
    key = [c for c in key_cols if c in df.columns]
    if not key:
        return df
    n = len(df)
    df = df.sort_values("Source File").drop_duplicates(subset=key, keep="first")
    if n != len(df):
        print(f"Dropped {n - len(df)} duplicate {what} rows "
              f"(same key in more than one CSV).")
    return df


def as_table(df):
    """to_markdown needs tabulate, which is not in either notebook's imports."""
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        with pd.option_context("display.width", 250, "display.max_columns", 60,
                               "display.float_format", "{:.4f}".format):
            return "```\n" + df.to_string(index=False) + "\n```"


def fold_summary(folds, metric, recall_floor):
    recall_col = "FoldVal Recall" if "FoldVal Recall" in folds.columns else "Test Recall"

    def agg(g):
        v = g[metric].astype(float)
        collapsed = (g[recall_col].astype(float) < recall_floor)
        low = (v < 0.10).sum()
        high = (v > 0.50).sum()
        return pd.Series({
            "n": len(v),
            "mean": v.mean(),
            "std": v.std(ddof=1) if len(v) > 1 else np.nan,
            "median": v.median(),
            "min": v.min(),
            "max": v.max(),
            "collapsed": int(collapsed.sum()),
            "collapse_rate": collapsed.mean(),
            # a run either works or fails; when both groups are populated the
            # mean sits in a gap where no fold landed
            "bimodal": bool(low > 0 and high > 0),
        })

    return folds.groupby(GROUP, dropna=False).apply(agg, include_groups=False).reset_index()


def headline_table(summary, ensembles, metric):
    out = summary[GROUP + ["n", "mean", "std", "median", "collapsed", "bimodal"]].copy()
    out = out.rename(columns={
        "mean": f"Fold mean {metric}",
        "std": f"Fold std {metric}",
        "median": f"Fold median {metric}",
        "collapsed": "Collapsed folds",
        "n": "Folds run",
    })

    if ensembles.empty or metric not in ensembles.columns:
        return out

    for filt in sorted(ensembles["Filter"].dropna().unique()):
        sub = ensembles[ensembles["Filter"] == filt]
        cols = GROUP + [metric] + (["Folds Combined"] if "Folds Combined" in sub.columns else [])
        sub = sub[cols].rename(columns={
            metric: f"Ensemble {metric} ({filt})",
            "Folds Combined": f"Folds combined ({filt})",
        })
        out = out.merge(sub, on=GROUP, how="left")

    return out


def scatter(folds, summary, metric, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (dataset, variant), g in folds.groupby(["Dataset", "Variant"], dropna=False):
        models = sorted(g["Model"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 5))
        offsets = np.linspace(-1.2, 1.2, max(len(models), 2))

        for off, model in zip(offsets, models):
            sub = g[g["Model"] == model]
            xs = sub["Label Fraction (%)"].astype(float) + off
            ax.scatter(xs, sub[metric].astype(float), alpha=0.75, s=42, label=model)

            means = sub.groupby("Label Fraction (%)")[metric].mean()
            ax.plot(means.index.astype(float) + off, means.values,
                    linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Labelled data (%)")
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset} — per-fold {metric} ({variant})")
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=0.3)
        ax.legend()
        path = os.path.join(out_dir, f"fold_scatter_{dataset}_{variant}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")


def write_markdown(summary, headline, metric, out_dir, recall_floor):
    lines = ["# Tier 0 re-analysis", ""]
    lines.append(f"Metric: {metric}. Collapse threshold: held-out fold recall "
                 f"below {recall_floor}.")
    lines.append("")

    lines.append("## Fold mean vs ensemble")
    lines.append("")
    lines.append(as_table(headline))
    lines.append("")

    flagged = summary[summary["collapsed"] > 0]
    if not flagged.empty:
        lines.append("## Cells with collapsed folds")
        lines.append("")
        lines.append("Any ensemble in these cells was built from a subset of its folds, "
                     "so it is not comparable with an ensemble in a cell where nothing "
                     "was excluded.")
        lines.append("")
        cols = GROUP + ["n", "collapsed", "collapse_rate", "mean", "median"]
        lines.append(as_table(flagged[cols]))
        lines.append("")

    bimodal = summary[summary["bimodal"]]
    if not bimodal.empty:
        lines.append("## Cells where the mean is misleading")
        lines.append("")
        lines.append("Folds split into working and failed groups with nothing in "
                     "between. Report the collapse rate and the median; the mean falls "
                     "in the gap.")
        lines.append("")
        cols = GROUP + ["n", "mean", "median", "min", "max"]
        lines.append(as_table(bimodal[cols]))
        lines.append("")

    path = os.path.join(out_dir, "tier0_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="results/tier0")
    p.add_argument("--metric", default="Test Dice")
    p.add_argument("--recall-floor", type=float, default=0.05)
    p.add_argument("--map", nargs="*", default=[],
                   help="extra filename-fragment=Label pairs, e.g. _btunet_=BT-UNet")
    args = p.parse_args()

    overrides = []
    for item in args.map:
        frag, _, label = item.partition("=")
        if not label:
            sys.exit(f"--map entries look like _fragment_=Label, got: {item}")
        overrides.append((frag, label))

    os.makedirs(args.out_dir, exist_ok=True)

    folds = load_fold_frames(args.results_dir, overrides)
    ensembles = load_ensemble_frames(args.results_dir, overrides)

    if args.metric not in folds.columns:
        sys.exit(f"{args.metric!r} not in the per-fold CSVs. "
                 f"Available: {[c for c in folds.columns if 'Test' in c]}")

    # Rows from the original notebooks have no Kind column. After concat, Kind
    # exists but is NaN for those rows, and NaN == "fold" is False, so a plain
    # equality filter drops every notebook row without a word. KDSB has only
    # notebook CSVs, so it vanished entirely. Missing Kind means fold.
    if "Kind" in folds.columns:
        n_before = len(folds)
        folds = folds[folds["Kind"].fillna("fold") == "fold"]
        if len(folds) != n_before:
            print(f"Filtered {n_before - len(folds)} non-fold rows "
                  f"(ensemble or summary rows).")

    summary = fold_summary(folds, args.metric, args.recall_floor)
    headline = headline_table(summary, ensembles, args.metric)

    print("Wrote:")
    for name, df in (("fold_summary.csv", summary),
                     ("headline.csv", headline),
                     ("collapse_rates.csv",
                      summary[GROUP + ["n", "collapsed", "collapse_rate", "bimodal"]])):
        path = os.path.join(args.out_dir, name)
        df.to_csv(path, index=False)
        print(f"  {path}")

    scatter(folds, summary, args.metric, args.out_dir)
    write_markdown(summary, headline, args.metric, args.out_dir, args.recall_floor)

    print()
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", "{:.4f}".format):
        print(headline.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
