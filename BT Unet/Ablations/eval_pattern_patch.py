#!/usr/bin/env python3
"""
Drop-in replacement for MODEL_PATTERN and parse_model_name in
Evaluation_Pipeline.ipynb (cell 10).

Your current pattern is:

    ^model_(?:(?P<dataset>[A-Za-z0-9]+)_)??(?P<family>BT-UNet|SimSiam-UNET|UNet)_
    (?P<pct>\\d+)pct(?:_fold(?P<fold>\\d+)|_(?P<ensemble>ensemble))?(?:\\.keras)?$

Nothing between pct and fold/ensemble is allowed, so every ablation filename
(model_ISIC2018_BT-UNet_5pct_frozen_fold0.keras and friends) fails to match.
discover_models drops non-matching files without raising, so the failure mode is an
empty or short results table rather than an error, and you would be hunting for a
missing dataset before you thought to check the regex.

The version below adds an optional variant segment. Existing filenames still parse,
with variant None.

To use: replace the MODEL_PATTERN and parse_model_name definitions in cell 10 with
the two below, and add "Variant" wherever the pipeline groups or pivots, so two
variants of the same family and fraction do not silently average together in
pivot_table's aggfunc="mean".

Run this file directly to execute the self-tests:  python eval_pattern_patch.py
"""

import re

# Variant must not swallow the trailing fold/ensemble marker, hence the lookaheads.
# Underscores are excluded from the variant so the segments stay unambiguous;
# model_keyname() in common.py enforces the same rule when writing filenames.
MODEL_PATTERN = re.compile(
    r"^model_"
    r"(?:(?P<dataset>[A-Za-z0-9]+)_)??"
    r"(?P<family>BT-UNet|SimSiam-UNET|UNet)_"
    r"(?P<pct>\d+)pct"
    r"(?:_(?!fold\d+(?:\.keras)?$)(?!ensemble(?:\.keras)?$)(?P<variant>[A-Za-z0-9.\-]+?))?"
    r"(?:_fold(?P<fold>\d+)|_(?P<ensemble>ensemble))?"
    r"(?:\.keras)?$",
    re.IGNORECASE,
)

FAMILY_LABELS = {"bt-unet": "BT-UNet", "simsiam-unet": "SimSiam", "unet": "Baseline"}


def parse_model_name(filename):
    """Return a dict describing a model file, or None if the name does not match."""
    import os
    m = MODEL_PATTERN.match(os.path.basename(filename))
    if not m:
        return None

    if m.group("ensemble"):
        kind = "ensemble"
    elif m.group("fold") is not None:
        kind = "fold"
    else:
        kind = "single"

    family_key = m.group("family").lower()
    return {
        "dataset": m.group("dataset"),
        "family": FAMILY_LABELS.get(family_key, m.group("family")),
        "pct": int(m.group("pct")),
        "fraction": int(m.group("pct")) / 100.0,
        "variant": m.group("variant"),
        "fold": int(m.group("fold")) if m.group("fold") is not None else None,
        "kind": kind,
    }


def _test():
    cases = [
        # existing filenames keep parsing, variant None
        ("model_ISIC2018_BT-UNet_5pct_fold0.keras",
         dict(dataset="ISIC2018", family="BT-UNet", pct=5, variant=None, fold=0, kind="fold")),
        ("model_ISIC2018_BT-UNet_50pct_ensemble.keras",
         dict(dataset="ISIC2018", family="BT-UNet", pct=50, variant=None, fold=None,
              kind="ensemble")),
        ("model_ISIC2018_SimSiam-UNET_5pct_fold3.keras",
         dict(dataset="ISIC2018", family="SimSiam", pct=5, variant=None, fold=3, kind="fold")),
        ("model_KDSB_UNet_20pct.keras",
         dict(dataset="KDSB", family="Baseline", pct=20, variant=None, fold=None,
              kind="single")),
        ("model_UNet_10pct_fold1",                      # no dataset, no extension
         dict(dataset=None, family="Baseline", pct=10, variant=None, fold=1, kind="fold")),
        # new ablation filenames
        ("model_ISIC2018_BT-UNet_5pct_frozen_fold0.keras",
         dict(dataset="ISIC2018", family="BT-UNet", pct=5, variant="frozen", fold=0,
              kind="fold")),
        ("model_ISIC2018_BT-UNet_5pct_proj512_ensemble.keras",
         dict(dataset="ISIC2018", family="BT-UNet", pct=5, variant="proj512", fold=None,
              kind="ensemble")),
        ("model_ISIC2018_SimSiam-UNET_5pct_nostopgrad_fold2.keras",
         dict(dataset="ISIC2018", family="SimSiam", pct=5, variant="nostopgrad", fold=2,
              kind="fold")),
        ("model_ISIC2018_BT-UNet_1pct_aug0.25_fold4.keras",
         dict(dataset="ISIC2018", family="BT-UNet", pct=1, variant="aug0.25", fold=4,
              kind="fold")),
        ("model_KDSB_UNet_100pct_sslbudget-5_fold0.keras",
         dict(dataset="KDSB", family="Baseline", pct=100, variant="sslbudget-5", fold=0,
              kind="fold")),
        # a variant with no fold/ensemble suffix
        ("model_ISIC2018_BT-UNet_5pct_frozen.keras",
         dict(dataset="ISIC2018", family="BT-UNet", pct=5, variant="frozen", fold=None,
              kind="single")),
    ]

    failures = 0
    for name, expected in cases:
        got = parse_model_name(name)
        if got is None:
            print(f"FAIL  {name}\n      did not match")
            failures += 1
            continue
        bad = {k: (v, got[k]) for k, v in expected.items() if got[k] != v}
        if bad:
            print(f"FAIL  {name}")
            for k, (want, have) in bad.items():
                print(f"      {k}: expected {want!r}, got {have!r}")
            failures += 1
        else:
            print(f"ok    {name}")

    for name in ["notamodel.keras", "model_ISIC2018_ResNet_5pct.keras",
                 "model_ISIC2018_BT-UNet.keras", "readme.txt"]:
        if parse_model_name(name) is not None:
            print(f"FAIL  {name} should not have matched")
            failures += 1
        else:
            print(f"ok    rejected {name}")

    print(f"\n{failures} failure(s)")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if _test() else 0)
