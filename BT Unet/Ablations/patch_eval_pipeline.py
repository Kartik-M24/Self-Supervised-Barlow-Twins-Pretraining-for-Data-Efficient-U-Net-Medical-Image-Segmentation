#!/usr/bin/env python3
"""
Patch Evaluation_Pipeline.ipynb so it recognises ablation filenames.

  python patch_eval_pipeline.py /path/to/Evaluation_Pipeline.ipynb

Writes Evaluation_Pipeline_PATCHED.ipynb next to the original. Nothing is
overwritten. Outputs are preserved; only source is edited.

Four edits, each verified present before it is applied:

 1. MODEL_PATTERN gains an optional variant segment. Your version allows nothing
    between {pct}pct and _fold{i}/_ensemble, so every ablation filename fails to
    match. discover_models skips non-matching files silently, so the symptom is a
    short results table rather than an error.

 2. parse_model_name returns "variant".

 3. Result rows carry a Variant column.

 4. The comparison pivot and the fold summary group by Variant. Without this,
    pivot_table(aggfunc="mean") averages, say, the frozen and base runs of BT-UNet
    at 5% into a single cell, which looks like a plausible number and is not one.
"""

import json
import sys
from pathlib import Path

NEW_PATTERN = '''MODEL_PATTERN = re.compile(
    r"^model_"
    r"(?:(?P<dataset>[A-Za-z0-9]+)_)??"
    r"(?P<family>BT-UNet|SimSiam-UNET|UNet)_"
    r"(?P<pct>\\d+)pct"
    # Optional ablation variant. Lookaheads stop it swallowing a trailing
    # fold marker or "ensemble"; the lazy quantifier stops it swallowing
    # ".keras". Variants may not contain underscores.
    r"(?:_(?!fold\\d+(?:\\.keras)?$)(?!ensemble(?:\\.keras)?$)(?P<variant>[A-Za-z0-9.\\-]+?))?"
    r"(?:_fold(?P<fold>\\d+)|_(?P<ensemble>ensemble))?"
    r"(?:\\.keras)?$",
    re.IGNORECASE,
)'''

OLD_PATTERN_HEAD = 'MODEL_PATTERN = re.compile('
OLD_PATTERN_TAIL = '    re.IGNORECASE,\n)'

EDITS = [
    # (description, needle, replacement)
    (
        "parse_model_name returns variant",
        '''        "pct":      int(m.group("pct")),
        "fraction": int(m.group("pct")) / 100.0,
        "fold":     int(m.group("fold")) if m.group("fold") is not None else None,''',
        '''        "pct":      int(m.group("pct")),
        "fraction": int(m.group("pct")) / 100.0,
        "variant":  m.group("variant"),
        "fold":     int(m.group("fold")) if m.group("fold") is not None else None,''',
    ),
    (
        "result rows carry Variant",
        '''            "Model":              info["family"],
            "Label Fraction (%)": info["pct"],
            "Kind":               info["kind"],''',
        '''            "Model":              info["family"],
            "Variant":            info.get("variant") or "base",
            "Label Fraction (%)": info["pct"],
            "Kind":               info["kind"],''',
    ),
    (
        "comparison pivot splits by variant",
        '''        pivot = sub.pivot_table(index="Label Fraction (%)", columns="Model",
                                values=METRIC, aggfunc="mean")
        print(f"\\n{'='*50}\\n{ds} — {METRIC}\\n{'='*50}")''',
        '''        cols = ["Model", "Variant"] if sub["Variant"].nunique() > 1 else "Model"
        pivot = sub.pivot_table(index="Label Fraction (%)", columns=cols,
                                values=METRIC, aggfunc="mean")
        print(f"\\n{'='*50}\\n{ds} — {METRIC}\\n{'='*50}")''',
    ),
    (
        "plot pivot splits by variant",
        '''        sub = results_df[results_df["Dataset"] == ds]
        pivot = sub.pivot_table(index="Label Fraction (%)", columns="Model",
                                values=METRIC, aggfunc="mean")
        pivot.plot(marker="o", ax=ax)''',
        '''        sub = results_df[results_df["Dataset"] == ds]
        cols = ["Model", "Variant"] if sub["Variant"].nunique() > 1 else "Model"
        pivot = sub.pivot_table(index="Label Fraction (%)", columns=cols,
                                values=METRIC, aggfunc="mean")
        pivot.plot(marker="o", ax=ax)''',
    ),
    (
        "fold summary groups by variant",
        '''    non_metric = {"Dataset", "Model", "Label Fraction (%)", "Kind", "Fold", "Val Source", "File"}
    metric_cols = [c for c in fold_rows.columns if c not in non_metric and fold_rows[c].dtype != object]
    summary = fold_rows.groupby(["Dataset", "Model", "Label Fraction (%)"])[metric_cols].agg(["mean", "std"])''',
        '''    non_metric = {"Dataset", "Model", "Variant", "Label Fraction (%)", "Kind", "Fold", "Val Source", "File"}
    metric_cols = [c for c in fold_rows.columns if c not in non_metric and fold_rows[c].dtype != object]
    summary = fold_rows.groupby(["Dataset", "Model", "Variant", "Label Fraction (%)"])[metric_cols].agg(["mean", "std"])''',
    ),
    (
        "baseline delta keeps variants apart",
        '''        if "Baseline" in pivot.columns:
            delta = pivot.drop(columns="Baseline").sub(pivot["Baseline"], axis=0)''',
        '''        base_col = ("Baseline", "base") if isinstance(cols, list) else "Baseline"
        if base_col in pivot.columns:
            delta = pivot.drop(columns=base_col).sub(pivot[base_col], axis=0)''',
    ),
]


def patch_pattern(src):
    """Replace the MODEL_PATTERN block, which spans lines and cannot be a literal
    needle without reproducing every escape exactly."""
    start = src.find(OLD_PATTERN_HEAD)
    if start == -1:
        return src, False
    end = src.find(OLD_PATTERN_TAIL, start)
    if end == -1:
        return src, False
    end += len(OLD_PATTERN_TAIL)
    if "variant" in src[start:end]:
        return src, False          # already patched
    return src[:start] + NEW_PATTERN + src[end:], True


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"not found: {path}")

    nb = json.loads(path.read_text())
    applied, skipped = [], []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        original = src

        src, did = patch_pattern(src)
        if did:
            applied.append("MODEL_PATTERN gains a variant segment")

        for desc, needle, repl in EDITS:
            if needle in src:
                src = src.replace(needle, repl, 1)
                applied.append(desc)

        if src != original:
            cell["source"] = src.splitlines(keepends=True)

    wanted = 1 + len(EDITS)
    for desc, needle, _ in EDITS:
        if desc not in applied:
            skipped.append(desc)
    if "MODEL_PATTERN gains a variant segment" not in applied:
        skipped.append("MODEL_PATTERN gains a variant segment")

    out = path.with_name(path.stem + "_PATCHED.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

    print(f"Applied {len(applied)}/{wanted} edits:")
    for a in applied:
        print(f"  ok      {a}")
    for s in skipped:
        print(f"  SKIPPED {s}  (text not found — cell may already differ)")
    print(f"\nWrote {out}")
    print("Original untouched. Diff the two before replacing anything.")
    if skipped:
        print("\nSome edits did not apply. Make those by hand using "
              "eval_pattern_patch.py as the reference.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
