#!/usr/bin/env python3
"""
K-fold fine-tuning across label fractions, for one encoder variant.

  # from-scratch baseline
  python finetune.py --dataset ISIC2018 --gpu 0 --scratch

  # pretrained, full fine-tuning (reproduces your completed runs at --variant base)
  python finetune.py --dataset ISIC2018 --gpu 1 --method barlow --encoder-variant base

  # frozen encoder
  python finetune.py --dataset ISIC2018 --gpu 1 --method barlow --encoder-variant base \
      --freeze-encoder --variant frozen --fractions 0.01 0.05 0.10

  # pretraining-length curve, cheap fractions only
  python finetune.py --dataset ISIC2018 --gpu 1 --method barlow --encoder-variant e25 \
      --variant e25 --fractions 0.05

Three differences from the notebook CV loop:

  RESUME. Every fold checks for its own .keras before training. A kernel death or a
  72-hour booking boundary costs you the fold in flight, not the sweep. Rows are
  appended to CSV as each fold finishes rather than at the end.

  BOTH ENSEMBLES. Each fraction gets a filtered ensemble (recall floor, matching
  your notebooks) and an unfiltered one over all folds. In your ISIC2018 run the
  floor dropped 3 of 5 baseline folds at 5% and none anywhere else, so the filtered
  baseline number is a best-of-2 sitting in a table next to best-of-5 numbers. The
  Filter and Folds Combined columns let you say which is which.

  SAVED PROBABILITIES. Test-set sigmoid outputs are cached per fold, so
  tier0_sweep.py can sweep thresholds and rebuild ensembles without a GPU.
"""

import argparse
import json
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["ISIC2018", "KDSB"])
    p.add_argument("--gpu", default="0")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scratch", action="store_true", help="from-scratch baseline U-Net")
    src.add_argument("--method", choices=["barlow", "simsiam"])

    p.add_argument("--encoder-variant", default=None,
                   help="which pretrained encoder to load (the --variant used in pretrain.py)")
    p.add_argument("--variant", default=None,
                   help="tag for the output model filenames; no underscores. "
                        "Omit to reproduce the original naming.")

    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.01, 0.05, 0.10, 0.20, 0.50])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss", default="bce_dice", choices=["bce_dice", "dice", "bce"])
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--recall-floor", type=float, default=0.05)

    p.add_argument("--fast-metrics", action="store_true",
                   help="skip Hausdorff and connected-component IoU per fold. Both run "
                        "on CPU over the whole test set and dominate wall time on the "
                        "cheap fractions. Ensembles are always scored in full.")
    p.add_argument("--no-save-probs", action="store_true")
    p.add_argument("--no-save-ensemble-model", action="store_true",
                   help="ensemble metrics are computed by averaging cached probabilities "
                        "either way; this only skips writing the .keras")
    p.add_argument("--retrain-foreign", action="store_true",
                   help="retrain models that exist on disk but were not written by "
                        "this runner. Default is to adopt them: score them and write "
                        "a results row, which backfills your completed notebook runs "
                        "without retraining anything.")
    p.add_argument("--force", action="store_true", help="retrain folds that already exist")

    g = p.add_argument_group("architecture")
    g.add_argument("--depth", type=int, default=4,
                   help="encoder stages; filters are base*2^i. Changes the "
                        "encoder, so the SSL encoder must be pretrained to match.")
    g.add_argument("--base-filters", type=int, default=16)
    g.add_argument("--activation", default="relu",
                   help="conv-block activation: relu leakyrelu elu gelu silu selu mish")

    g.add_argument("--output-activation", default="sigmoid",
                   help="sigmoid | tanhscaled | softmax2 | hardsigmoid. Raw tanh "
                        "is not offered: it spans [-1,1] and BCE goes NaN.")
    g.add_argument("--no-output-bn", action="store_true",
                   help="drop the BatchNorm between the logit conv and the "
                        "activation; the likely cause of KDSB threshold drift")
    g.add_argument("--attention", action="store_true",
                   help="attention gates on the skip connections")
    g.add_argument("--no-skips", action="store_true",
                   help="remove skip connections entirely")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import KFold
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger,
    )

    import common

    common.ensure_dirs()
    common.configure_gpu()

    method = "scratch" if args.scratch else args.method
    family = common.FAMILY_BY_METHOD[method]

    encoder_path = None
    if not args.scratch:
        encoder_path = os.path.join(
            common.SAVE_DIR,
            common.encoder_filename(args.dataset, args.method, args.encoder_variant),
        )
        if not os.path.exists(encoder_path):
            sys.exit(f"Encoder not found: {encoder_path}\nRun pretrain.py first.")
        print(f"Encoder: {encoder_path}")

    data = common.load_dataset(args.dataset)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    prob_dir = os.path.join(common.SAVE_NP, "test_probs")
    os.makedirs(prob_dir, exist_ok=True)

    run_tag = args.variant or "base"
    rows_path = os.path.join(
        common.RESULTS_DIR, f"{args.dataset}_{method}_{run_tag}_cv_results.csv")
    ens_path = os.path.join(
        common.RESULTS_DIR, f"{args.dataset}_{method}_{run_tag}_ensemble_results.csv")

    existing = pd.read_csv(rows_path) if os.path.exists(rows_path) else pd.DataFrame()

    def already_done(keyname):
        if args.force or existing.empty or "Keyname" not in existing.columns:
            return False
        return keyname in set(existing["Keyname"])

    def append_row(path, row):
        df = pd.DataFrame([row])
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    config = vars(args).copy()
    config["family"] = family
    with open(os.path.join(common.RESULTS_DIR,
                           f"{args.dataset}_{method}_{run_tag}_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    for fraction in args.fractions:
        pct = int(round(fraction * 100))
        X_subset, Y_subset, n_labelled = common.labelled_subset(X_train, Y_train, fraction)
        if n_labelled < 2:
            print(f"Skipping {pct}%: only {n_labelled} images.")
            continue

        n_folds = min(args.folds, n_labelled)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.cv_seed)

        print(f"\n{'=' * 60}")
        print(f"{family} | {run_tag} | {pct}% labelled | {n_folds} folds | {n_labelled} images")
        print(f"{'=' * 60}")

        fold_probs = {}

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_subset)):
            keyname = common.model_keyname(args.dataset, family, pct,
                                           variant=args.variant, fold=fold_idx)
            model_path = os.path.join(common.SAVE_DIR, f"model_{keyname}.keras")
            partial_path = model_path + ".partial"
            prob_path = os.path.join(prob_dir, f"{keyname}_testprob.npy")

            X_tr, Y_tr = X_subset[tr_idx], Y_subset[tr_idx]
            X_val, Y_val = X_subset[val_idx], Y_subset[val_idx]

            # Resume state machine.
            #
            # A fold counts as finished only when its CSV row exists. The .keras
            # alone is not proof: ModelCheckpoint writes best-so-far every time
            # val_dice_coeff improves, so an interrupted fold leaves a real file
            # behind at whatever epoch it died on.
            #
            # But a .keras with no row has a second possible cause. Base-variant
            # filenames are identical to the notebooks', so an existing file may
            # be a finished model from your completed ISIC2018 runs. Deleting
            # that would destroy hours of real work. The .partial sentinel
            # distinguishes them: this runner writes it before fit and removes it
            # after, so only a file with a live sentinel is ours and unfinished.
            # Anything else gets adopted -- loaded, scored, and given a row --
            # which also backfills rows for the runs you have already done.
            row_done = already_done(keyname)
            model_exists = os.path.exists(model_path)
            probs_exist = os.path.exists(prob_path)
            ours_partial = os.path.exists(partial_path)

            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ({keyname}) ---")

            if row_done and model_exists and probs_exist and not args.force:
                print("  Complete, skipping.")
                fold_probs[fold_idx] = np.load(prob_path).astype(np.float32)
                continue

            adopt = False
            if args.force:
                retrain = True
            elif model_exists and ours_partial:
                print("  Checkpoint has a live .partial sentinel: this runner was "
                      "interrupted mid-training. Discarding it and retraining.")
                os.remove(model_path)
                os.remove(partial_path)
                model_exists = False
                retrain = True
            elif model_exists and row_done:
                retrain = False          # only the cached probabilities are missing
            elif model_exists and not row_done:
                if args.retrain_foreign:
                    print("  Existing model not written by this runner; "
                          "--retrain-foreign given, retraining.")
                    retrain = True
                else:
                    print("  Existing model not written by this runner (no .partial "
                          "sentinel). Treating it as finished: scoring it and writing "
                          "a results row, without retraining. Pass --retrain-foreign "
                          "to override.")
                    retrain, adopt = False, True
            else:
                retrain = True

            model = common.build_segmentation_model(
                keyname,
                encoder_path=encoder_path,
                freeze_encoder=args.freeze_encoder,
                loss=args.loss,
                lr=args.lr,
                depth=args.depth,
                base_filters=args.base_filters,
                activation=args.activation,
                output_activation=args.output_activation,
                output_bn=not args.no_output_bn,
                attention=args.attention,
                use_skips=not args.no_skips,
            )

            t0 = time.time()
            if not retrain and not adopt:
                # Row and model both exist, only the cached probabilities are
                # missing (an earlier --no-save-probs run). Re-predict, do not
                # retrain, and do not append a duplicate row.
                print("  Already scored; regenerating cached probabilities only.")
                model.load_weights(model_path)
                prob_test = model.predict(X_test, verbose=0).astype(np.float32)
                np.save(prob_path, prob_test.astype(np.float16))
                fold_probs[fold_idx] = prob_test
                del model
                tf.keras.backend.clear_session()
                continue

            if adopt:
                model.load_weights(model_path)
                wall = 0.0
            else:
                callbacks = [
                    ModelCheckpoint(model_path, monitor="val_dice_coeff", mode="max",
                                    verbose=1, save_best_only=True),
                    ReduceLROnPlateau(monitor="val_dice_coeff", mode="max",
                                      factor=0.1, patience=10),
                    EarlyStopping(monitor="val_dice_coeff", mode="max", verbose=1,
                                  patience=args.patience, restore_best_weights=True),
                    CSVLogger(os.path.join(common.LOG_DIR,
                                           f"{keyname}_{common.timestamp()}.log")),
                ]
                with open(partial_path, "w") as f:
                    f.write(common.timestamp())
                model.fit(X_tr, Y_tr,
                          validation_data=(X_val, Y_val),
                          batch_size=args.batch,
                          epochs=args.epochs,
                          callbacks=callbacks,
                          verbose=1)
                wall = time.time() - t0
                if os.path.exists(partial_path):
                    os.remove(partial_path)

            prob_val = model.predict(X_val, verbose=0).astype(np.float32)
            prob_test = model.predict(X_test, verbose=0).astype(np.float32)

            if not args.no_save_probs:
                np.save(prob_path, prob_test.astype(np.float16))
            fold_probs[fold_idx] = prob_test

            pred_val = (prob_val > args.threshold).astype(np.float32)
            pred_test = (prob_test > args.threshold).astype(np.float32)

            print("  Held-out fold validation:")
            val_m = common.evalResult(Y_val.astype(np.float32), pred_val,
                                      fast=args.fast_metrics)
            print("  External test set:")
            test_m = common.evalResult(Y_test.astype(np.float32), pred_test,
                                       fast=args.fast_metrics)

            row = {
                "Dataset": args.dataset, "Model": family, "Variant": run_tag,
                "Label Fraction (%)": pct, "Fold": fold_idx, "Kind": "fold",
                "Keyname": keyname, "N Labelled": n_labelled,
                "Train Seconds": round(wall, 1),
                # collapse bookkeeping, so the rate is a column rather than a
                # thing you reconstruct from logs later
                "Collapsed": bool(val_m["Recall"] < args.recall_floor),
                "Prob Max": float(prob_test.max()),
                "Prob Mean": float(prob_test.mean()),
            }
            for split, metrics in (("FoldVal", val_m), ("Test", test_m)):
                for k, v in metrics.items():
                    row[f"{split} {k}"] = float(v)
            append_row(rows_path, row)

            del model
            tf.keras.backend.clear_session()

        if not fold_probs:
            continue

        # ---- Ensembles, both ways ----
        # Averaging sigmoid outputs then thresholding is exactly what a Keras
        # Average layer over the fold models does, so these numbers match the
        # notebook ensembles without loading five models into memory at once.
        val_recall = {}
        if os.path.exists(rows_path):
            df = pd.read_csv(rows_path)
            sub = df[(df["Label Fraction (%)"] == pct) & (df["Variant"] == run_tag)]
            val_recall = dict(zip(sub["Fold"], sub["FoldVal Recall"]))

        keep = [i for i in sorted(fold_probs)
                if val_recall.get(i, 1.0) >= args.recall_floor]
        allf = sorted(fold_probs)

        # Ensemble rows are scored with the full metric set, Hausdorff included,
        # which runs on CPU over all 1000 test images. Re-running the cell after
        # a resume must not redo that work or append a second copy of the row.
        done_ens = set()
        if os.path.exists(ens_path) and not args.force:
            edf = pd.read_csv(ens_path)
            if {"Label Fraction (%)", "Filter", "Variant"} <= set(edf.columns):
                done_ens = set(
                    edf[(edf["Variant"] == run_tag)]
                    .apply(lambda r: (int(r["Label Fraction (%)"]), r["Filter"],
                                      int(r.get("Folds Combined", -1))), axis=1)
                ) if len(edf) else set()

        for label_, members in (("filtered", keep), ("unfiltered", allf)):
            if not members:
                print(f"\n[{label_}] every fold excluded at {pct}%, skipping ensemble.")
                continue
            if (pct, label_, len(members)) in done_ens:
                print(f"\nEnsemble ({label_}, {len(members)} folds) at {pct}%: "
                      f"already scored, skipping.")
                continue
            mean_prob = np.mean([fold_probs[i] for i in members], axis=0)
            pred = (mean_prob > args.threshold).astype(np.float32)
            print(f"\nEnsemble ({label_}, {len(members)}/{len(allf)} folds) — test set:")
            m = common.evalResult(Y_test.astype(np.float32), pred, fast=False)

            erow = {
                "Dataset": args.dataset, "Model": family, "Variant": run_tag,
                "Label Fraction (%)": pct, "Kind": "ensemble", "Filter": label_,
                "Folds Combined": len(members), "Folds Total": len(allf),
                "Excluded Folds": ",".join(str(i) for i in allf if i not in members),
            }
            for k, v in m.items():
                erow[f"Test {k}"] = float(v)
            append_row(ens_path, erow)

        if not args.no_save_ensemble_model and keep:
            ens_key = common.model_keyname(args.dataset, family, pct,
                                           variant=args.variant, ensemble=True)
            ens_path_keras = os.path.join(common.SAVE_DIR, f"model_{ens_key}.keras")
            if not os.path.exists(ens_path_keras) or args.force:
                members_paths = [
                    os.path.join(common.SAVE_DIR, "model_" + common.model_keyname(
                        args.dataset, family, pct, variant=args.variant, fold=i) + ".keras")
                    for i in keep
                ]
                fold_models = [tf.keras.models.load_model(p, compile=False)
                               for p in members_paths]
                inp = tf.keras.Input(shape=(common.IMG_HEIGHT, common.IMG_WIDTH,
                                            common.IMG_CHANNELS))
                outs = [m(inp) for m in fold_models]
                avg = tf.keras.layers.Average()(outs) if len(outs) > 1 else outs[0]
                tf.keras.Model(inp, avg, name=ens_key).save(ens_path_keras)
                print(f"Saved ensemble model to {ens_path_keras}")
                del fold_models
                tf.keras.backend.clear_session()

        fold_probs.clear()

    print(f"\nDone.\n  per-fold: {rows_path}\n  ensembles: {ens_path}")


if __name__ == "__main__":
    sys.exit(main())
