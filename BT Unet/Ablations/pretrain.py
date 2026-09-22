#!/usr/bin/env python3
"""
Run one SSL pretraining variant and save the encoder.

  python pretrain.py --method barlow  --dataset ISIC2018 --gpu 1 --variant base
  python pretrain.py --method barlow  --dataset ISIC2018 --gpu 1 --epochs 25   --variant e25
  python pretrain.py --method barlow  --dataset ISIC2018 --gpu 1 --proj-dim 512 --variant proj512
  python pretrain.py --method simsiam --dataset ISIC2018 --gpu 2 --no-stopgrad --variant nostopgrad
  python pretrain.py --method barlow  --dataset ISIC2018 --gpu 1 --ssl-fraction 0.05 --variant sslbudget5

--epochs 0 saves the randomly initialised encoder without training. That gives you
the zero point on the pretraining-length curve, and fine-tuning from it should land
on top of the from-scratch baseline. If it does not, something other than the
pretrained weights differs between your BT/SimSiam and baseline pipelines.

Encoders land in saved_models/{DATASET}_{method}[_{variant}]_encoder.keras and are
skipped if they already exist, unless --force.
"""

import argparse
import json
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["barlow", "simsiam"])
    p.add_argument("--dataset", required=True, choices=["ISIC2018", "KDSB"])
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--variant", default=None,
                   help="tag for the encoder filename; no underscores")

    p.add_argument("--epochs", type=int, default=100,
                   help="0 = save random init without training")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--proj-dim", type=int, default=128)

    p.add_argument("--lambd", type=float, default=5e-3, help="Barlow Twins only")
    p.add_argument("--pred-dim", type=int, default=64, help="SimSiam only")
    p.add_argument("--no-stopgrad", action="store_true",
                   help="SimSiam only: remove the stop-gradient (collapse demo)")

    p.add_argument("--aug-strength", type=float, default=0.5)
    p.add_argument("--jitter-p", type=float, default=0.8)
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--no-rot90", action="store_true")

    p.add_argument("--ssl-fraction", type=float, default=1.0,
                   help="pretrain on the first N%% of train images only. Use the same "
                        "value as the downstream label fraction to answer 'is the gain "
                        "from the SSL objective or from the extra unlabelled images'")

    p.add_argument("--force", action="store_true", help="retrain even if the encoder exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Must precede the tensorflow import or it is a no-op.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorflow.keras.optimizers import SGD

    import common
    import ssl_models

    common.ensure_dirs()
    common.configure_gpu()

    out_name = common.encoder_filename(args.dataset, args.method, args.variant)
    out_path = os.path.join(common.SAVE_DIR, out_name)

    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists. Nothing to do (pass --force to retrain).")
        return

    data = common.load_dataset(args.dataset)
    X_train = data["X_train"]

    if args.ssl_fraction < 1.0:
        n = int(X_train.shape[0] * args.ssl_fraction)
        X_train = X_train[:n]
        print(f"SSL data budget: pretraining on the first {n} images "
              f"({args.ssl_fraction:.0%} of train)")

    ssl_encoder = ssl_models.build_ssl_encoder(args.method, proj_dim=args.proj_dim)
    print(f"Encoder: {ssl_encoder.count_params():,} params, "
          f"projection dim {args.proj_dim}")

    trainer = ssl_models.build_ssl_trainer(
        args.method, ssl_encoder,
        lambd=args.lambd,
        pred_dim=args.pred_dim,
        stop_gradient=not args.no_stopgrad,
    )

    meta = {
        "method": args.method, "dataset": args.dataset, "variant": args.variant,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "proj_dim": args.proj_dim, "lambd": args.lambd, "pred_dim": args.pred_dim,
        "stop_gradient": not args.no_stopgrad,
        "aug_strength": args.aug_strength, "jitter_p": args.jitter_p,
        "crop": not args.no_crop, "flip": not args.no_flip, "rot90": not args.no_rot90,
        "ssl_fraction": args.ssl_fraction, "n_ssl_images": int(X_train.shape[0]),
    }

    if args.epochs == 0:
        print("epochs=0: saving the randomly initialised encoder untrained.")
        ssl_encoder.save(out_path)
        meta["wall_seconds"] = 0.0
        meta["final_loss"] = None
    else:
        augment_fn = ssl_models.make_augmenter(
            strength=args.aug_strength,
            jitter_p=args.jitter_p,
            use_crop=not args.no_crop,
            use_flip=not args.no_flip,
            use_rot90=not args.no_rot90,
        )
        ssl_ds = ssl_models.make_ssl_dataset(X_train, args.batch, augment_fn)

        steps_per_epoch = len(X_train) // args.batch
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(args.epochs * 0.1) * steps_per_epoch

        lr_schedule = ssl_models.WarmUpCosine(
            learning_rate_base=args.lr,
            total_steps=total_steps,
            warmup_learning_rate=0.0,
            warmup_steps=warmup_steps,
        )
        trainer.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9))

        print(f"Pretraining {args.method} for {args.epochs} epochs "
              f"({steps_per_epoch} steps/epoch, batch {args.batch})")
        t0 = time.time()
        history = trainer.fit(ssl_ds, epochs=args.epochs)
        wall = time.time() - t0

        trainer.encoder.save(out_path)
        meta["wall_seconds"] = round(wall, 1)
        meta["final_loss"] = float(history.history["loss"][-1])

        fig_path = os.path.join(common.LOG_DIR,
                                f"pretrain_{os.path.splitext(out_name)[0]}.png")
        plt.figure()
        plt.plot(history.history["loss"])
        plt.title(f"{args.method} pretraining loss ({args.variant or 'base'})")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.3)
        plt.savefig(fig_path, dpi=110, bbox_inches="tight")
        plt.close()

        # Loss curves are the only signal you get from SSL before fine-tuning.
        # Save them so the pretraining-length curve can be plotted without reruns.
        import pandas as pd
        pd.DataFrame({"epoch": range(1, len(history.history["loss"]) + 1),
                      "loss": history.history["loss"]}).to_csv(
            os.path.join(common.LOG_DIR,
                         f"pretrain_{os.path.splitext(out_name)[0]}.csv"), index=False)

        print(f"Done in {wall / 60:.1f} min. Final loss {meta['final_loss']:.4f}")

    with open(out_path.replace(".keras", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved encoder to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
