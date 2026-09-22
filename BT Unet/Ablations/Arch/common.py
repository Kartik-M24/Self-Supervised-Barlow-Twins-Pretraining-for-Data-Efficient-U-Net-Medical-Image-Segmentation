"""
Shared building blocks for the ablation runs.

Everything here is copied from your three notebooks so the ablation results stay
comparable with the ISIC2018 runs you already have. Two things changed on purpose:

  1. build_backbone() cuts the pretrained encoder by LAYER NAME instead of by
     negative index. Your notebooks use layers[-9] (BT) and layers[-10] (SimSiam),
     and those two numbers differ only because the SimSiam projection head has one
     extra BatchNormalization. Any ablation that changes the projection or predictor
     head shifts the index and silently hands the decoder the wrong tensor. Names
     do not move.

  2. evalResult() takes fast=True, which skips Hausdorff distance and the
     connected-component IoU. Both run on CPU over all 1000 test images and neither
     appears in the per-epoch timings, so they are worth skipping per fold and
     computing once per ensemble.

Import order matters: set CUDA_VISIBLE_DEVICES before importing this module.
The CLI entrypoints (pretrain.py, finetune.py) already do that.
"""

import os
import datetime

import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage.morphology import label
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, SeparableConv2D, Conv2D, BatchNormalization, Activation, Dropout,
    MaxPooling2D, UpSampling2D, add, concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import (
    Accuracy, Precision, Recall, MeanIoU, MeanAbsoluteError,
)
from tensorflow.keras.losses import binary_crossentropy

# Imported at module level, not lazily, so the custom layers arch.py registers
# (SelectChannel, used by the softmax2 output head) are known to Keras before any
# model is loaded. Without this, tier0_sweep would fail to reload a softmax2
# model with an unknown-layer error.
import arch  # noqa: F401

try:
    from hausdorff import hausdorff_distance
    _HAS_HAUSDORFF = True
except ImportError:      # fast=True does not need it
    _HAS_HAUSDORFF = False


# ============================================================
# CONFIGURATION
# ============================================================

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
THRESHOLD = 0.5

SAVE_DIR = "saved_models"
SAVE_NP = "saved_np"
RESULTS_DIR = "results"
LOG_DIR = "logs"

DATASET_CONFIGS = {
    "KDSB": {
        "train_path": "datasets/KDSB/train/",
        "val_path":   "datasets/KDSB/val/",
        "test_path":  "datasets/KDSB/test/",
        "img_dir":    "org",
        "mask_dir":   "gt",
        "mask_name":  lambda img_name: img_name,
    },
    "ISIC2018": {
        "train_path": "datasets/ISIC2018/train/",
        "val_path":   "datasets/ISIC2018/val/",
        "test_path":  "datasets/ISIC2018/test/",
        "img_dir":    "org",
        "mask_dir":   "gt",
        "mask_name":  lambda img_name: os.path.splitext(img_name)[0] + "_segmentation.png",
    },
}


def ensure_dirs():
    for d in (SAVE_DIR, SAVE_NP, RESULTS_DIR, os.path.join(LOG_DIR, "fit")):
        os.makedirs(d, exist_ok=True)


def configure_gpu():
    """Memory growth on every visible GPU. Call once, after import."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"TensorFlow {tf.__version__} | GPUs visible: {[g.name for g in gpus]}")
    return gpus


# ============================================================
# DATA
# ============================================================

def load_split(split_path, img_dir, mask_dir, mask_name_fn):
    ids = next(os.walk(os.path.join(split_path, img_dir)))[2][:]
    X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        img = imread(os.path.join(split_path, img_dir, id_))[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
        X[n] = img

        mask = imread(os.path.join(split_path, mask_dir, mask_name_fn(id_)))
        mask = np.expand_dims(mask, axis=-1)
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode="constant", preserve_range=True)
        Y[n][mask[:, :, 0] / 255 > 0.] = 1.

    return X, Y


def load_dataset(dataset_name):
    """Same caching scheme and same .npy filenames as the notebooks, so the arrays
    you already built get reused rather than rebuilt."""
    cfg = DATASET_CONFIGS[dataset_name]
    img_dir, mask_dir, mask_fn = cfg["img_dir"], cfg["mask_dir"], cfg["mask_name"]

    has_val = os.path.isdir(os.path.join(cfg["val_path"], img_dir))

    def p(split, arr):
        return os.path.join(SAVE_NP, f"{dataset_name}_{arr}_{split}_{IMG_HEIGHT}x{IMG_WIDTH}.npy")

    paths = {
        "X_train": os.path.join(SAVE_NP, f"{dataset_name}_X_train_{IMG_HEIGHT}x{IMG_WIDTH}.npy"),
        "Y_train": os.path.join(SAVE_NP, f"{dataset_name}_Y_train_{IMG_HEIGHT}x{IMG_WIDTH}.npy"),
        "X_test":  os.path.join(SAVE_NP, f"{dataset_name}_X_test_{IMG_HEIGHT}x{IMG_WIDTH}.npy"),
        "Y_test":  os.path.join(SAVE_NP, f"{dataset_name}_Y_test_{IMG_HEIGHT}x{IMG_WIDTH}.npy"),
    }
    if has_val:
        paths["X_val"] = os.path.join(SAVE_NP, f"{dataset_name}_X_val_{IMG_HEIGHT}x{IMG_WIDTH}.npy")
        paths["Y_val"] = os.path.join(SAVE_NP, f"{dataset_name}_Y_val_{IMG_HEIGHT}x{IMG_WIDTH}.npy")

    if all(os.path.exists(v) for v in paths.values()):
        print("Found cached arrays, loading...")
        data = {k: np.load(v) for k, v in paths.items()}
    else:
        print("No cached arrays found, loading and resizing from disk...")
        data = {}
        data["X_train"], data["Y_train"] = load_split(cfg["train_path"], img_dir, mask_dir, mask_fn)
        data["X_test"], data["Y_test"] = load_split(cfg["test_path"], img_dir, mask_dir, mask_fn)
        if has_val:
            data["X_val"], data["Y_val"] = load_split(cfg["val_path"], img_dir, mask_dir, mask_fn)
        os.makedirs(SAVE_NP, exist_ok=True)
        for k, v in paths.items():
            np.save(v, data[k])

    if not has_val:
        data["X_val"], data["Y_val"] = None, None

    data["HAS_VAL"] = has_val
    shapes = {k: v.shape for k, v in data.items() if isinstance(v, np.ndarray)}
    print(f"Loaded {dataset_name}: {shapes}")
    return data


def labelled_subset(X_train, Y_train, fraction):
    """Sequential slice, matching the notebooks exactly.

    This is X_train[:n], not a random draw. Keeping it identical means the
    ablations sit on the same subset as your completed ISIC2018 runs and stay
    directly comparable. It also means every result here is conditional on that
    one draw, which is the limitation the seed sweep would have addressed.
    """
    n = int(X_train.shape[0] * fraction)
    return X_train[:n], Y_train[:n], n


# ============================================================
# ARCHITECTURE  (verbatim from the notebooks)
# ============================================================

def double_conv_layer(x, filter_size, size, dropout, batch_norm=False, prefix="dec"):
    axis = 3
    conv = SeparableConv2D(size, (filter_size, filter_size), padding="same",
                           name=f"{prefix}_sepconv_a")(x)
    if batch_norm:
        conv = BatchNormalization(axis=axis, name=f"{prefix}_bn_a")(conv)
    conv = Activation("relu", name=f"{prefix}_relu_a")(conv)
    conv = SeparableConv2D(size, (filter_size, filter_size), padding="same",
                           name=f"{prefix}_sepconv_b")(conv)
    if batch_norm:
        conv = BatchNormalization(axis=axis, name=f"{prefix}_bn_b")(conv)
    conv = Activation("relu", name=f"{prefix}_relu_b")(conv)
    if dropout > 0:
        conv = Dropout(dropout, name=f"{prefix}_drop")(conv)
    shortcut = Conv2D(size, kernel_size=(1, 1), padding="same", name=f"{prefix}_shortcut")(x)
    if batch_norm:
        shortcut = BatchNormalization(axis=axis, name=f"{prefix}_shortcut_bn")(shortcut)
    return add([shortcut, conv], name=f"{prefix}_add")


def encoder(inputs, prefix="enc"):
    num_filters = [16, 32, 64, 128]
    skip_connections = []
    x = inputs
    for i, f in enumerate(num_filters):
        stage = f"{prefix}_stage{i}"
        a = double_conv_layer(x, 3, f, 0.1, True, prefix=stage)
        skip_connections.append(a)
        x = MaxPooling2D(pool_size=(2, 2), name=f"{stage}_pool")(a)
    return x, skip_connections


def bottleneck(inputs, prefix="bneck"):
    return double_conv_layer(inputs, 3, 256, 0.1, True, prefix=prefix)


def decoder(inputs, skip_connections, prefix="dec"):
    num_filters = [128, 64, 32, 16]
    skip_connections = skip_connections[::-1]
    x = inputs
    for i, f in enumerate(num_filters):
        stage = f"{prefix}_stage{i}"
        x_up = UpSampling2D(size=(2, 2), data_format="channels_last", name=f"{stage}_up")(x)
        x_att = concatenate([x_up, skip_connections[i]], axis=-1, name=f"{stage}_concat")
        x = double_conv_layer(x_att, 3, f, 0.1, True, prefix=stage)
    return x


def output_layer(inputs, prefix="out"):
    x = Conv2D(1, kernel_size=(1, 1), name=f"{prefix}_conv")(inputs)
    x = BatchNormalization(name=f"{prefix}_bn")(x)
    x = Activation("sigmoid", name=f"{prefix}_sigmoid")(x)
    return x


# Layer names the decoder needs back from a pretrained encoder. Depth is read
# off the saved encoder rather than declared, so a depth-3 encoder cannot be
# paired with a depth-4 decoder.
SKIP_LAYER_NAMES = [f"enc_stage{i}_add" for i in range(4)]
BOTTLENECK_LAYER_NAME = "bneck_add"


# ============================================================
# LOSSES AND METRICS
# ============================================================

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 0.4 * binary_crossentropy(y_true, y_pred) + 0.6 * dice_loss(y_true, y_pred)


def bce_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


LOSSES = {
    "bce_dice": bce_dice_loss,   # the default your completed runs used
    "dice":     dice_loss,
    "bce":      bce_loss,
}


def iou_metric(y_true_in, y_pred_in):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(),
                                  bins=(true_objects, pred_objects))[0]
    area_true = np.expand_dims(np.histogram(labels, bins=true_objects)[0], -1)
    area_pred = np.expand_dims(np.histogram(y_pred, bins=pred_objects)[0], 0)
    union = area_true + area_pred - intersection
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        tp = np.sum(np.sum(matches, axis=1) == 1)
        fp = np.sum(np.sum(matches, axis=0) == 0)
        fn = np.sum(np.sum(matches, axis=1) == 0)
        return tp, fp, fn

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        prec.append(p)
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    return sum(iou_metric(y_true_in[b], y_pred_in[b])
               for b in range(y_true_in.shape[0])) / y_true_in.shape[0]


def haud_dist(y_true, y_pred):
    return hausdorff_distance(np.squeeze(y_true), np.squeeze(y_pred))


def haud_dist_batch(y_true, y_pred):
    if len(y_true.shape) == 2:
        return haud_dist(y_true, y_pred)
    return sum(haud_dist(y_true[b], y_pred[b]) for b in range(y_true.shape[0])) / y_true.shape[0]


def evalResult(gt, pred, num_class=2, fast=False, quiet=False):
    """Same metrics as the notebooks. fast=True drops HD and MyIoU, which are the
    two CPU-bound ones."""
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    acc = Accuracy();  acc.update_state(gt, pred);  r_acc = acc.result().numpy()
    pr = Precision();  pr.update_state(gt, pred);   r_pr = pr.result().numpy()
    rc = Recall();     rc.update_state(gt, pred);   r_rc = rc.result().numpy()
    mi = MeanIoU(num_class); mi.update_state(gt, pred); r_mi = mi.result().numpy()
    dc = sum(dice_coeff(gt[i], pred[i]).numpy() for i in range(gt.shape[0])) / gt.shape[0]
    r_mae = MeanAbsoluteError()(gt, pred).numpy()

    out = dict(Accuracy=r_acc, Precision=r_pr, Recall=r_rc,
               MeanIoU=r_mi, Dice=dc, MAE=r_mae)

    if not fast:
        if not _HAS_HAUSDORFF:
            # Raising here would discard a finished run over a missing optional
            # package, and this call happens after all the training. Warn and
            # carry on without the two CPU-bound metrics instead.
            if not evalResult._warned_hausdorff:
                print("  [warning] hausdorff not installed; skipping HD and MyIoU. "
                      "Install with: pip install hausdorff")
                evalResult._warned_hausdorff = True
        else:
            out["HD"] = haud_dist_batch(gt, pred)
            out["MyIoU"] = iou_metric_batch(gt, pred)

    if not quiet:
        bits = "  ".join(f"{k}={v:.4f}" for k, v in out.items())
        print(f"  {bits}")
    return out


evalResult._warned_hausdorff = False


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_backbone(encoder_path, trainable=True, return_depth=False):
    """Load a pretrained SSL encoder and cut it back to the bottleneck.

    Cut by name, not index. layers[-9] and layers[-10] in your notebooks both mean
    'the bottleneck add', but only for the exact projection heads those notebooks
    build. Change proj_dim depth or add a predictor and the index is wrong while
    the code still runs, which is the worst kind of wrong.
    """
    enc = tf.keras.models.load_model(encoder_path, compile=False)
    depth = arch.infer_depth(enc)
    backbone = tf.keras.Model(
        enc.input,
        enc.get_layer(BOTTLENECK_LAYER_NAME).output,
        name="backbone",
    )
    backbone.trainable = trainable
    return (backbone, depth) if return_depth else backbone


def build_segmentation_model(keyname, encoder_path=None, freeze_encoder=False,
                             loss="bce_dice", lr=None,
                             depth=4, base_filters=16, activation="relu",
                             output_activation="sigmoid", output_bn=True,
                             attention=False, use_skips=True):
    """One entrypoint for all three families.

    encoder_path=None gives the from-scratch baseline U-Net.
    freeze_encoder=True holds the pretrained weights fixed, which also puts the
    encoder's BatchNormalization layers in inference mode for the whole run. That
    is a real behavioural change on top of the frozen weights, so report it as
    'frozen encoder incl. BN' rather than 'frozen weights'.
    """
    if encoder_path is None:
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = layers.Rescaling(1.0 / 255)(inputs)
        x, skips = arch.encoder(s, depth=depth, base_filters=base_filters,
                                activation=activation)
        x = arch.bottleneck(x, depth=depth, base_filters=base_filters,
                            activation=activation)
    else:
        # Depth comes from the encoder on disk. Passing --depth at fine-tune time
        # cannot silently disagree with the encoder that was pretrained.
        backbone, depth = build_backbone(encoder_path, trainable=not freeze_encoder,
                                         return_depth=True)
        base_filters = arch.infer_base_filters(backbone, depth)
        skips = [backbone.get_layer(n).output for n in arch.skip_layer_names(depth)]
        inputs, x = backbone.input, backbone.output

    x = arch.decoder(x, skips, depth=depth, base_filters=base_filters,
                     prefix=f"{keyname}_dec", activation=activation,
                     attention=attention, use_skips=use_skips)
    outputs = arch.output_head(x, prefix=f"{keyname}_out",
                               activation=output_activation, use_bn=output_bn)
    model = Model(inputs, outputs, name=keyname)

    model.compile(
        loss=LOSSES[loss],
        optimizer=Adam() if lr is None else Adam(learning_rate=lr),
        metrics=["accuracy", Precision(), MeanIoU(num_classes=2), Recall(),
                 dice_coeff, MeanAbsoluteError()],
    )
    return model


# ============================================================
# NAMING
# ============================================================

FAMILY_BY_METHOD = {
    "barlow": "BT-UNet",
    "simsiam": "SimSiam-UNET",   # matches the filenames on disk
    "scratch": "UNet",
}

# Your notebooks save encoders as f"{DATASET_NAME}_barlow_twins_encoder.keras" and
# f"{DATASET_NAME}_simsiam_encoder.keras". Keep those stems so a base-variant run
# finds the encoders you already trained instead of retraining them.
BASE_ENCODER_STEM = {"barlow": "barlow_twins", "simsiam": "simsiam"}


def model_keyname(dataset, family, pct, variant=None, fold=None, ensemble=False):
    """model_{DS}_{FAMILY}_{pct}pct[_{variant}][_fold{i}|_ensemble]

    With variant=None this is byte-identical to the notebooks' naming, so base
    runs and your completed runs share the same files.

    variant must not contain underscores, and must not be the literal 'ensemble'
    or look like 'foldN'. eval_pattern_patch.py depends on both.
    """
    if variant:
        assert "_" not in variant, f"variant may not contain '_': {variant}"
        assert variant != "ensemble" and not variant.startswith("fold"), \
            f"reserved variant name: {variant}"
    parts = [dataset, family, f"{pct}pct"]
    if variant:
        parts.append(variant)
    if ensemble:
        parts.append("ensemble")
    elif fold is not None:
        parts.append(f"fold{fold}")
    return "_".join(parts)


def encoder_filename(dataset, method, variant=None):
    """Base variant reproduces the notebooks' filename exactly."""
    stem = BASE_ENCODER_STEM.get(method, method)
    tag = f"_{variant}" if variant else ""
    return f"{dataset}_{stem}{tag}_encoder.keras"


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
