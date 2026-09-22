#!/usr/bin/env python3
"""
Assert that arch.py at default settings reproduces common.py exactly, and that
every ablation variant actually builds.

  python verify_arch.py

The first half matters most. If the defaults drift by even one layer, base runs
under the new flags stop being comparable with everything already trained, and
nothing downstream would tell you.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, SeparableConv2D, Conv2D, BatchNormalization, Activation, Dropout,
    MaxPooling2D, UpSampling2D, add, concatenate,
)
from tensorflow.keras.models import Model

import arch

IMG = 256
CH = 3


# ---------- the original, copied verbatim from common.py ----------

def _orig_double_conv(x, filter_size, size, dropout, batch_norm=False, prefix="dec"):
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
    shortcut = Conv2D(size, kernel_size=(1, 1), padding="same",
                      name=f"{prefix}_shortcut")(x)
    if batch_norm:
        shortcut = BatchNormalization(axis=axis, name=f"{prefix}_shortcut_bn")(shortcut)
    return add([shortcut, conv], name=f"{prefix}_add")


def _orig_encoder(inputs, prefix="enc"):
    skips = []
    x = inputs
    for i, f in enumerate([16, 32, 64, 128]):
        stage = f"{prefix}_stage{i}"
        a = _orig_double_conv(x, 3, f, 0.1, True, prefix=stage)
        skips.append(a)
        x = MaxPooling2D(pool_size=(2, 2), name=f"{stage}_pool")(a)
    return x, skips


def _orig_bottleneck(inputs, prefix="bneck"):
    return _orig_double_conv(inputs, 3, 256, 0.1, True, prefix=prefix)


def _orig_decoder(inputs, skips, prefix="dec"):
    skips = skips[::-1]
    x = inputs
    for i, f in enumerate([128, 64, 32, 16]):
        stage = f"{prefix}_stage{i}"
        x_up = UpSampling2D(size=(2, 2), data_format="channels_last",
                            name=f"{stage}_up")(x)
        x_att = concatenate([x_up, skips[i]], axis=-1, name=f"{stage}_concat")
        x = _orig_double_conv(x_att, 3, f, 0.1, True, prefix=stage)
    return x


def _orig_output(inputs, prefix="out"):
    x = Conv2D(1, kernel_size=(1, 1), name=f"{prefix}_conv")(inputs)
    x = BatchNormalization(name=f"{prefix}_bn")(x)
    return Activation("sigmoid", name=f"{prefix}_sigmoid")(x)


def build_original():
    inp = Input((IMG, IMG, CH))
    s = layers.Rescaling(1.0 / 255)(inp)
    x, skips = _orig_encoder(s)
    x = _orig_bottleneck(x)
    x = _orig_decoder(x, skips, prefix="k_dec")
    out = _orig_output(x, prefix="k_out")
    return Model(inp, out, name="original")


def build_new(**kw):
    inp = Input((IMG, IMG, CH))
    s = layers.Rescaling(1.0 / 255)(inp)
    depth = kw.pop("depth", 4)
    base_filters = kw.pop("base_filters", 16)
    act = kw.pop("activation", "relu")
    x, skips = arch.encoder(s, depth=depth, base_filters=base_filters, activation=act)
    x = arch.bottleneck(x, depth=depth, base_filters=base_filters, activation=act)
    x = arch.decoder(x, skips, depth=depth, base_filters=base_filters, prefix="k_dec",
                     activation=act, attention=kw.pop("attention", False),
                     use_skips=kw.pop("use_skips", True))
    out = arch.output_head(x, prefix="k_out",
                           activation=kw.pop("output_activation", "sigmoid"),
                           use_bn=kw.pop("output_bn", True))
    assert not kw, f"unused kwargs: {kw}"
    return Model(inp, out, name="new")


# ---------- checks ----------

def check_defaults():
    print("=" * 72)
    print("Defaults must reproduce common.py exactly")
    print("=" * 72)
    a, b = build_original(), build_new()

    pa, pb = a.count_params(), b.count_params()
    print(f"  params      original {pa:,}   new {pb:,}")
    ok_params = pa == pb

    la = [l.name for l in a.layers]
    lb = [l.name for l in b.layers]
    print(f"  layer count original {len(la)}     new {len(lb)}")

    # Activation layers were renamed relu_a -> act_a so the name can carry any
    # activation. That rename is the ONLY intended difference.
    #
    # Keras also auto-numbers layers it names itself (input_layer, rescaling)
    # when a second model is built in the same session, so input_layer becomes
    # input_layer_1. That is a session artifact, not an architecture difference.
    import re as _re

    def norm(n):
        n = n.replace("_relu_a", "_act_a").replace("_relu_b", "_act_b")
        if n.startswith(("input_layer", "rescaling")):
            n = _re.sub(r"_\d+$", "", n)
        return n

    na = [norm(n) for n in la]
    nb = [norm(n) for n in lb]
    ok_names = na == nb
    if not ok_names:
        only_a = [n for n in na if n not in nb][:6]
        only_b = [n for n in nb if n not in na][:6]
        print(f"    only in original: {only_a}")
        print(f"    only in new:      {only_b}")

    ok_shape = a.output_shape == b.output_shape
    print(f"  output shape original {a.output_shape}  new {b.output_shape}")

    # Same weights in, same numbers out. This catches a reordered graph that
    # happens to have identical layer names and counts.
    b.set_weights(a.get_weights())
    x = np.random.RandomState(0).uniform(0, 255, (2, IMG, IMG, CH)).astype("float32")
    ya, yb = a.predict(x, verbose=0), b.predict(x, verbose=0)
    max_diff = float(np.abs(ya - yb).max())
    ok_num = max_diff < 1e-6
    print(f"  max output difference on shared weights: {max_diff:.3e}")

    for label, ok in [("param count", ok_params), ("layer names", ok_names),
                      ("output shape", ok_shape), ("numerics", ok_num)]:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return all([ok_params, ok_names, ok_shape, ok_num])


def check_variants():
    print("\n" + "=" * 72)
    print("Every variant must build, output 1 channel at full resolution, and")
    print("produce values inside [0, 1] so the BCE+Dice loss stays valid")
    print("=" * 72)

    cases = []
    for a_ in arch.CONV_ACTIVATIONS:
        cases.append((f"activation={a_}", dict(activation=a_)))
    for o in arch.OUTPUT_ACTIVATIONS:
        cases.append((f"output={o}", dict(output_activation=o)))
    for d in (2, 3, 4, 5):
        cases.append((f"depth={d}", dict(depth=d)))
    for w in (8, 32):
        cases.append((f"base_filters={w}", dict(base_filters=w)))
    cases.append(("output_bn=False", dict(output_bn=False)))
    cases.append(("attention", dict(attention=True)))
    cases.append(("no skips", dict(use_skips=False)))
    cases.append(("depth=3 + gelu + no bn",
                  dict(depth=3, activation="gelu", output_bn=False)))

    x = np.random.RandomState(1).uniform(0, 255, (2, IMG, IMG, CH)).astype("float32")
    failures = 0
    for label, kw in cases:
        try:
            m = build_new(**kw)
            y = m.predict(x, verbose=0)
            shape_ok = y.shape == (2, IMG, IMG, 1)
            range_ok = float(y.min()) >= -1e-6 and float(y.max()) <= 1 + 1e-6
            if shape_ok and range_ok:
                print(f"  ok    {label:28s} params={m.count_params():>10,}  "
                      f"range=[{y.min():.3f}, {y.max():.3f}]")
            else:
                print(f"  FAIL  {label:28s} shape={y.shape} "
                      f"range=[{y.min():.3f}, {y.max():.3f}]")
                failures += 1
            tf.keras.backend.clear_session()
        except Exception as e:
            print(f"  FAIL  {label:28s} {type(e).__name__}: {str(e)[:90]}")
            failures += 1
    return failures == 0


def check_depth_inference():
    print("\n" + "=" * 72)
    print("Depth must be recoverable from a saved encoder")
    print("=" * 72)
    failures = 0
    for d in (3, 4, 5):
        for w in (16, 32):
            inp = Input((IMG, IMG, CH))
            s = layers.Rescaling(1.0 / 255)(inp)
            x, _ = arch.encoder(s, depth=d, base_filters=w)
            x = arch.bottleneck(x, depth=d, base_filters=w)
            enc = Model(inp, layers.GlobalAvgPool2D()(x))
            gd, gw = arch.infer_depth(enc), arch.infer_base_filters(enc, d)
            ok = gd == d and gw == w
            print(f"  {'ok  ' if ok else 'FAIL'}  built depth={d} width={w} "
                  f"-> inferred depth={gd} width={gw}")
            failures += (not ok)
            tf.keras.backend.clear_session()
    return failures == 0


def check_tanh_identity():
    print("\n" + "=" * 72)
    print("tanhscaled should equal sigmoid(2x), i.e. a sigmoid with twice the gain")
    print("=" * 72)
    v = np.linspace(-6, 6, 2001).astype("float32")
    tanh_scaled = (np.tanh(v) + 1) / 2
    sigmoid_2x = 1 / (1 + np.exp(-2 * v))
    diff = float(np.abs(tanh_scaled - sigmoid_2x).max())
    print(f"  max difference over logits in [-6, 6]: {diff:.3e}")
    print(f"  {'PASS' if diff < 1e-6 else 'FAIL'}  algebraic identity")
    return diff < 1e-6


def check_roundtrip():
    """Every variant must survive save -> load.

    Keras 3 refuses to deserialize Lambda layers under safe mode, so a variant
    built with one trains normally and then fails at load time -- inside the
    ensemble step, after all the training is done, or later in tier0_sweep.
    """
    import tempfile

    print("\n" + "=" * 72)
    print("Every variant must survive a save/load round-trip")
    print("=" * 72)

    cases = [("default", {})]
    cases += [(f"activation={a}", dict(activation=a)) for a in arch.CONV_ACTIVATIONS]
    cases += [(f"output={o}", dict(output_activation=o)) for o in arch.OUTPUT_ACTIVATIONS]
    cases += [(f"depth={d}", dict(depth=d)) for d in (3, 5)]
    cases += [("no output bn", dict(output_bn=False)),
              ("attention", dict(attention=True)),
              ("no skips", dict(use_skips=False))]

    x = np.random.RandomState(2).uniform(0, 255, (2, IMG, IMG, CH)).astype("float32")
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        for label, kw in cases:
            path = os.path.join(td, "m.keras")
            try:
                m = build_new(**kw)
                before = m.predict(x, verbose=0)
                m.save(path)
                r = tf.keras.models.load_model(path, compile=False)
                after = r.predict(x, verbose=0)
                d = float(np.abs(before - after).max())
                if d < 1e-6:
                    print(f"  ok    {label:28s} reload matches (max diff {d:.1e})")
                else:
                    print(f"  FAIL  {label:28s} outputs differ by {d:.3e}")
                    failures += 1
                os.remove(path)
                tf.keras.backend.clear_session()
            except Exception as e:
                print(f"  FAIL  {label:28s} {type(e).__name__}: {str(e)[:80]}")
                failures += 1
    return failures == 0


def main():
    results = [
        ("defaults match common.py", check_defaults()),
        ("all variants build", check_variants()),
        ("depth inference", check_depth_inference()),
        ("tanhscaled identity", check_tanh_identity()),
        ("save/load round-trip", check_roundtrip()),
    ]
    print("\n" + "=" * 72)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 72)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
