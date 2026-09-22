"""
Parameterised architecture, for the activation / depth / skip / output-head
ablations.

Every default here reproduces common.py exactly: same layers, same names, same
parameter count. verify_arch.py asserts that, so a base run stays comparable with
everything you have already trained.

Three things worth knowing before you use it.

OUTPUT ACTIVATION IS CONSTRAINED. The loss is BCE+Dice on [0,1] targets.
  sigmoid      the default.
  tanhscaled   (tanh(x)+1)/2, which is algebraically sigmoid(2x) -- a sigmoid
               with twice the gain, so logits are pushed away from 0.5 harder.
               Given how many KDSB folds came back threshold-sensitive, this is
               the interesting one: it tests whether the problem is the decision
               boundary being too soft.
  softmax2     a 2-channel head with softmax over channels, then channel 1 taken
               as the foreground probability. This is the standard multi-class
               formulation of the same problem and keeps the loss unchanged.
  hardsigmoid  piecewise-linear sigmoid, saturates completely outside [-3, 3].
Raw tanh is NOT offered. It spans [-1, 1], so binary cross-entropy takes the log
of a negative number and the loss goes NaN on the first batch. If you want a
tanh-shaped output, tanhscaled is the version that trains.

CHANGING DEPTH OR ENCODER ACTIVATION MEANS RE-PRETRAINING. Both alter the
encoder, so an SSL encoder trained at depth 4 cannot initialise a depth-3 model.
pretrain.py takes the same flags; pass matching ones.

DEPTH IS INFERRED, NOT DECLARED, AT FINE-TUNE TIME. build_backbone reads the
depth back off the saved encoder's layer names, so you cannot accidentally pair a
depth-3 encoder with a depth-4 decoder.
"""

import re

from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import (
    SeparableConv2D, Conv2D, BatchNormalization, Activation, Dropout,
    MaxPooling2D, UpSampling2D, add, multiply, concatenate,
)


# ============================================================
# ACTIVATIONS
# ============================================================

# Names accepted by --activation for the conv blocks.
CONV_ACTIVATIONS = ["relu", "leakyrelu", "elu", "gelu", "silu", "selu", "mish"]

# Names accepted by --output-activation.
OUTPUT_ACTIVATIONS = ["sigmoid", "tanhscaled", "softmax2", "hardsigmoid"]


def activation_layer(name, layer_name=None):
    """LeakyReLU and PReLU are layers, not activation strings, so they cannot go
    through Activation(). This returns a callable either way."""
    name = name.lower()
    if name == "leakyrelu":
        return layers.LeakyReLU(negative_slope=0.01, name=layer_name)
    if name == "prelu":
        return layers.PReLU(name=layer_name)
    if name in ("relu", "elu", "gelu", "silu", "swish", "selu", "tanh", "sigmoid",
                "mish", "hard_sigmoid"):
        return Activation(name, name=layer_name)
    raise ValueError(f"unknown activation {name!r}; try one of {CONV_ACTIVATIONS}")


@register_keras_serializable(package="ablations")
class SelectChannel(layers.Layer):
    """Slice one channel, keeping the channel axis.

    A registered Layer rather than a Lambda: Keras 3 blocks deserialization of
    Lambda layers holding Python lambdas, so a saved model using one cannot be
    loaded back without enable_unsafe_deserialization().
    """

    def __init__(self, index=1, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[..., self.index:self.index + 1]

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)

    def get_config(self):
        return {**super().get_config(), "index": self.index}


# ============================================================
# BLOCKS
# ============================================================

def double_conv_layer(x, filter_size, size, dropout, batch_norm=False,
                      prefix="dec", activation="relu"):
    axis = 3
    conv = SeparableConv2D(size, (filter_size, filter_size), padding="same",
                           name=f"{prefix}_sepconv_a")(x)
    if batch_norm:
        conv = BatchNormalization(axis=axis, name=f"{prefix}_bn_a")(conv)
    conv = activation_layer(activation, f"{prefix}_act_a")(conv)
    conv = SeparableConv2D(size, (filter_size, filter_size), padding="same",
                           name=f"{prefix}_sepconv_b")(conv)
    if batch_norm:
        conv = BatchNormalization(axis=axis, name=f"{prefix}_bn_b")(conv)
    conv = activation_layer(activation, f"{prefix}_act_b")(conv)
    if dropout > 0:
        conv = Dropout(dropout, name=f"{prefix}_drop")(conv)
    shortcut = Conv2D(size, kernel_size=(1, 1), padding="same",
                      name=f"{prefix}_shortcut")(x)
    if batch_norm:
        shortcut = BatchNormalization(axis=axis, name=f"{prefix}_shortcut_bn")(shortcut)
    return add([shortcut, conv], name=f"{prefix}_add")


def attention_gate(x_skip, gating, inter_channels, prefix, activation="relu"):
    """Additive attention gate (Oktay et al., Attention U-Net).

    The gating signal comes from the coarser decoder feature map and reweights the
    skip before concatenation, so the decoder can suppress skip regions the deeper
    layers consider irrelevant. This is an ablation of the skip pathway, not a
    different architecture: everything else is untouched.
    """
    theta = Conv2D(inter_channels, (1, 1), padding="same",
                   name=f"{prefix}_att_theta")(x_skip)
    phi = Conv2D(inter_channels, (1, 1), padding="same",
                 name=f"{prefix}_att_phi")(gating)
    f = activation_layer(activation, f"{prefix}_att_act")(add([theta, phi],
                                                             name=f"{prefix}_att_add"))
    psi = Conv2D(1, (1, 1), padding="same", name=f"{prefix}_att_psi")(f)
    alpha = Activation("sigmoid", name=f"{prefix}_att_sigmoid")(psi)
    return multiply([x_skip, alpha], name=f"{prefix}_att_mul")


# ============================================================
# ENCODER / BOTTLENECK / DECODER
# ============================================================

def filters_for(depth, base_filters):
    return [base_filters * (2 ** i) for i in range(depth)]


def encoder(inputs, depth=4, base_filters=16, prefix="enc", activation="relu"):
    skip_connections = []
    x = inputs
    for i, f in enumerate(filters_for(depth, base_filters)):
        stage = f"{prefix}_stage{i}"
        a = double_conv_layer(x, 3, f, 0.1, True, prefix=stage, activation=activation)
        skip_connections.append(a)
        x = MaxPooling2D(pool_size=(2, 2), name=f"{stage}_pool")(a)
    return x, skip_connections


def bottleneck(inputs, depth=4, base_filters=16, prefix="bneck", activation="relu"):
    return double_conv_layer(inputs, 3, base_filters * (2 ** depth), 0.1, True,
                             prefix=prefix, activation=activation)


def decoder(inputs, skip_connections, depth=4, base_filters=16, prefix="dec",
            activation="relu", attention=False, use_skips=True):
    num_filters = filters_for(depth, base_filters)[::-1]
    skips = skip_connections[::-1]
    x = inputs
    for i, f in enumerate(num_filters):
        stage = f"{prefix}_stage{i}"
        x_up = UpSampling2D(size=(2, 2), data_format="channels_last",
                            name=f"{stage}_up")(x)
        if use_skips:
            s = skips[i]
            if attention:
                s = attention_gate(s, x_up, max(f // 2, 1), stage, activation)
            x_in = concatenate([x_up, s], axis=-1, name=f"{stage}_concat")
        else:
            # Skip-connection ablation: the decoder must reconstruct from the
            # bottleneck alone. Expect a large drop; the question is how large.
            x_in = x_up
        x = double_conv_layer(x_in, 3, f, 0.1, True, prefix=stage,
                              activation=activation)
    return x


def output_head(inputs, prefix="out", activation="sigmoid", use_bn=True):
    """Conv -> [BatchNorm] -> activation, returning a 1-channel map in [0, 1].

    use_bn=False drops the BatchNormalization that currently sits between the
    logit conv and the sigmoid. That layer normalises with batch statistics during
    training and with running averages at inference; on a modality-heterogeneous
    set like KDSB those averages are a single estimate spanning distributions that
    do not overlap, which shifts every logit by a constant and is a plausible
    cause of the threshold sensitivity.
    """
    activation = activation.lower()
    n_out = 2 if activation == "softmax2" else 1

    x = Conv2D(n_out, kernel_size=(1, 1), name=f"{prefix}_conv")(inputs)
    if use_bn:
        x = BatchNormalization(name=f"{prefix}_bn")(x)

    if activation == "sigmoid":
        return Activation("sigmoid", name=f"{prefix}_sigmoid")(x)

    if activation == "hardsigmoid":
        return Activation("hard_sigmoid", name=f"{prefix}_hardsigmoid")(x)

    if activation == "tanhscaled":
        # (tanh(x) + 1) / 2 is exactly sigmoid(2x): same shape, twice the gain.
        # Built from Activation + Rescaling rather than a Lambda, because Keras 3
        # refuses to deserialize Lambda layers under safe mode -- the model would
        # train fine and then fail to reload.
        t = Activation("tanh", name=f"{prefix}_tanh")(x)
        return layers.Rescaling(scale=0.5, offset=0.5,
                                name=f"{prefix}_tanhscaled")(t)

    if activation == "softmax2":
        p = Activation("softmax", name=f"{prefix}_softmax")(x)
        # Keep channel 1 as foreground so the loss, metrics and threshold logic
        # downstream stay identical to the 1-channel case.
        return SelectChannel(1, name=f"{prefix}_fg")(p)

    raise ValueError(f"unknown output activation {activation!r}; "
                     f"try one of {OUTPUT_ACTIVATIONS}")


# ============================================================
# NAMES AND DEPTH INFERENCE
# ============================================================

_STAGE_RE = re.compile(r"^enc_stage(\d+)_add$")
BOTTLENECK_LAYER_NAME = "bneck_add"


def skip_layer_names(depth):
    return [f"enc_stage{i}_add" for i in range(depth)]


def infer_depth(model):
    """Read the depth back off a saved encoder rather than trusting a flag.

    Pairing a depth-3 encoder with a depth-4 decoder would otherwise fail deep
    inside the graph with a shape error that says nothing about the cause.
    """
    stages = [int(m.group(1)) for m in
              (_STAGE_RE.match(l.name) for l in model.layers) if m]
    if not stages:
        raise ValueError(
            "No enc_stage*_add layers found; this does not look like an encoder "
            "built by arch.py or common.py."
        )
    return max(stages) + 1


def infer_base_filters(model, depth):
    """First encoder stage's filter count is the base width."""
    layer = model.get_layer("enc_stage0_add")
    return int(layer.output.shape[-1])
