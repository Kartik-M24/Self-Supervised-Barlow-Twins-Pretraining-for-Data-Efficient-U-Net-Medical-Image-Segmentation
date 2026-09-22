"""
Barlow Twins and SimSiam, with every hyperparameter the ablations need exposed
as an argument instead of a module-level constant.

Faithful to your notebooks at the default settings:
  Barlow Twins  proj_dim=128, lambd=5e-3, batch=8, epochs=100
  SimSiam       proj_dim=128, pred_dim=64, batch=8, epochs=100

Augmentation defaults reproduce custom_augment() from both notebooks, including
the colour-jitter probability of 0.8 that you flagged in the BT comment as a
deliberate departure from the reference implementation's 0.9.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Activation, GlobalAvgPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from common import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, encoder, bottleneck,
)

AUTO = tf.data.AUTOTUNE
SEED = 42
WEIGHT_DECAY = 5e-4


# ============================================================
# AUGMENTATION
# ============================================================

def make_augmenter(strength=0.5, jitter_p=0.8, use_crop=True, use_flip=True,
                   use_rot90=True, crop_scale=(0.75, 1.0)):
    """Returns a tf.data-mappable augmentation fn.

    The component switches drive the augmentation-ablation runs. Turning
    everything off leaves an identity map, which is a useful floor: SSL with no
    augmentation should collapse, and if it does not, the pretraining is not
    doing what you think it is.
    """
    crop_to = IMG_HEIGHT

    def random_resize_crop(image):
        image_shape = IMG_HEIGHT
        image = tf.image.resize(image, (image_shape, image_shape))
        size = tf.random.uniform(shape=(1,), minval=crop_scale[0] * image_shape,
                                 maxval=crop_scale[1] * image_shape, dtype=tf.float32)
        size = tf.cast(size, tf.int32)[0]
        crop = tf.image.random_crop(image, (size, size, 3))
        return tf.image.resize(crop, (crop_to, crop_to))

    def color_jitter(x):
        x = tf.image.random_brightness(x, max_delta=0.8 * strength)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength)
        x = tf.image.random_hue(x, max_delta=0.2 * strength)
        return tf.clip_by_value(x, 0, 255)

    def random_apply(func, x, p):
        return func(x) if tf.random.uniform([], 0, 1) < p else x

    def custom_augment(image):
        image = tf.cast(image, tf.float32)
        if use_flip:
            image = tf.image.random_flip_left_right(image)
        if use_crop:
            image = random_resize_crop(image)
        else:
            image = tf.image.resize(image, (crop_to, crop_to))
        if use_rot90:
            image = random_apply(lambda x: tf.image.rot90(x), image, p=0.5)
        if jitter_p > 0:
            image = random_apply(color_jitter, image, p=jitter_p)
        return image

    return custom_augment


def make_ssl_dataset(X, batch_size, augment_fn, shuffle_buffer=1024):
    """Two independently augmented views of the same array, zipped.

    The whole thing is built under /cpu:0. Without that, from_tensor_slices
    materialises X eagerly onto the GPU, which is what ate GPU 0 before.
    """
    with tf.device("/cpu:0"):
        ds_one = (
            tf.data.Dataset.from_tensor_slices(X)
            .shuffle(shuffle_buffer, seed=SEED)
            .map(augment_fn, num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
        )
        ds_two = (
            tf.data.Dataset.from_tensor_slices(X)
            .shuffle(shuffle_buffer, seed=SEED)
            .map(augment_fn, num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
        )
        return tf.data.Dataset.zip((ds_one, ds_two))


# ============================================================
# LR SCHEDULE
# ============================================================

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("total_steps must be >= warmup_steps")
        lr = (0.5 * self.learning_rate_base *
              (1 + tf.cos(self.pi * (tf.cast(step, tf.float32) - self.warmup_steps)
                          / float(self.total_steps - self.warmup_steps))))
        if self.warmup_steps > 0:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            lr = tf.where(step < self.warmup_steps, warmup_rate, lr)
        return tf.where(step > self.total_steps, 0.0, lr)

    def get_config(self):
        return dict(learning_rate_base=self.learning_rate_base,
                    total_steps=self.total_steps,
                    warmup_learning_rate=self.warmup_learning_rate,
                    warmup_steps=self.warmup_steps)


# ============================================================
# ENCODERS
# ============================================================

def _projection_head_barlow(x, hidden_dim):
    for i in range(2):
        x = Dense(hidden_dim, kernel_regularizer=l2(WEIGHT_DECAY),
                  name=f"projection_layer_{i}")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = Dense(hidden_dim, name="projection_output")(x)
    return x


def _projection_head_simsiam(x, hidden_dim):
    for i in range(2):
        x = Dense(hidden_dim, kernel_regularizer=l2(WEIGHT_DECAY),
                  name=f"proj_layer_{i}")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = Dense(hidden_dim, name="proj_output", kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)      # BN on output, no ReLU (SimSiam paper sec 3)
    return x


def build_ssl_encoder(method, proj_dim=128, depth=4, base_filters=16,
                      activation="relu"):
    """Trunk is identical across both methods and to the baseline U-Net encoder.
    Only the head differs, which is what keeps the three families comparable."""
    import arch
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    s = layers.Rescaling(1.0 / 255)(inputs)
    x, _ = arch.encoder(s, depth=depth, base_filters=base_filters,
                        activation=activation)
    x = arch.bottleneck(x, depth=depth, base_filters=base_filters,
                        activation=activation)
    trunk_output = GlobalAvgPool2D()(x)

    if method == "barlow":
        proj = _projection_head_barlow(trunk_output, proj_dim)
        name = "bt_encoder"
    elif method == "simsiam":
        proj = _projection_head_simsiam(trunk_output, proj_dim)
        name = "simsiam_encoder"
    else:
        raise ValueError(f"unknown method: {method}")

    return Model(inputs, proj, name=name)


# ============================================================
# BARLOW TWINS
# ============================================================

def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    return (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)


def barlow_loss(z_a, z_b, lambd):
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size
    on_diag = tf.reduce_sum(tf.pow(tf.linalg.diag_part(c) + (-1), 2))
    off_diag = tf.reduce_sum(tf.pow(off_diagonal(c), 2))
    return on_diag + (lambd * off_diag)


class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super().__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        ds_one, ds_two = data
        with tf.GradientTape() as tape:
            z_a = self.encoder(ds_one, training=True)
            z_b = self.encoder(ds_two, training=True)
            loss = barlow_loss(z_a, z_b, self.lambd)
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# ============================================================
# SIMSIAM
# ============================================================

class SimSiam(tf.keras.Model):
    """stop_gradient=False removes the stop-gradient, which is the collapse
    demonstration from the paper. The run should show loss diving toward -1 while
    downstream Dice falls apart."""

    def __init__(self, encoder, pred_hidden_dim=64, stop_gradient=True):
        super().__init__()
        self.encoder = encoder
        self.stop_gradient = stop_gradient
        self.predictor = self._build_predictor(encoder.output_shape[-1], pred_hidden_dim)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def _build_predictor(self, output_dim, hidden_dim):
        inp = Input(shape=(output_dim,))
        x = Dense(hidden_dim, kernel_regularizer=l2(WEIGHT_DECAY))(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(output_dim, kernel_regularizer=l2(WEIGHT_DECAY))(x)
        return Model(inp, x, name="predictor")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def cosine_loss(self, p, z):
        if self.stop_gradient:
            z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))

    def train_step(self, data):
        view_one, view_two = data
        with tf.GradientTape() as tape:
            z1 = self.encoder(view_one, training=True)
            z2 = self.encoder(view_two, training=True)
            p1 = self.predictor(z1, training=True)
            p2 = self.predictor(z2, training=True)
            loss = self.cosine_loss(p1, z2) / 2 + self.cosine_loss(p2, z1) / 2
        trainable_vars = self.encoder.trainable_variables + self.predictor.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def build_ssl_trainer(method, ssl_encoder, lambd=5e-3, pred_dim=64, stop_gradient=True):
    if method == "barlow":
        return BarlowTwins(ssl_encoder, lambd=lambd)
    if method == "simsiam":
        return SimSiam(ssl_encoder, pred_hidden_dim=pred_dim, stop_gradient=stop_gradient)
    raise ValueError(f"unknown method: {method}")
