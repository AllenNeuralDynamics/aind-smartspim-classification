"""
Custom layers
"""

import keras
import keras.ops as ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="Custom")
class GroupNormalization3D(Layer):
    """
    Group Normalization for 3D data.

    Normalizes features within groups, making the model invariant to:
    - Different imaging power levels
    - Batch composition
    - Per-sample intensity variations

    This is CRITICAL for generalization across datasets with different
    acquisition settings.

    Args:
        groups: Number of groups to split channels into
        epsilon: Small constant for numerical stability
        center: If True, add learned offset (beta)
        scale: If True, add learned scale (gamma)
    """

    def __init__(self, groups=8, epsilon=1e-5, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        # Input shape: (batch, depth, height, width, channels)
        self.channels = input_shape[-1]

        if self.channels % self.groups != 0:
            raise ValueError(
                f"Number of channels ({self.channels}) must be divisible by "
                f"number of groups ({self.groups})"
            )

        shape = (self.channels,)

        if self.scale:
            self.gamma = self.add_weight(
                name="gamma", shape=shape, initializer="ones", trainable=True
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta", shape=shape, initializer="zeros", trainable=True
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        # Input shape: (N, D, H, W, C)
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]

        # Reshape to (N, D, H, W, groups, C // groups)
        x = ops.reshape(
            inputs,
            [
                batch_size,
                input_shape[1],
                input_shape[2],
                input_shape[3],
                self.groups,
                self.channels // self.groups,
            ],
        )

        # Compute mean and variance over spatial dims and channels within each group
        # Axis: (1, 2, 3, 5) = (D, H, W, channels_per_group)
        mean = ops.mean(x, axis=[1, 2, 3, 5], keepdims=True)
        variance = ops.var(x, axis=[1, 2, 3, 5], keepdims=True)

        # Normalize
        x = (x - mean) / ops.sqrt(variance + self.epsilon)

        # Reshape back to (N, D, H, W, C)
        x = ops.reshape(x, input_shape)

        # Apply scale and shift
        if self.scale:
            x = x * self.gamma
        if self.center:
            x = x + self.beta

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "groups": self.groups,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="Custom")
class ReduceMean3D(Layer):
    """Reduce mean along channel axis - replaces Lambda layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        return ops.mean(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="Custom")
class ReduceMax3D(Layer):
    """Reduce max along channel axis - replaces Lambda layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        return ops.max(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        return super().get_config()
