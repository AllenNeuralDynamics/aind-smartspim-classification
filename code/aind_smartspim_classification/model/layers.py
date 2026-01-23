"""
Custom Keras layers for 3D volumes.

This module provides lightweight, serializable layers tailored for 3D
medical or microscopy data:

- `GroupNormalization3D`: Group Normalization operating over 3D spatial
    dimensions plus channel groups. Robust to intensity and acquisition
    variability across datasets. Fully serializable through
    `keras.saving.register_keras_serializable`.
- `ReduceMean3D`: Reduces the channel dimension by mean, returning a
    single-channel volume. Useful for channel aggregation.
- `ReduceMax3D`: Reduces the channel dimension by max, returning a
    single-channel volume. Useful for peak/activation aggregation.

All layers are compatible with Keras Core / `keras.ops` backends and can be
saved/loaded with `keras.models.save_model` / `keras.models.load_model`.
"""

import keras
import keras.ops as ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="Custom")
class GroupNormalization3D(Layer):
    """
    Group Normalization for 3D data.

    Normalizes features within channel groups across spatial dimensions,
    improving robustness to:
    - Imaging power level differences
    - Batch composition changes
    - Per-sample intensity variations

    This often aids generalization across datasets with heterogeneous
    acquisition settings.

    Parameters
    ----------
    groups : int, default 8
        Number of groups to split channels into. Must divide the channel
        count exactly.
    epsilon : float, default 1e-5
        Small constant for numerical stability when normalizing.
    center : bool, default True
        If True, learn and apply channel-wise bias (`beta`).
    scale : bool, default True
        If True, learn and apply channel-wise scale (`gamma`).

    Notes
    -----
    - Input shape: (N, D, H, W, C)
    - Output shape: (N, D, H, W, C)
    - Serialization: Registered with Keras via `package="Custom"`.
    """

    def __init__(self, groups=8, epsilon=1e-5, center=True, scale=True, **kwargs):
        """Initialize the layer.

        See the class docstring for parameter details.
        """
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        """Create parameters (`gamma`, `beta`) and validate shape.

        Parameters
        ----------
        input_shape : tuple
            Expected as (N, D, H, W, C). `C` must be divisible by `groups`.

        Raises
        ------
        ValueError
            If the number of channels is not divisible by `groups`.
        """
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
        """Apply group normalization.

        Parameters
        ----------
        inputs : Tensor
            5D tensor of shape (N, D, H, W, C).

        Returns
        -------
        Tensor
            Normalized tensor of the same shape as inputs.

        Notes
        -----
        - Splits channels into `groups`; computes mean/variance across
          spatial dims and channels-per-group, then normalizes.
        - Applies optional affine transform (`gamma`, `beta`).
        """
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
        """Return the output shape (unchanged).

        Parameters
        ----------
        input_shape : tuple
            Input shape (N, D, H, W, C).

        Returns
        -------
        tuple
            Same as `input_shape`.
        """
        return input_shape

    def get_config(self):
        """Return serializable config for saving/loading the layer."""
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
    """Reduce mean along the channel axis.

    Aggregates multi-channel 3D inputs into a single-channel volume by
    computing the mean over the channel dimension.

    Notes
    -----
    - Input shape: (N, D, H, W, C)
    - Output shape: (N, D, H, W, 1)
    - Serialization: Registered with Keras via `package="Custom"`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        """Compute mean over channels.

        Parameters
        ----------
        inputs : Tensor
            5D tensor of shape (N, D, H, W, C).
        mask : Any, optional
            Unused; present for API compatibility.

        Returns
        -------
        Tensor
            5D tensor of shape (N, D, H, W, 1).
        """
        return ops.mean(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        """Return output shape with channels reduced to 1."""
        return input_shape[:-1] + (1,)

    def get_config(self):
        """Return serializable config (no parameters)."""
        return super().get_config()


@keras.saving.register_keras_serializable(package="Custom")
class ReduceMax3D(Layer):
    """Reduce max along the channel axis.

    Aggregates multi-channel 3D inputs into a single-channel volume by
    computing the max over the channel dimension.

    Notes
    -----
    - Input shape: (N, D, H, W, C)
    - Output shape: (N, D, H, W, 1)
    - Serialization: Registered with Keras via `package="Custom"`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        """Compute max over channels.

        Parameters
        ----------
        inputs : Tensor
            5D tensor of shape (N, D, H, W, C).
        mask : Any, optional
            Unused; present for API compatibility.

        Returns
        -------
        Tensor
            5D tensor of shape (N, D, H, W, 1).
        """
        return ops.max(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        """Return output shape with channels reduced to 1."""
        return input_shape[:-1] + (1,)

    def get_config(self):
        """Return serializable config (no parameters)."""
        return super().get_config()
