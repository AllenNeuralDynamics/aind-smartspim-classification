"""
Serializable focal loss classes for model saving.

This module includes fully serializable implementations of binary and
categorical focal loss, compatible with Keras Core (`keras.ops`) and
registered via `keras.saving.register_keras_serializable`.

"""

import keras
import keras.ops as ops


@keras.saving.register_keras_serializable(package="Custom")
class BinaryFocalLoss(keras.losses.Loss):
    """
    Binary focal loss.

    Parameters
    ----------
    gamma : float, default 2.0
        Focusing parameter. Larger values reduce the loss contribution
        from well-classified examples: $(1 - p_t)^\gamma$.
    alpha : float, default 0.25
        Class balancing factor applied to positive vs negative classes.

    Notes
    -----
    - `y_true` and `y_pred` are expected to be broadcastable to the same
      shape. `y_pred` should be probabilities in (0, 1).
    - Registered as a serializable custom loss for model saving/loading.
    """

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute the binary focal loss.

        Parameters
        ----------
        y_true : Tensor
            Binary targets. Shape broadcastable to `y_pred`.
        y_pred : Tensor
            Predicted probabilities in (0, 1). Shape broadcastable to `y_true`.

        Returns
        -------
        Tensor
            Element-wise loss values, reducible by Keras depending on the
            configured loss reduction.
        """
        y_pred = ops.cast(y_pred, "float32")
        y_true = ops.cast(y_true, "float32")

        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Binary cross entropy
        bce = -(y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))

        # Focal weight: (1 - pt)^gamma
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = ops.power(1.0 - pt, self.gamma)

        # Alpha balancing
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        return focal_weight * alpha_weight * bce

    def get_config(self):
        """Return config for serialization (gamma and alpha)."""
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="Custom")
class CategoricalFocalLoss(keras.losses.Loss):
    """
    Categorical focal loss.

    Parameters
    ----------
    gamma : float, default 2.0
        Focusing parameter. Larger values reduce the loss contribution
        from well-classified examples: $(1 - p_t)^\gamma$.
    alpha : float or sequence, default 0.25
        Class balancing factor. If a list/tuple is provided, it should match
        the number of classes and will be applied per-class.

    Notes
    -----
    - `y_true` should be one-hot encoded. `y_pred` should contain
      probabilities per class and sum to 1 across the class dimension.
    - Registered as a serializable custom loss for model saving/loading.
    """

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute the categorical focal loss.

        Parameters
        ----------
        y_true : Tensor
            One-hot targets. Same shape as `y_pred`.
        y_pred : Tensor
            Predicted class probabilities in (0, 1). Same shape as `y_true`.

        Returns
        -------
        Tensor
            Per-example loss values reduced over the class axis.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        epsilon = keras.backend.epsilon()
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy
        ce = -y_true * ops.log(y_pred)

        # pt
        pt = ops.sum(y_true * y_pred, axis=-1, keepdims=True)

        # Focal weight
        focal_weight = ops.power(1.0 - pt, self.gamma)
        loss = focal_weight * ce

        # Alpha weighting
        if isinstance(self.alpha, (list, tuple)):
            alpha_tensor = ops.convert_to_tensor(self.alpha, dtype="float32")
            alpha_weight = y_true * alpha_tensor
            loss = alpha_weight * loss
        else:
            loss = self.alpha * loss

        return ops.sum(loss, axis=-1)

    def get_config(self):
        """Return config for serialization (gamma and alpha)."""
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
            }
        )
        return config
