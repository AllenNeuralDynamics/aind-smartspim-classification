"""
Seriazable loss classes for model saving
"""

import keras
import keras.ops as ops


@keras.saving.register_keras_serializable(package="Custom")
class BinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss as a proper Keras Loss class.
    Fully serializable - fixes the functools.partial error.
    """

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute binary focal loss."""
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
        """Return config for serialization."""
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
    Categorical Focal Loss as a proper Keras Loss class.
    Fully serializable - fixes the functools.partial error.
    """

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute categorical focal loss."""
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
        """Return config for serialization."""
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
            }
        )
        return config
