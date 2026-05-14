"""
Tests for aind_smartspim_classification.model.losses module.

Tests run on CPU (no CUDA required).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import keras.ops as ops  # noqa: E402
from aind_smartspim_classification.model.losses import (  # noqa: E402
    BinaryFocalLoss,
    CategoricalFocalLoss,
)


def _np(tensor):
    """Convert keras tensor to numpy."""
    return np.array(tensor)


class TestBinaryFocalLoss:
    """Tests for BinaryFocalLoss class that implements focal loss for binary classification."""

    def test_perfect_prediction_near_zero(self):
        """Perfect predictions (pt≈1) should yield near-zero focal loss."""
        loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
        y_true = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        # Near-perfect: 1→0.999, 0→0.001
        y_pred = np.array([[0.999, 0.001, 0.999, 0.001]], dtype=np.float32)
        loss = _np(loss_fn(y_true, y_pred))
        assert np.all(loss < 1e-3)

    def test_worst_prediction_large_loss(self):
        """Worst predictions (pt≈0) should yield large focal loss."""
        loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
        y_true = np.array([[1.0, 0.0]], dtype=np.float32)
        y_pred = np.array([[0.001, 0.999]], dtype=np.float32)
        loss = _np(loss_fn(y_true, y_pred))
        # Mean reduction is applied by Keras; value should be clearly positive
        assert float(np.mean(loss)) > 0.0

    def test_gamma_zero_equals_weighted_bce(self):
        """With gamma=0, focal loss reduces to alpha-weighted binary cross-entropy."""
        alpha = 0.5
        loss_focal = BinaryFocalLoss(gamma=0.0, alpha=alpha)
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, (4, 8)).astype(np.float32)
        y_pred = rng.uniform(0.1, 0.9, (4, 8)).astype(np.float32)

        focal_val = float(_np(ops.mean(loss_focal(y_true, y_pred))))

        eps = keras.backend.epsilon()
        y_pred_c = np.clip(y_pred, eps, 1.0 - eps)
        bce = -(y_true * np.log(y_pred_c) + (1 - y_true) * np.log(1 - y_pred_c))
        alpha_w = y_true * alpha + (1 - y_true) * (1 - alpha)
        bce_val = float(np.mean(alpha_w * bce))

        assert focal_val == pytest.approx(bce_val, rel=1e-4)

    def test_higher_gamma_reduces_easy_loss(self):
        """Higher gamma should reduce the loss for well-classified (easy) examples."""
        y_true = np.array([[1.0, 1.0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.9]], dtype=np.float32)

        loss_low = float(_np(ops.mean(BinaryFocalLoss(gamma=0.5)(y_true, y_pred))))
        loss_high = float(_np(ops.mean(BinaryFocalLoss(gamma=5.0)(y_true, y_pred))))
        assert loss_high < loss_low

    def test_config_roundtrip(self):
        """get_config / from_config should reconstruct gamma and alpha."""
        loss_fn = BinaryFocalLoss(gamma=3.0, alpha=0.4)
        cfg = loss_fn.get_config()
        restored = BinaryFocalLoss.from_config(cfg)
        assert restored.gamma == pytest.approx(loss_fn.gamma)
        assert restored.alpha == pytest.approx(loss_fn.alpha)

    def test_output_is_finite(self):
        """Loss output should never be NaN or Inf."""
        loss_fn = BinaryFocalLoss()
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, (8, 16)).astype(np.float32)
        y_pred = rng.uniform(0.0, 1.0, (8, 16)).astype(np.float32)
        loss = _np(loss_fn(y_true, y_pred))
        assert np.all(np.isfinite(loss))


class TestCategoricalFocalLoss:
    """Tests for CategoricalFocalLoss class that implements focal loss for multi-class classification."""

    def test_perfect_one_hot_near_zero(self):
        """Perfect one-hot predictions should yield near-zero loss."""
        loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=0.25)
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        y_pred = np.array(
            [[0.998, 0.001, 0.001], [0.001, 0.998, 0.001]], dtype=np.float32
        )
        loss = _np(loss_fn(y_true, y_pred))
        assert float(np.mean(loss)) < 0.01

    def test_worst_prediction_positive_loss(self):
        """Loss should be clearly positive when predictions are completely wrong."""
        loss_fn = CategoricalFocalLoss(gamma=2.0)
        y_true = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        y_pred = np.array([[0.001, 0.499, 0.500]], dtype=np.float32)
        loss = _np(loss_fn(y_true, y_pred))
        assert float(np.mean(loss)) > 0.0

    def test_per_class_alpha(self):
        """Scalar and list alpha should produce different (but valid) outputs."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y_pred = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)

        loss_scalar = float(
            _np(ops.mean(CategoricalFocalLoss(alpha=0.25)(y_true, y_pred)))
        )
        loss_list = float(
            _np(ops.mean(CategoricalFocalLoss(alpha=[0.1, 0.9])(y_true, y_pred)))
        )
        # They should differ because alpha balances are different
        assert loss_scalar != pytest.approx(loss_list, rel=1e-2)

    def test_config_roundtrip_scalar_alpha(self):
        """Scalar alpha is serializable via get_config."""
        loss_fn = CategoricalFocalLoss(gamma=1.5, alpha=0.3)
        cfg = loss_fn.get_config()
        restored = CategoricalFocalLoss.from_config(cfg)
        assert restored.gamma == pytest.approx(loss_fn.gamma)
        assert restored.alpha == pytest.approx(loss_fn.alpha)

    def test_config_roundtrip_list_alpha(self):
        """List alpha is serializable via get_config."""
        loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=[0.2, 0.5, 0.3])
        cfg = loss_fn.get_config()
        restored = CategoricalFocalLoss.from_config(cfg)
        assert restored.alpha == loss_fn.alpha

    def test_output_is_finite(self):
        """Loss output should never be NaN or Inf."""
        loss_fn = CategoricalFocalLoss()
        rng = np.random.default_rng(99)
        y_true_idx = rng.integers(0, 4, 16)
        y_true = np.eye(4)[y_true_idx].astype(np.float32)
        raw = rng.uniform(0.1, 1.0, (16, 4)).astype(np.float32)
        y_pred = raw / raw.sum(axis=1, keepdims=True)

        loss = _np(loss_fn(y_true, y_pred))
        assert np.all(np.isfinite(loss))
