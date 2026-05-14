"""
Tests for aind_smartspim_classification.model.layers module.

Tests run on CPU (no CUDA required) by relying on the torch Keras backend.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
from aind_smartspim_classification.model.layers import (  # noqa: E402
    GroupNormalization3D,
    ReduceMax3D,
    ReduceMean3D,
)

BATCH, D, H, W, C = 2, 4, 4, 4, 8


def _ones(shape):
    """Helper to create a keras tensor of ones with the given shape."""
    return np.ones(shape, dtype=np.float32)


class TestGroupNormalization3D:
    """Tests for GroupNormalization3D layer that implements group normalization for 3D inputs."""

    def test_output_shape(self):
        """Output shape should equal input shape (N, D, H, W, C)."""
        layer = GroupNormalization3D(groups=4)
        x = _ones((BATCH, D, H, W, C))
        y = layer(x)
        assert y.shape == (BATCH, D, H, W, C)

    def test_normalizes_to_near_zero_mean(self):
        """Group-normalized output should have approximately zero mean within each group."""
        layer = GroupNormalization3D(groups=4)
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 2.0, (1, D, H, W, C)).astype(np.float32)
        y = np.array(layer(x))
        assert abs(y.mean()) < 0.5

    def test_raises_on_indivisible_channels(self):
        """ValueError when channel count is not divisible by number of groups."""
        layer = GroupNormalization3D(groups=3)
        with pytest.raises(ValueError, match="divisible"):
            layer(_ones((1, D, H, W, C)))  # C=8, groups=3 → not divisible

    def test_config_roundtrip(self):
        """get_config / from_config should reconstruct identical hyperparameters."""
        layer = GroupNormalization3D(groups=4, epsilon=1e-3, center=False, scale=True)
        cfg = layer.get_config()
        restored = GroupNormalization3D.from_config(cfg)
        assert restored.groups == layer.groups
        assert restored.epsilon == pytest.approx(layer.epsilon)
        assert restored.center == layer.center
        assert restored.scale == layer.scale

    def test_trainable_weights_created(self):
        """Layer should have gamma and beta weights after build."""
        layer = GroupNormalization3D(groups=4)
        layer(_ones((1, D, H, W, C)))  # triggers build
        weight_names = [w.name for w in layer.weights]
        assert any("gamma" in n for n in weight_names)
        assert any("beta" in n for n in weight_names)

    def test_no_scale_no_gamma(self):
        """When scale=False, gamma should not be created."""
        layer = GroupNormalization3D(groups=4, scale=False, center=False)
        layer(_ones((1, D, H, W, C)))
        assert layer.gamma is None
        assert layer.beta is None


class TestReduceMean3D:
    """Tests for ReduceMean3D layer that computes mean across channel dimension for 3D inputs."""

    def test_output_shape(self):
        """Output should collapse channel dim to 1: (N, D, H, W, 1)."""
        layer = ReduceMean3D()
        y = layer(_ones((BATCH, D, H, W, C)))
        assert y.shape == (BATCH, D, H, W, 1)

    def test_correct_mean_values(self):
        """Output should equal the per-voxel channel mean."""
        layer = ReduceMean3D()
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, (1, D, H, W, C)).astype(np.float32)
        y = np.array(layer(x))
        expected = x.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_compute_output_shape(self):
        """compute_output_shape should shrink last dim to 1."""
        layer = ReduceMean3D()
        out_shape = layer.compute_output_shape((BATCH, D, H, W, C))
        assert out_shape == (BATCH, D, H, W, 1)

    def test_config_roundtrip(self):
        """get_config should be serializable (no extra params)."""
        layer = ReduceMean3D(name="mean_layer")
        cfg = layer.get_config()
        restored = ReduceMean3D.from_config(cfg)
        assert restored.name == layer.name


class TestReduceMax3D:
    """Tests for ReduceMax3D layer that computes max across channel dimension for 3D inputs."""

    def test_output_shape(self):
        """Output should collapse channel dim to 1: (N, D, H, W, 1)."""
        layer = ReduceMax3D()
        y = layer(_ones((BATCH, D, H, W, C)))
        assert y.shape == (BATCH, D, H, W, 1)

    def test_correct_max_values(self):
        """Output should equal the per-voxel channel max."""
        layer = ReduceMax3D()
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 1, (1, D, H, W, C)).astype(np.float32)
        y = np.array(layer(x))
        expected = x.max(axis=-1, keepdims=True)
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_compute_output_shape(self):
        """compute_output_shape should shrink last dim to 1."""
        layer = ReduceMax3D()
        out_shape = layer.compute_output_shape((BATCH, D, H, W, C))
        assert out_shape == (BATCH, D, H, W, 1)

    def test_max_geq_mean(self):
        """Channel-max must always be >= channel-mean for non-negative inputs."""
        layer_max = ReduceMax3D()
        layer_mean = ReduceMean3D()
        x = np.abs(np.random.default_rng(3).normal(0, 1, (2, D, H, W, C))).astype(
            np.float32
        )
        y_max = np.array(layer_max(x))
        y_mean = np.array(layer_mean(x))
        assert np.all(y_max >= y_mean - 1e-6)

    def test_config_roundtrip(self):
        """get_config should be serializable (no extra params)."""
        layer = ReduceMax3D(name="max_layer")
        cfg = layer.get_config()
        restored = ReduceMax3D.from_config(cfg)
        assert restored.name == layer.name
