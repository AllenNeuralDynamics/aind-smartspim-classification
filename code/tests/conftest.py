"""
Shared pytest fixtures for aind-smartspim-classification tests.
"""

import logging
import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend so tests never open a display window
matplotlib.use("Agg")

# Set Keras backend before any keras import happens
os.environ.setdefault("KERAS_BACKEND", "torch")


@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_logger():
    """A real logger that discards all output."""
    logger = logging.getLogger("test")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def big_block():
    """A 4-channel 3D numpy block of shape (4, 10, 10, 10)."""
    return np.arange(4 * 10 * 10 * 10, dtype=np.float32).reshape(4, 10, 10, 10)


@pytest.fixture
def bimodal_likelihoods_df():
    """DataFrame with bimodal Cell Likelihood distribution (clear threshold near 0.5)."""
    rng = np.random.default_rng(42)
    low = rng.uniform(0.0, 0.3, 200)
    high = rng.uniform(0.7, 1.0, 200)
    likelihoods = np.concatenate([low, high])
    df = pd.DataFrame(
        {
            "x": rng.integers(0, 100, len(likelihoods)),
            "y": rng.integers(0, 100, len(likelihoods)),
            "z": rng.integers(0, 100, len(likelihoods)),
            "Cell Likelihood": likelihoods,
        }
    )
    return df


@pytest.fixture
def classified_df(bimodal_likelihoods_df):
    """DataFrame with a 'Class' column added (threshold at 0.5)."""
    df = bimodal_likelihoods_df.copy()
    df.insert(3, "Class", (df["Cell Likelihood"] >= 0.5).astype(int))
    return df
