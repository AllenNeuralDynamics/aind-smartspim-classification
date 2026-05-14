"""
Tests for aind_smartspim_classification.classification module.

Covers pure-numpy and pandas functions that do not require a GPU,
a trained model, or Zarr image data.
"""

import os

import numpy as np
import pandas as pd
import pytest
from aind_smartspim_classification.classification import (
    calculate_threshold,
    cumulative_likelihoods,
    extract_centered_3d_block,
    merge_csv,
    upsample_position,
)


class TestExtractCentered3DBlock:
    """Tests for extract_centered_3d_block function."""

    def test_interior_block_correct_shape(self, big_block):
        """Block extracted from the interior should have exactly the requested size."""
        result = extract_centered_3d_block(big_block, center=(5, 5, 5), size=(3, 3, 3))
        assert result.shape == (4, 3, 3, 3)

    def test_interior_block_correct_values(self, big_block):
        """Values extracted from the interior should match the source array."""
        result = extract_centered_3d_block(big_block, center=(5, 5, 5), size=(2, 2, 2))
        expected = big_block[:, 4:6, 4:6, 4:6]
        np.testing.assert_array_equal(result, expected)

    def test_boundary_block_padded_shape(self, big_block):
        """Block at the corner should still return the requested size (padded)."""
        result = extract_centered_3d_block(big_block, center=(0, 0, 0), size=(3, 3, 3))
        assert result.shape == (4, 3, 3, 3)

    def test_boundary_padding_uses_pad_value(self, big_block):
        """Padding outside array bounds must use the specified pad_value."""
        result = extract_centered_3d_block(
            big_block, center=(0, 0, 0), size=(3, 3, 3), pad_value=-1
        )
        # The top-left-front corner is out-of-bounds, so top slice should contain -1
        assert np.any(result == -1)

    def test_default_pad_value_is_zero(self, big_block):
        """Default pad_value should be 0."""
        result = extract_centered_3d_block(big_block, center=(0, 0, 0), size=(3, 3, 3))
        # There must be padding (zeros) because center is at edge
        assert np.any(result == 0)

    def test_full_array_size(self, big_block):
        """Requesting the same size as the array should return the full array."""
        result = extract_centered_3d_block(
            big_block, center=(5, 5, 5), size=(10, 10, 10)
        )
        assert result.shape == (4, 10, 10, 10)


class TestUpsamplePosition:
    """Tests for upsample_position function."""

    def test_factor_one_returns_same(self):
        """Factor=1 means 2^1=2 multiplier."""
        z, y, x = np.array([10]), np.array([20]), np.array([30])
        uz, uy, ux = upsample_position([z, y, x], downsample_factor=1)
        assert int(uz) == 20
        assert int(uy) == 40
        assert int(ux) == 60

    def test_factor_two(self):
        """Factor=2 means 2^2=4 multiplier."""
        z, y, x = np.array([5]), np.array([5]), np.array([5])
        uz, uy, ux = upsample_position([z, y, x], downsample_factor=2)
        assert int(uz) == 20
        assert int(uy) == 20
        assert int(ux) == 20

    def test_factor_zero(self):
        """Factor=0 means 2^0=1, so coordinates are unchanged."""
        z, y, x = np.array([7]), np.array([8]), np.array([9])
        uz, uy, ux = upsample_position([z, y, x], downsample_factor=0)
        assert int(uz) == 7
        assert int(uy) == 8
        assert int(ux) == 9

    def test_output_dtype_is_uint32(self):
        """Upsampled coordinates should be uint32."""
        z, y, x = np.array([3]), np.array([4]), np.array([5])
        uz, uy, ux = upsample_position([z, y, x], downsample_factor=1)
        assert uz.dtype == np.uint32
        assert uy.dtype == np.uint32
        assert ux.dtype == np.uint32


class TestCalculateThreshold:
    """Tests for calculate_threshold function."""

    def test_returns_tuple_of_df_and_float(
        self, tmp_dir, mock_logger, bimodal_likelihoods_df
    ):
        """Function should return (DataFrame, float)."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        result_df, threshold = calculate_threshold(
            bimodal_likelihoods_df, tmp_dir, mock_logger
        )
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(threshold, float)

    def test_adds_class_column(self, tmp_dir, mock_logger, bimodal_likelihoods_df):
        """The returned DataFrame must have a 'Class' column."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        result_df, _ = calculate_threshold(bimodal_likelihoods_df, tmp_dir, mock_logger)
        assert "Class" in result_df.columns

    def test_class_column_is_binary(self, tmp_dir, mock_logger, bimodal_likelihoods_df):
        """Class column should contain only 0 and 1."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        result_df, _ = calculate_threshold(bimodal_likelihoods_df, tmp_dir, mock_logger)
        assert set(result_df["Class"].unique()).issubset({0, 1})

    def test_threshold_is_in_valid_range(
        self, tmp_dir, mock_logger, bimodal_likelihoods_df
    ):
        """Threshold value should be between 0 and 1."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        _, threshold = calculate_threshold(bimodal_likelihoods_df, tmp_dir, mock_logger)
        assert 0.0 <= threshold <= 1.0

    def test_saves_threshold_png(self, tmp_dir, mock_logger, bimodal_likelihoods_df):
        """A PNG visualisation should be saved to save_path/proposals/."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        calculate_threshold(bimodal_likelihoods_df, tmp_dir, mock_logger)
        png_path = os.path.join(tmp_dir, "proposals", "threshold_identification.png")
        assert os.path.exists(png_path)

    def test_fallback_min_catch_high(self, tmp_dir, mock_logger):
        """All-high likelihoods should fall back to min_catch_high threshold."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        df = pd.DataFrame({"x": [1], "y": [1], "z": [1], "Cell Likelihood": [0.99]})
        _, threshold = calculate_threshold(
            df, tmp_dir, mock_logger, min_catch_high=0.85
        )
        assert threshold == pytest.approx(0.85)

    def test_fallback_min_catch_low(self, tmp_dir, mock_logger):
        """All-low likelihoods should fall back to min_catch_low threshold."""
        os.makedirs(os.path.join(tmp_dir, "proposals"), exist_ok=True)
        df = pd.DataFrame({"x": [1], "y": [1], "z": [1], "Cell Likelihood": [0.01]})
        _, threshold = calculate_threshold(df, tmp_dir, mock_logger, min_catch_low=0.05)
        assert threshold == pytest.approx(0.05)


class TestMergeCsv:
    """Tests for merge_csv function."""

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_dir):
        """Set up metadata and save directories for tests."""
        self.metadata_path = tmp_dir
        self.save_path = os.path.join(tmp_dir, "output")
        os.makedirs(self.save_path)

    def _write_block_csv(self, filename, df):
        """Helper to write a classified_block_*.csv file in the metadata directory."""
        path = os.path.join(self.metadata_path, filename)
        df.to_csv(path)
        return path

    def _make_block_df(self, n=5, seed=0):
        """Helper to create a DataFrame with n rows and random coordinates and likelihoods."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "x": rng.integers(0, 100, n),
                "y": rng.integers(0, 100, n),
                "z": rng.integers(0, 100, n),
                "Cell Likelihood": rng.uniform(0, 1, n),
            }
        )

    def test_merges_multiple_csvs(self, mock_logger):
        """merge_csv should concatenate all classified_block_*.csv files."""
        self._write_block_csv("classified_block_000.csv", self._make_block_df(5, 0))
        self._write_block_csv("classified_block_001.csv", self._make_block_df(5, 1))

        _, cells_df, threshold = merge_csv(
            self.metadata_path, self.save_path, mock_logger
        )
        assert isinstance(cells_df, pd.DataFrame)
        assert isinstance(threshold, float)

    def test_saves_output_csv(self, mock_logger):
        """merge_csv should write cell_likelihoods.csv and detected_cells.csv."""
        self._write_block_csv("classified_block_000.csv", self._make_block_df(10, 42))

        output_csv, _, _ = merge_csv(self.metadata_path, self.save_path, mock_logger)
        assert os.path.exists(output_csv)
        assert os.path.exists(
            os.path.join(self.save_path, "proposals", "cell_likelihoods.csv")
        )

    def test_raises_on_no_csvs(self, mock_logger):
        """merge_csv should raise RuntimeError when no classified_block_*.csv exist."""
        with pytest.raises(RuntimeError, match="No classified block CSVs"):
            merge_csv(self.metadata_path, self.save_path, mock_logger)


class TestCumulativeLikelihoods:
    """Tests for cumulative_likelihoods function."""

    def test_saves_metrics_csv(self, tmp_dir, mock_logger, classified_df):
        """cumulative_likelihoods should write cell_likelihood_metrics.csv."""
        proposals_dir = os.path.join(tmp_dir, "proposals")
        os.makedirs(proposals_dir)
        classified_df.to_csv(os.path.join(proposals_dir, "cell_likelihoods.csv"))

        cumulative_likelihoods(threshold=0.5, save_path=tmp_dir, logger=mock_logger)

        metrics_path = os.path.join(tmp_dir, "cell_likelihood_metrics.csv")
        assert os.path.exists(metrics_path)

    def test_metrics_contain_expected_keys(self, tmp_dir, mock_logger, classified_df):
        """The metrics CSV should contain all six reporting columns."""
        proposals_dir = os.path.join(tmp_dir, "proposals")
        os.makedirs(proposals_dir)
        classified_df.to_csv(os.path.join(proposals_dir, "cell_likelihoods.csv"))

        cumulative_likelihoods(threshold=0.5, save_path=tmp_dir, logger=mock_logger)

        df_metrics = pd.read_csv(
            os.path.join(tmp_dir, "cell_likelihood_metrics.csv"), index_col=0
        )
        expected_cols = {
            "Cell Counts",
            "Cell Likelihood Mean",
            "Cell Likelihood STD",
            "Noncell Counts",
            "Noncell Likelihood Mean",
            "Noncell Likelihood STD",
            "Classification Threshold",
        }
        assert expected_cols.issubset(set(df_metrics.columns))

    def test_cell_counts_match_class_column(self, tmp_dir, mock_logger, classified_df):
        """Cell Counts in metrics should equal the number of rows with Class==1."""
        proposals_dir = os.path.join(tmp_dir, "proposals")
        os.makedirs(proposals_dir)
        classified_df.to_csv(os.path.join(proposals_dir, "cell_likelihoods.csv"))

        cumulative_likelihoods(threshold=0.5, save_path=tmp_dir, logger=mock_logger)

        df_metrics = pd.read_csv(
            os.path.join(tmp_dir, "cell_likelihood_metrics.csv"), index_col=0
        )
        expected_cells = int((classified_df["Class"] == 1).sum())
        assert int(df_metrics.loc["Metrics", "Cell Counts"]) == expected_cells

    def test_threshold_recorded_in_metrics(self, tmp_dir, mock_logger, classified_df):
        """Classification Threshold in metrics should match the passed threshold."""
        proposals_dir = os.path.join(tmp_dir, "proposals")
        os.makedirs(proposals_dir)
        classified_df.to_csv(os.path.join(proposals_dir, "cell_likelihoods.csv"))

        cumulative_likelihoods(threshold=0.42, save_path=tmp_dir, logger=mock_logger)

        df_metrics = pd.read_csv(
            os.path.join(tmp_dir, "cell_likelihood_metrics.csv"), index_col=0
        )
        assert df_metrics.loc["Metrics", "Classification Threshold"] == pytest.approx(
            0.42
        )
