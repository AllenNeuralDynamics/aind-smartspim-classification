"""
Tests cell classification
"""

import logging
import os
import shutil
import sys
import tempfile
import unittest
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.append("../")

from aind_smartspim_classification.classification import (
    cumulative_likelihoods, extract_centered_3d_block, merge_csv,
    upsample_position)


class TestCellClassification(unittest.TestCase):
    """
    Tests cell classification code
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Setup folders and data blocks"""

        os.environ["KERAS_BACKEND"] = "torch"

        cls.big_block = np.arange(4 * 10 * 10 * 10).reshape(4, 10, 10, 10)

        cls.test_dir = tempfile.mkdtemp(prefix="unittest_")
        cls.save_dir = tempfile.mkdtemp(prefix="unittest_")

        os.makedirs(f"{cls.save_dir}/proposals", exist_ok=True)

        # Create sample CSV files
        df1 = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [3, 4],
                "z": [5, 6],
                "Class": [1, 0],
                "Cell Likelihood": [0.9, 0.1],
            }
        )
        df2 = pd.DataFrame(
            {
                "x": [7, 8],
                "y": [9, 10],
                "z": [11, 12],
                "Class": [0, 1],
                "Cell Likelihood": [0.2, 0.8],
            }
        )

        df1.to_csv(os.path.join(cls.test_dir, "classified_block_1.csv"))
        df2.to_csv(os.path.join(cls.test_dir, "classified_block_2.csv"))

        cls.logger = MagicMock(spec=logging.Logger)  # Mock logger

    def test_extract_centered_block_no_padding(self):
        """Test extracting a block completely inside the big block."""
        print(os.environ)
        center = (5, 5, 5)
        size = (3, 3, 3)
        extracted = extract_centered_3d_block(self.big_block, center, size)
        self.assertEqual(extracted.shape, (3, 3, 3, 3))

    def test_extract_edge_case(self):
        """Test extracting a block that is exactly on the boundary."""
        center = (9, 9, 9)
        size = (3, 3, 3)
        extracted = extract_centered_3d_block(self.big_block, center, size, pad_value=0)
        self.assertEqual(extracted.shape, (3, 3, 3, 3))
        self.assertTrue(np.any(extracted == 0))

    def test_upsample_position(self):
        """Test upsampling positions."""
        position = [2, 3, 4]
        downsample_factor = (2,)
        upsampled = upsample_position(position, downsample_factor)
        expected = (8, 12, 16)
        self.assertEqual(upsampled, expected)

    def test_upsample_position_single_factor(self):
        """Test upsampling when downsample factor is 1."""
        position = [1, 2, 3]
        downsample_factor = (1,)
        upsampled = upsample_position(position, downsample_factor)
        expected = (2, 4, 6)
        self.assertEqual(upsampled, expected)

    @patch("glob.glob")  # Mock glob to avoid filesystem dependency
    @patch("pandas.read_csv")  # Mock pandas read_csv
    @patch("os.makedirs")  # Mock os.makedirs to avoid real directory creation
    def test_merge_csv(self, mock_makedirs, mock_read_csv, mock_glob):
        """Test merging CSV files into a single output."""
        mock_glob.return_value = [
            os.path.join(self.test_dir, "classified_block_1.csv"),
            os.path.join(self.test_dir, "classified_block_2.csv"),
        ]

        mock_read_csv.side_effect = lambda f, index_col: pd.read_csv(
            f, index_col=index_col
        )

        output_csv = merge_csv(self.test_dir, self.save_dir, self.logger)

        # Verify calls
        self.logger.info.assert_called()  # Check if logging happened
        self.assertTrue(os.path.exists(output_csv))  # Ensure output file is created

        df_merged = pd.read_csv(output_csv)
        self.assertEqual(df_merged.shape[0], 4)  # Check if rows were combined correctly

    @patch("pandas.read_csv")  # Mock pandas read_csv
    def test_cumulative_likelihoods(self, mock_read_csv):
        """Test computing cumulative likelihood metrics."""
        df_mock = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [5, 6, 7, 8],
                "z": [9, 10, 11, 12],
                "Class": [1, 1, 0, 0],
                "Cell Likelihood": [0.9, 0.8, 0.2, 0.3],
            }
        )

        mock_read_csv.return_value = df_mock

        cumulative_likelihoods(self.save_dir, self.logger)

        metrics_path = os.path.join(self.save_dir, "cell_likelihood_metrics.csv")
        self.assertTrue(os.path.exists(metrics_path))

        df_metrics = pd.read_csv(metrics_path)
        self.assertEqual(df_metrics.loc[0, "Cell Counts"], 2)
        self.assertEqual(df_metrics.loc[0, "Noncell Counts"], 2)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

        if os.path.exists(cls.save_dir):
            shutil.rmtree(cls.save_dir, ignore_errors=True)
