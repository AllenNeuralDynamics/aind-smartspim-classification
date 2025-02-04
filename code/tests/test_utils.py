"""Test module for utils"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

sys.path.append("../")

from aind_smartspim_classification.utils.utils import (
    create_folder, generate_processing, get_code_ocean_cpu_limit,
    read_json_as_dict)


class TestUtilities(unittest.TestCase):
    """
    Test utilities
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        # Folder to test the zarr writing from PNGs
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

    @patch("os.environ.get")
    @patch("psutil.cpu_count")
    def test_get_code_ocean_cpu_limit(self, mock_cpu_count, mock_env_get):
        """
        Tests we get the code ocean CPU limits if
        it's a code ocean instance
        """
        mock_env_get.side_effect = lambda x: "4" if x == "CO_CPUS" else None
        mock_cpu_count.return_value = 8

        self.assertEqual(get_code_ocean_cpu_limit(), "4")

        mock_env_get.side_effect = lambda x: None
        with patch("builtins.open", mock_open(read_data="100000")) as mock_file:
            self.assertEqual(get_code_ocean_cpu_limit(), 1)

        mock_file.side_effect = FileNotFoundError
        self.assertEqual(get_code_ocean_cpu_limit(), 8)

    @patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "job_id"}, clear=True)
    def test_get_code_ocean_cpu_limit_aws_batch(self):
        """
        Tests the case where it's a pipeline execution
        """
        self.assertEqual(get_code_ocean_cpu_limit(), 1)

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("os.path.exists", return_value=True)
    def test_read_json_as_dict_valid(self, mock_exists, mock_open):
        """
        Test read a valid citionari
        """
        result = read_json_as_dict("fake_path.json")
        self.assertEqual(result, {"key": "value"})

    def test_create_folder(self):
        """
        Tests the creation of a folder
        """
        with patch("os.makedirs") as mock_makedirs:
            create_folder("mock_folder", verbose=True)
            mock_makedirs.assert_called_once()

    def test_generate_processing(self):
        """
        Tests that we generate the processing manifest
        """
        generate_processing(
            data_processes=[],
            dest_processing=self.temp_folder,
            processor_full_name="Test User",
            pipeline_version="1.0",
        )

        processing_path = Path(self.temp_folder).joinpath("processing.json")

        self.assertEqual(processing_path.exists(), True)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)
