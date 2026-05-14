"""
Tests for aind_smartspim_classification.utils.utils module.

Focuses on pure-Python/numpy functions and those with well-defined
I/O contracts that can be exercised without a GPU or Zarr dataset.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest
from aind_smartspim_classification.utils.utils import (
    check_path_instance,
    create_folder,
    get_cpu_limit,
    get_memory_limit_bytes,
    get_size,
    read_json_as_dict,
)


class TestGetSize:
    """Tests for get_size function that converts bytes to human-readable format."""

    @pytest.mark.parametrize(
        "n_bytes, expected",
        [
            (500, "500.00B"),
            (1024, "1.00KB"),
            (1024**2, "1.00MB"),
            (1024**3, "1.00GB"),
            (1024**4, "1.00TB"),
            (1024**5, "1.00PB"),
        ],
    )
    def test_unit_conversions(self, n_bytes, expected):
        """Should correctly convert bytes to the appropriate unit string."""
        assert get_size(n_bytes) == expected

    def test_fractional_megabytes(self):
        """Should correctly convert fractional megabytes to MB string."""
        result = get_size(int(1.5 * 1024**2))
        assert result.endswith("MB")


class TestReadJsonAsDict:
    """Tests for read_json_as_dict function that reads a JSON file into a dict."""

    def test_reads_valid_json(self, tmp_path):
        """Should read a valid JSON file and return the corresponding dict."""
        data = {"key": "value", "number": 42}
        filepath = tmp_path / "test.json"
        filepath.write_text(json.dumps(data))
        result = read_json_as_dict(str(filepath))
        assert result == data

    def test_returns_empty_dict_for_missing_file(self):
        """Should return an empty dict if the file does not exist, without raising."""
        result = read_json_as_dict("/nonexistent/path/file.json")
        assert result == {}

    def test_handles_nested_dict(self, tmp_path):
        """Should correctly read nested dictionaries from JSON."""
        data = {"a": {"b": {"c": 123}}}
        filepath = tmp_path / "nested.json"
        filepath.write_text(json.dumps(data))
        result = read_json_as_dict(str(filepath))
        assert result["a"]["b"]["c"] == 123


class TestCreateFolder:
    """Tests for create_folder function that creates a directory if it doesn't exist."""

    def test_creates_new_directory(self, tmp_path):
        """Should create a new directory if it doesn't exist."""
        new_dir = str(tmp_path / "new_subdir")
        assert not os.path.exists(new_dir)
        create_folder(new_dir)
        assert os.path.isdir(new_dir)

    def test_idempotent_on_existing_directory(self, tmp_path):
        """Calling create_folder on an existing dir should not raise."""
        create_folder(str(tmp_path))  # already exists — must not raise

    def test_creates_nested_directories(self, tmp_path):
        """Should create nested directories if they don't exist."""
        nested = str(tmp_path / "a" / "b" / "c")
        create_folder(nested)
        assert os.path.isdir(nested)


class TestGetCpuLimit:
    """Tests for get_cpu_limit function that determines CPU limits from env vars or psutil."""

    def test_returns_co_cpus_env_var(self):
        """Should return CO_CPUS environment variable if set, without raising."""
        with patch.dict(os.environ, {"CO_CPUS": "8"}, clear=False):
            result = get_cpu_limit()
        assert result == "8"

    def test_aws_batch_returns_one(self):
        """If AWS_BATCH_JOB_ID is set, should return 1 regardless of other env vars or files."""
        env = {"AWS_BATCH_JOB_ID": "some-job-id"}
        # Make sure CO_CPUS is absent so the AWS_BATCH branch is reached
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CO_CPUS", None)
            result = get_cpu_limit()
        assert result == 1

    def test_slurm_cpus_env_var(self):
        """Should return SLURM_JOB_CPUS_PER_NODE environment variable if set, without raising."""
        env = {"SLURM_JOB_CPUS_PER_NODE": "16"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CO_CPUS", None)
            os.environ.pop("AWS_BATCH_JOB_ID", None)
            result = get_cpu_limit()
        assert result == "16"

    def test_falls_back_to_psutil(self):
        """When no env vars are set and cgroup files are absent, use psutil."""
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CO_CPUS", "AWS_BATCH_JOB_ID", "SLURM_JOB_CPUS_PER_NODE")
        }
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("builtins.open", side_effect=FileNotFoundError):
                result = get_cpu_limit()
        # Result is either int (cpu count) or 0 — just check it's an int
        assert isinstance(result, int)


class TestGetMemoryLimitBytes:
    """Tests for get_memory_limit_bytes function that determines memory limits from env vars or psutil."""

    def test_co_memory_env_var(self):
        """Should return CO_MEMORY environment variable as an int number of bytes if set, without raising."""
        with patch.dict(os.environ, {"CO_MEMORY": "16"}, clear=False):
            result = get_memory_limit_bytes()
        assert result == 16

    def test_slurm_mem_per_node(self):
        """If SLURM_MEM_PER_NODE is set, should return that value converted from MB to bytes, without raising."""
        env = {"SLURM_MEM_PER_NODE": "4096"}  # 4 GB in MB
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CO_MEMORY", None)
            with patch("builtins.open", side_effect=FileNotFoundError):
                result = get_memory_limit_bytes()
        assert result == 4096 * 1024**2

    def test_falls_back_to_psutil(self):
        """When no env vars are set, falls back to psutil.virtual_memory().total."""

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CO_MEMORY", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")
        }
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("builtins.open", side_effect=FileNotFoundError):
                result = get_memory_limit_bytes()
        assert result == psutil.virtual_memory().total

    def test_invalid_co_memory_falls_through(self):
        """Invalid CO_MEMORY value should fall through to next method."""
        with patch.dict(os.environ, {"CO_MEMORY": "not_a_number"}, clear=False):
            with patch("builtins.open", side_effect=FileNotFoundError):
                result = get_memory_limit_bytes()
        # Should not raise; result is some valid int
        assert isinstance(result, int)
        assert result > 0


class TestCheckPathInstance:
    """Tests for check_path_instance function that checks if input is a Path instance."""

    def test_posix_path_is_path_instance(self):
        """Should return True for a pathlib.Path instance."""
        assert check_path_instance(Path("/some/path")) is True

    def test_string_is_not_path_instance(self):
        """Should return False for a string, even if it looks like a path."""
        assert check_path_instance("/some/path") is False

    def test_integer_is_not_path_instance(self):
        """Should return False for a non-string, non-Path input."""
        assert check_path_instance(42) is False
