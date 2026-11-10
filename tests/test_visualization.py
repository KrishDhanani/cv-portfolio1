"""Starter tests for visualization utilities."""

from src.data.dataset_utils import repository_root


def test_repository_root_exists():
    assert repository_root().exists()
