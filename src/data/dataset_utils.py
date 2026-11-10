"""Small reusable helpers for dataset paths and configuration."""

from pathlib import Path


def repository_root() -> Path:
    """Return the repository root based on this file's location."""
    return Path(__file__).resolve().parents[2]


def resolve_data_config(config_path: str = "configs/data.yaml") -> Path:
    """Resolve a config path relative to the repository root."""
    return repository_root() / config_path
