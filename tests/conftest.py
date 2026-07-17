"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def sample_experiment_1() -> Path:
    return Path(__file__).resolve().parent / "sample_experiment1"


@pytest.fixture(scope="session")
def sample_experiment_2() -> Path:
    return Path(__file__).resolve().parent / "sample_experiment2"


@pytest.fixture()
def dataset(sample_experiment_1):
    from datakit import Dataset
    return Dataset.from_directory(sample_experiment_1)


@pytest.fixture()
def multi_dataset(sample_experiment_1, sample_experiment_2):
    from datakit import Dataset
    return Dataset.from_directory([sample_experiment_1, sample_experiment_2])
