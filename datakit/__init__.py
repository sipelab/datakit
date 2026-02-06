"""Convenience bootstrap for the ``datakit`` package.

The module wires up a namespaced logger and re-exports high-level entry
points so exploratory notebooks can simply ``import datakit`` and start
interacting with :class:`datakit.experiment.ExperimentData` without extra
ceremony.
"""

from __future__ import annotations

import logging
import os
from typing import Final

_LOGGER_NAME: Final[str] = "datakit"

logger = logging.getLogger(_LOGGER_NAME)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

if os.getenv("DATAKIT_SAFE_MODE") == "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from .experiment import ExperimentData, ExperimentMetadata  # noqa: E402  (import after logger to avoid circular import)
from .discover import discover_manifest  # noqa: E402  (import after logger to avoid circular import)
from .loader import ExperimentStore  # noqa: E402  (import after logger to avoid circular import)
from .config import settings  # noqa: E402  (import after logger to avoid circular import)


__all__ = [
    "logger",
    "ExperimentData",
    "ExperimentMetadata",
    "discover_manifest",
    "ExperimentStore",
    "settings",
]
