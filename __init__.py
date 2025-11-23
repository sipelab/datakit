"""Convenience bootstrap for the ``datakit`` package.

The module wires up a namespaced logger and re-exports high-level entry
points so exploratory notebooks can simply ``import datakit`` and start
interacting with :class:`datakit.experiment.ExperimentData` without extra
ceremony.
"""

from __future__ import annotations

import logging
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

__all__ = ["logger", "ExperimentData"]

from .experiment import ExperimentData  # noqa: E402  (import after logger to avoid circular import)
