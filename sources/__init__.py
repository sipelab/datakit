"""
DataKit data source implementations organized by category.

This module contains all data source implementations that can load various
types of experimental data. Data sources are automatically registered with
the DataSource registry and support versioning.
"""

# Import all data sources to ensure they register with the DataSource registry
from .camera.mesoscope import MesoMetadataSource
from .camera.pupil import PupilMetadataSource
from .camera.suite2p import Suite2p
from .camera.suite2pV2 import Suite2pV2
from .behavior.treadmill import TreadmillSource
from .behavior.dataqueue import DataqueueSource
from .behavior.wheel import WheelEncoder
from .behavior.psychopy import Psychopy
from .analysis.mesoscope import MesoMeanSource
from .analysis.mesomap import MesoMapSource
from .analysis.pupil import PupilDLCSource
from .session.config import SessionConfigSource
from .session.notes import SessionNotesSource
from .session.timestamps import SessionTimestampsSource

__all__ = [
    "MesoMetadataSource",
    "PupilMetadataSource",
    "Suite2p",
    "Suite2pV2",
    "TreadmillSource",
    "DataqueueSource",
    "WheelEncoder",
    "Psychopy",
    "MesoMeanSource",
    "MesoMapSource",
    "PupilDLCSource",
    "SessionConfigSource",
    "SessionNotesSource",
    "SessionTimestampsSource",
]