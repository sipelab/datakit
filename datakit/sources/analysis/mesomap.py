"""Mesomap trace loader.

Parses wide-field mesomap traces exported as ``*_mesoscope.ome_traces.csv``
files. The accompanying ``.mask.npy`` and ``.regions.csv`` files are treated
as session-level metadata so they can be stored once per subject/session
instead of repeating for every task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from datakit.config import settings
from datakit.datamodel import LoadedStream
from datakit.sources.register import DataSource


class MesoMapSource(DataSource):
	"""Load mesomap trace CSVs plus optional mask/region metadata."""

	tag = "mesomap"
	patterns = ("**/*_mesoscope.ome_traces.csv",)
	camera_tag = "meso_metadata"
	version = "1.0"
	flatten_payload = True
	timeline_columns = ("frame",)
 
	mask_suffix = ".mask.npy"
	regions_suffix = ".regions.csv"

	frame_column = "frame"
	time_basis = "mesomap_frame"

	def load(self, path: Path) -> LoadedStream:
		trace_df = pd.read_csv(path)

		if self.frame_column not in trace_df.columns:
			raise ValueError(f"Mesomap traces missing required column '{self.frame_column}' in {path}")

		trace_df = trace_df.copy()
		trace_df.sort_values(by=self.frame_column, inplace=True)

		trace_columns = [col for col in trace_df.columns if col != self.frame_column]
		timeline = trace_df[self.frame_column].to_numpy(dtype=np.float64)

		regions_df, roi_to_mask, missing_regions = self._load_regions(path, trace_columns)
		mask_info = self._load_mask(path)

		meta = {
			"source_file": str(path),
			"n_frames": len(trace_df),
			"n_rois": len(trace_columns),
			"trace_columns": trace_columns,
			"roi_to_mask_label": roi_to_mask,
			"roi_missing_region_metadata": missing_regions,
			"time_basis": self.time_basis,
			"scope": settings.dataset.session_scope,
			**mask_info,
		}

		if regions_df is not None:
			meta.update(
				{
					"regions_path": str(self._regions_path(path)),
					"region_count": len(regions_df),
					"regions_table": regions_df,
				}
			)

		return self._create_stream(self.tag, timeline, trace_df, meta=meta)

	def _mask_path(self, path: Path) -> Path:
		return path.with_suffix(self.mask_suffix)

	def _regions_path(self, path: Path) -> Path:
		return path.with_suffix(self.regions_suffix)

	def _load_mask(self, path: Path) -> dict:
		mask_path = self._mask_path(path)
		if not mask_path.exists():
			return {
				"mask_path": str(mask_path),
				"mask_missing": True,
			}

		mask_array = np.load(mask_path)
		unique_labels = np.unique(mask_array)
		return {
			"mask_path": str(mask_path),
			"mask_shape": tuple(int(dim) for dim in mask_array.shape),
			"mask_dtype": str(mask_array.dtype),
			"mask_labels": unique_labels.tolist(),
			"mask_missing": False,
		}

	def _load_regions(
		self, path: Path, trace_columns: Iterable[str]
	) -> Tuple[pd.DataFrame | None, dict[str, int], list[str]]:
		regions_path = self._regions_path(path)
		if not regions_path.exists():
			return None, {}, list(trace_columns)

		regions_df = pd.read_csv(regions_path)

		required_cols = {"acronym", "mask_label"}
		if not required_cols.issubset(regions_df.columns):
			missing = ", ".join(sorted(required_cols - set(regions_df.columns)))
			raise ValueError(f"Regions file {regions_path} missing required columns: {missing}")

		region_lookup = regions_df.set_index("acronym")["mask_label"]
		roi_to_mask = {
			roi: int(region_lookup[roi]) for roi in trace_columns if roi in region_lookup
		}
		missing_regions = sorted(set(trace_columns) - set(region_lookup.index))

		return regions_df, roi_to_mask, missing_regions
