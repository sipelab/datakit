"""Pupil DeepLabCut analysis data source.

Parses the nested pickle emitted by the DeepLabCut pipeline, reconstructs per-
frame landmark coordinates and confidences, and summarizes them as a convenient
time-indexed :class:`~datakit.datamodel.LoadedStream` with derived pupil
diameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from datakit import logger
from datakit.sources.register import DataSource
from datakit.datamodel import LoadedStream


class PupilDLCSource(DataSource):
    """Load DeepLabCut pupil analysis results and compute diameters.

    The loader unwraps DeepLabCut's dictionary-of-dictionaries format, converts
    the coordinates/confidence arrays into tidy pandas columns, and annotates
    each sample with an evenly spaced ``time_elapsed_s`` axis derived from the
    configured frame rate.  It additionally runs a small analysis pass that
    estimates pupil diameter in millimetres, which downstream dashboards can
    plot immediately without re-running post-processing code.
    """
    tag = "pupil_dlc"
    patterns = ("**/*_pupilDLC_*_full.pickle",)
    camera_tag = "pupil_metadata"  # Bind to pupil camera
    version = "1.0"
    default_frame_rate_hz = 20.0
    confidence_threshold = 0.7
    pixel_to_mm = 53.6
    dpi = 300
    landmark_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    metadata_key = "metadata"
    coordinates_key = "coordinates"
    confidence_key = "confidence"
    frame_index_name = "frame"
    warn_on_low_confidence = True
    
    def load(self, path: Path) -> LoadedStream:
        """Deserialize a DeepLabCut pickle and return a structured stream."""
        # Use the proven pupil analysis functions
        raw_df = self._load_deeplabcut_pickle(path)
        analyzed_df = self._analyze_pupil_data(raw_df)

        n_frames = len(analyzed_df)
        t = np.arange(n_frames, dtype=np.float64) / float(self.default_frame_rate_hz)
        analyzed_df = analyzed_df.copy()
        analyzed_df["time_elapsed_s"] = t

        return LoadedStream(
            tag=self.tag,
            t=t,
            value=analyzed_df,
            meta={"source_file": str(path), "n_frames": len(analyzed_df)}
        )
    
    def _load_deeplabcut_pickle(self, filepath: Path) -> pd.DataFrame:
        """Load DeepLabCut pickle output into standardized DataFrame."""
        data = pd.read_pickle(filepath)

        # Build dictionaries for coordinates and confidence
        coordinates_dict = {}
        confidence_dict = {}
        for frame_key, frame_data in data.items():
            if frame_key == self.metadata_key:
                continue  # Skip metadata entry
                
            # Extract coordinates and confidence from the nested structure
            coords_raw = frame_data.get(self.coordinates_key)
            conf_raw = frame_data.get(self.confidence_key)
            
            if coords_raw is not None and conf_raw is not None:
                # Handle the nested structure: coordinates is a tuple containing a list of arrays
                # confidence is a list of arrays
                if isinstance(coords_raw, tuple) and len(coords_raw) > 0:
                    coord_arrays = coords_raw[0]  # Extract the list from the tuple
                    # Stack the coordinate arrays into a single array of shape (n_landmarks, 2)
                    coordinates = np.vstack([arr.flatten()[:2] for arr in coord_arrays])
                else:
                    coordinates = coords_raw
                    
                if isinstance(conf_raw, list):
                    # Extract confidence values from the list of arrays
                    confidence = np.array([arr.flatten()[0] for arr in conf_raw])
                else:
                    confidence = conf_raw
                    
                coordinates_dict[frame_key] = coordinates
                confidence_dict[frame_key] = confidence

        # Create series from the dictionaries
        coords_series = pd.Series(coordinates_dict)
        conf_series = pd.Series(confidence_dict)

        # Create the DataFrame
        df = pd.DataFrame({
            self.coordinates_key: coords_series,
            self.confidence_key: conf_series,
        })
        df.index.name = self.frame_index_name
        
        return df
    
    def _analyze_pupil_data(
        self,
        pickle_data: pd.DataFrame,
        *,
        confidence_threshold: float | None = None,
        pixel_to_mm: float | None = None,
        dpi: int | None = None,
    ) -> pd.DataFrame:
        """
        Analyze pupil data from DeepLabCut output.

        This function processes a pandas DataFrame containing per-frame DeepLabCut outputs
        with 'coordinates' and 'confidence' columns, skipping an initial metadata row,
        and computes interpolated pupil diameters in millimetres.

        Steps
        -----
        1. Skip the first (metadata) row.
        2. Extract and convert 'coordinates' and 'confidence' to NumPy arrays.
        3. For each frame:
           - Squeeze arrays and validate dimensions.
           - Mark landmarks with confidence ≥ threshold.
           - Compute Euclidean distances for predefined landmark pairs.
           - Average valid distances as pupil diameter or assign NaN.
        4. Build a pandas Series of diameters, interpolate missing values, convert from pixels to mm.
        5. Reindex to include the metadata index, then drop the initial NaN to align with valid frames.

        Parameters
        ----------
        pickle_data : pandas.DataFrame
            Input DataFrame with an initial metadata row. Must contain:
            - 'coordinates': array-like of shape (n_points, 2) per entry
            - 'confidence': array-like of shape (n_points,) per entry
        confidence_threshold : float, optional
            Minimum confidence to include a landmark in diameter computation.
            Default is 0.7.
        pixel_to_mm : float, optional
            Conversion factor from pixels to millimetres.
            Default is 53.6.
        dpi : int, optional
            Dots-per-inch resolution (not used directly).
            Default is 300.

        Returns
        -------
        pandas.DataFrame
            One-column DataFrame ('pupil_diameter_mm') indexed by the input labels
            (excluding the metadata row), containing linearly interpolated
            pupil diameter measurements in millimetres.
        """
        import statistics as st
        import math
        
        def euclidean_distance(coord1, coord2):
            """Calculate the Euclidean distance between two points."""
            return math.dist(coord1, coord2)

        # 1) pull lists (no need to skip metadata row since we already filtered it in loader)
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        px_to_mm = self.pixel_to_mm if pixel_to_mm is None else pixel_to_mm
        dpi_value = self.dpi if dpi is None else dpi

        coords_list = pickle_data[self.coordinates_key].tolist()
        conf_list = pickle_data[self.confidence_key].tolist()
        
        # Return a warning if no confidence values are above the threshold
        if self.warn_on_low_confidence and not any(np.any(np.array(c) >= threshold) for c in conf_list):
            logger.warning(
                "PupilDLCSource: no confidence values above threshold",
                extra={
                    "phase": "pupil_dlc_analysis",
                    "threshold": threshold,
                    "dpi": dpi_value,
                },
            )
            
        # 2) to numpy arrays (they should already be numpy arrays from the loader)
        coords_arrs = coords_list
        conf_arrs = conf_list

        # DEBUG: print first 3 shapes
        # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
        #     print(f"[DEBUG] frame {idx} coords.shape={c.shape}, conf.shape={f.shape}")
            
        # Print the first few values of c and f
        # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
        #     print(f"[DEBUG] frame {idx} coords values:\n{c}")
        #     print(f"[DEBUG] frame {idx} conf values:\n{f}")
            
        # 3) compute mean diameters
        diameters = []
        for i, (coords, conf) in enumerate(zip(coords_arrs, conf_arrs)):
            pts   = np.squeeze(coords)   # expect (n_points, 2)
            cvals = np.squeeze(conf)     # expect (n_points,)
            # DEBUG unexpected shapes
            if pts.ndim != 2 or cvals.ndim != 1:
                if self.warn_on_low_confidence:
                    logger.warning(
                        "PupilDLCSource: unexpected coordinate or confidence shape",
                        extra={
                            "phase": "pupil_dlc_analysis",
                            "frame_index": int(i),
                            "coord_shape": str(pts.shape),
                            "conf_shape": str(cvals.shape),
                        },
                    )
                diameters.append(np.nan)
                continue
            #print(f"cval type ={type(cvals)}, with values of type {cvals.dtype}\n compared to {type(confidence_threshold)}")
            valid = cvals >= threshold
            # print("cvals:", cvals)
            # print("threshold:", confidence_threshold)
            # print("mask  :", valid)  
            ds = [
                euclidean_distance(pts[a], pts[b])
                for a, b in self.landmark_pairs
                if a < pts.shape[0] and b < pts.shape[0] and valid[a] and valid[b]
            ]
            diameters.append(st.mean(ds) if ds else np.nan)

        # 4) interpolate & convert to mm, align with original index
        pupil_series = (
            pd.Series(diameters, index=pickle_data.index)
            .interpolate()
            .divide(px_to_mm)
        )

        # DEBUG
        # print(f"[DEBUG analyze_pupil_data] input index={pickle_data.index}")
        # print(f"[DEBUG analyze_pupil_data] output series head:\n{pupil_series.head()}")

        # 5) return DataFrame 
        return pd.DataFrame({'pupil_diameter_mm': pupil_series})