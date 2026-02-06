import glob

PROCESSED_DIR = r'F:/251205_ETOH_RO1/processed/'


import math
import pandas as pd
import numpy as np
import statistics as st

def euclidean_distance(coord1, coord2):
    """Calculate the Euclidean distance between two points."""
    return math.dist(coord1, coord2)


def confidence_filter_coordinates(frames_coords, frames_conf, threshold):
    """
    Vectorized version: batches all frames, extracts coords/conf,
    applies threshold, and returns per‐frame lists.
    """
    # Skip first frame if needed
    coords_stack = np.stack([c[0, :, 0, :] for c in frames_coords[1:]], axis=0)
    conf_stack   = np.stack([f[:, 0, 0]    for f in frames_conf[1:]],   axis=0)
    labels_stack = conf_stack >= threshold

    # Build the output list: each entry is [coords, conf, labels]
    return [
        [coords_stack[i], conf_stack[i], labels_stack[i]]
        for i in range(coords_stack.shape[0])
    ]


def analyze_pupil_data(
    pickle_data: pd.DataFrame,
    confidence_threshold: float = 0.7,
    pixel_to_mm: float = 53.6,
    dpi: int = 300
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
    threshold : float, optional
        Minimum confidence to include a landmark in diameter computation.
        Default is 0.1.
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

    Example
    -------
    Suppose the function returns a DataFrame `result_df`. Its structure would look like:

       frame | pupil_diameter_mm
       ------|------------------
         1   | 1.23
         2   | 1.25
         3   | 1.22
         4   | 1.27
        ...  | ...
    """

    # 1) pull lists, skip metadata row
    coords_list = pickle_data['coordinates'].tolist()[1:]
    conf_list   = pickle_data['confidence'].tolist()[1:]
    
    # Return a warning if no confidence values are above the threshold
    if not any(np.any(np.array(c) >= confidence_threshold) for c in conf_list):
        print(f"[WARNING] {pickle_data.index[0:3]} No confidence values above threshold {confidence_threshold}.")
        
    # 2) to numpy arrays
    coords_arrs = [np.array(c) for c in coords_list]
    conf_arrs   = [np.array(c) for c in conf_list]

    # DEBUG: print first 3 shapes
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords.shape={c.shape}, conf.shape={f.shape}")
        
    # Print the first few values of c and f
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords values:\n{c}")
    #     print(f"[DEBUG] frame {idx} conf values:\n{f}")
        
    # 3) compute mean diameters
    pairs     = [(0, 1), (2, 3), (4, 5), (6, 7)]
    diameters = []
    for i, (coords, conf) in enumerate(zip(coords_arrs, conf_arrs)):
        pts   = np.squeeze(coords)   # expect (n_points, 2)
        cvals = np.squeeze(conf)     # expect (n_points,)
        # DEBUG unexpected shapes
        if pts.ndim != 2 or cvals.ndim != 1:
            print(f"[WARNING] frame {i} unexpected pts.shape={pts.shape}, conf.shape={cvals.shape}")
            diameters.append(np.nan)
            continue
        valid = cvals >= confidence_threshold
        ds = [
            euclidean_distance(pts[a], pts[b])
            for a, b in pairs
            if a < pts.shape[0] and b < pts.shape[0] and valid[a] and valid[b]
        ]
        diameters.append(st.mean(ds) if ds else np.nan)

    # 4) interpolate & convert to mm, align with original index
    pupil_series = (
        pd.Series(diameters, index=pickle_data.index[1:])
          .interpolate()
          .divide(pixel_to_mm)
    )
    pupil_full = pupil_series.reindex(pickle_data.index)

    # DEBUG
    # print(f"[DEBUG analyze_pupil_data] input index={pickle_data.index}")
    # print(f"[DEBUG analyze_pupil_data] output series head:\n{pupil_full.head()}")

    # 5) return DataFrame without the metadata NaN
    return pd.DataFrame({'pupil_diameter_mm': pupil_full.iloc[1:]})


pickle_data = glob.glob(PROCESSED_DIR + '/**/*full.pickle', recursive=True)

file = pickle_data[1]

df = pd.DataFrame(pd.read_pickle(file)).T

pupil_trace = analyze_pupil_data(df, confidence_threshold=0.7)
# apply a median filter (window size = 5) and plot the result
filtered_trace = pupil_trace['pupil_diameter_mm'] \
    .rolling(window=10, center=True, min_periods=1) \
    .median()
    
import matplotlib.pyplot as plt

plt.plot(filtered_trace)
plt.show()