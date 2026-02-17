import numpy as np
import matplotlib.pyplot as plt

def plot_mesomap_pupil_treadmill(dataset, subject="GS26", session="ses-04", task="task-spont", roi_count=6):
    x = dataset.loc[(subject, session, task)]

    # Mesomap ROIs (array-like cells)
    meso_series = x["mesomap"].drop("frame", errors="ignore")
    meso_series = meso_series[meso_series.apply(lambda v: hasattr(v, "__len__") and not np.isscalar(v))]
    lengths = meso_series.apply(len)
    target_len = lengths.mode().iloc[0]
    meso_series = meso_series[lengths == target_len]

    roi_names = ['L_VISp', 'R_VISp']
    meso_traces = np.stack(meso_series.loc[roi_names].values)

    # Time (master elapsed) trimmed to mesomap
    t_raw = np.asarray(x[("time", "master_elapsed_s")])
    t_meso = t_raw[1 : 1 + target_len]

    # Pupil
    t_pupil = np.asarray(x[("pupil", "time_elapsed_s")])
    pupil = np.asarray(x[("pupil", "pupil_diameter_mm")])
    pupil_interp = np.interp(t_meso, t_pupil, pupil)

    # Plot (no locomotion)
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for trace, name in zip(meso_traces, roi_names):
        axes[0].plot(t_meso, trace, alpha=0.6, linewidth=0.8, label=name)
    axes[0].set_ylabel("ΔF/F")
    axes[0].set_title("Mesomap ROIs")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(t_meso, pupil_interp, color="tab:orange")
    axes[1].set_ylabel("Pupil diameter (mm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Pupil (aligned)")

    # Zoom to 400–500 s
    axes[1].set_xlim(380, 800)

    plt.tight_layout()
    plt.show()

# Example:
plot_mesomap_pupil_treadmill(dataset)

