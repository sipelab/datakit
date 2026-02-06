import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

CSV_PATH = "/mnt/data/20251215_142459_sub-GS26_ses-01_task-spont_mesoscope.ome_traces.csv"
FS = 50.0

BAND_LO, BAND_HI = 2.0, 5.0
WIN_S = 8.0
OVERLAP_FRAC = 0.875
FMAX = 12.0

TARGETS = {
    "VISp": "L_VISp",
    "MOs": "L_MOs",
    "SSp-ll": "L_SSp-ll",
}

def detrend_zscore_1d(x):
    xd = signal.detrend(x, type="linear")
    sd = xd.std(ddof=1)
    if sd == 0:
        sd = 1.0
    return (xd - xd.mean()) / sd

def bandpass_1d(x, fs, lo, hi, order=4):
    nyq = fs / 2.0
    sos = signal.butter(order, [lo/nyq, hi/nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)

def spectrogram_db(x, fs, win_s, overlap_frac):
    nperseg = int(win_s * fs)
    noverlap = int(nperseg * overlap_frac)
    f, tt, Sxx = signal.spectrogram(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        detrend=False, scaling="density", mode="psd"
    )
    return f, tt, 10*np.log10(Sxx + 1e-12)

# Load
df = pd.read_csv(CSV_PATH)
regions = [c for c in df.columns if c != "frame"]

# Resolve targets; fall back if missing
resolved = {}
for k, col in TARGETS.items():
    if col in regions:
        resolved[k] = col
    else:
        # heuristic fallback: choose any L_ matching substring
        cands = [c for c in regions if c.startswith("L_") and (k in c)]
        resolved[k] = cands[0] if cands else regions[0]

# Build signals
signals = {}
for k, col in resolved.items():
    x = df[col].to_numpy(dtype=float)
    signals[k] = detrend_zscore_1d(x)

# Global mean (z-scored after mean)
X_all = df[regions].to_numpy(dtype=float)
X_all = signal.detrend(X_all, axis=0, type="linear")
g = X_all.mean(axis=1)
g = detrend_zscore_1d(g)
signals["GLOBAL"] = g

t = np.arange(len(g)) / FS

# Compute 2–5 Hz bandpassed, envelope, and spectrogram (on bandpassed signal)
out_paths = []
envs = {}
specs = {}

for k, x0 in signals.items():
    xb = bandpass_1d(x0, FS, BAND_LO, BAND_HI)
    env = np.abs(signal.hilbert(xb))
    envs[k] = env

    f, tt, Sdb = spectrogram_db(xb, FS, WIN_S, OVERLAP_FRAC)
    m = (f >= 0) & (f <= FMAX)
    specs[k] = (f[m], tt, Sdb[m, :])

# Plot A: stacked envelopes (downsample for clarity)
ds = 10
plt.figure(figsize=(12, 7))
keys = ["VISp", "MOs", "SSp-ll", "GLOBAL"]
for i, k in enumerate(keys, start=1):
    ax = plt.subplot(len(keys), 1, i)
    ax.plot(t[::ds], envs[k][::ds], linewidth=1.0)
    ax.set_ylabel(k)
    if i < len(keys):
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)")
plt.suptitle(f"2–5 Hz envelope (Hilbert amplitude), bandpass {BAND_LO:.0f}–{BAND_HI:.0f} Hz", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = "/mnt/data/plot13_stacked_envelopes_2-5Hz.png"
plt.savefig(out1, dpi=200)
plt.close()
out_paths.append(out1)

# Plot B: 2x2 spectrogram panel
plt.figure(figsize=(12, 8))
for i, k in enumerate(keys, start=1):
    f, tt, Sdb = specs[k]
    ax = plt.subplot(2, 2, i)
    pcm = ax.pcolormesh(tt, f, Sdb, shading="auto")
    ax.set_ylim(0, FMAX)
    ax.set_title(k)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
plt.suptitle(f"Spectrograms after 2–5 Hz bandpass (win={WIN_S:.1f}s, overlap={OVERLAP_FRAC:.3f})", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = "/mnt/data/plot14_spectrograms_2x2_2-5Hz.png"
plt.savefig(out2, dpi=200)
plt.close()
out_paths.append(out2)

# Provide a quick overlap metric: correlation of envelopes with GLOBAL
overlap = {}
g_env = envs["GLOBAL"]
for k in ["VISp", "MOs", "SSp-ll"]:
    overlap[k] = float(np.corrcoef(envs[k], g_env)[0, 1])

out_paths, resolved, overlap
