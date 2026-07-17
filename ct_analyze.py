"""
Clean the classical-CV tracks (outlier rejection + interpolation + light smoothing),
remove the slow growth drift, and analyse the remaining circumnutation oscillation
(rotation direction + dominant period).

Unlike the bean video, this is a controlled circumnutation experiment, so once the drift is
removed the loops are clearly visible. See RESULTS.md Part B.

Input:  data/cv_tracks.npz  (produced by ct_track.py)
Output: figures/circumnutation_detrended.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ct_track import load_origins_gt

BASE = os.path.dirname(os.path.abspath(__file__))
STEP_MIN = 5.0   # minutes between frames (from the DB settings)
GERM = 150       # frames before this are pre-germination and too noisy to analyse

def clean(track, maxjump=25.0, smooth=5):
    t = track.copy()
    n = len(t)
    # outlier rejection: a jump larger than maxjump from the last valid point -> NaN
    last = None
    for i in range(n):
        if np.isnan(t[i, 0]):
            continue
        if last is not None and np.hypot(*(t[i]-last)) > maxjump:
            t[i] = np.nan
        else:
            last = t[i]
    # fill NaNs by linear interpolation
    idx = np.arange(n)
    for k in (0, 1):
        col = t[:, k]; good = ~np.isnan(col)
        if good.sum() >= 2:
            t[:, k] = np.interp(idx, idx[good], col[good])
    # light smoothing with a moving average
    if smooth > 1:
        ker = np.ones(smooth)/smooth
        for k in (0, 1):
            t[:, k] = np.convolve(t[:, k], ker, mode="same")
    return t

def rotation_and_period(t):
    """Angle about the centre -> net rotation direction (CW/CCW) and dominant period."""
    c = t - t.mean(axis=0)
    ang = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
    net = ang[-1] - ang[0]                 # total angle change (rad)
    turns = net/(2*np.pi)
    direction = "CCW (counter-clockwise)" if net > 0 else "CW (clockwise)"
    # period: first peak of the autocorrelation of y(t)
    sig = c[:, 1] - c[:, 1].mean()
    ac = np.correlate(sig, sig, "full")[len(sig)-1:]
    ac /= ac[0] if ac[0] != 0 else 1
    peak = None
    for i in range(2, len(ac)-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.2:
            peak = i; break
    return turns, direction, peak

def movavg(a, w):
    ker = np.ones(w)/w
    return np.convolve(a, ker, mode="same")

def detrend(t, w=96):
    """Remove the slow growth drift (moving average) -> pure circumnutation oscillation."""
    r = np.empty_like(t)
    r[:, 0] = t[:, 0] - movavg(t[:, 0], w)
    r[:, 1] = t[:, 1] - movavg(t[:, 1], w)
    return r

if __name__ == "__main__":
    org, gt = load_origins_gt()
    d = np.load(os.path.join(BASE, "data", "cv_tracks.npz"))
    raw = {pid: d[f"p{pid}"] for pid in org}
    clean_tr = {pid: clean(raw[pid], maxjump=30, smooth=3) for pid in org}

    W = 96          # ~8 h window: preserves a ~5 h period while removing the drift
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    print("plant | net turns | direction              | period")
    for i, pid in enumerate(sorted(org)):
        ax = axs[i//4, i%4]
        seg = clean_tr[pid][GERM:]
        res = detrend(seg, W)
        res = res[W//2:-W//2]          # drop the moving-average edge artifacts
        turns, direction, peak = rotation_and_period(res)
        ax.plot(res[:, 0], res[:, 1], "-", color="#1f77b4", lw=.8)
        ax.plot(0, 0, "k+")
        per_h = f"{peak*STEP_MIN/60:.1f}h" if peak else "-"
        ax.set_title(f"seedling {pid}\n{turns:+.1f} turns, T={per_h}", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        print(f"  {pid}   | {turns:+5.2f}     | {direction:22s} | {per_h}")
    fig.suptitle("Circumnutation oscillations (detrended, post-germination) - distilled water",
                 fontsize=12)
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "figures", "circumnutation_detrended.png"), dpi=95)
    print("\nsaved figures/circumnutation_detrended.png")
