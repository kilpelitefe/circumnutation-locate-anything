"""
Period analysis: oscillation period of the bean seedlings tracked with locate-anything.

Method:
 - Remove the growth drift with a LINEAR detrend (a moving average would also wipe out the
   slow oscillations we want to keep; a linear fit preserves all periods).
 - Autocorrelation -> first significant peak = dominant period.
 - Cross-check: FFT power spectrum -> dominant frequency.

Time scale (APPROXIMATE): the Commons description says "over a six day period, taking more
than 1,600 photos". The plant footage spans ~3268 frames -> 1 frame ~= 2.6 min. The video
may be edited, so the conversion to real time is NOT exact — periods are therefore reported
in frames as well as hours.

Input:  data/bean_tracks.npz   (produced by bean_track.py)
Output: figures/bean_period.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
MIN_PER_FRAME = 144*60/3268    # ~2.64 min/frame (6 days / 3268 plant frames) - APPROXIMATE
NPLANT = 3
STEP = 15                      # auto-detected from the stored frames (in main)

def lin_detrend(a):
    t = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() < 3: return a - np.nanmean(a)
    c = np.polyfit(t[good], a[good], 1)
    return a - np.polyval(c, t)

def autocorr(x):
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    ac = np.correlate(x, x, "full")[len(x)-1:]
    return ac/ac[0] if ac[0] else ac

def first_peak(ac, minlag=2):
    for i in range(minlag, len(ac)-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.2:
            return i
    return None

def fft_period(x):
    x = np.nan_to_num(x - np.nanmean(x))
    n = len(x)
    f = np.fft.rfftfreq(n, d=1.0)          # cycles per sample
    p = np.abs(np.fft.rfft(x*np.hanning(n)))**2
    p[0] = 0
    k = p.argmax()
    return (1/f[k]) if f[k] > 0 else None   # period in samples

def fmt(samples):
    if samples is None: return "-"
    fr = samples*STEP
    hrs = fr*MIN_PER_FRAME/60
    return f"{samples:.1f} samples = {fr:.0f} frames ~ {hrs:.1f} h"

def main():
    global STEP
    d = np.load(os.path.join(BASE, "data", "bean_tracks.npz"))
    frames, cen = d["frames"], d["cen"]
    n = len(frames)
    # auto-detect the sampling step from the stored frames
    if n > 1:
        STEP = int(np.median(np.diff(frames)))
    print(f"{n} samples, step {STEP} frames (~{STEP*MIN_PER_FRAME:.0f} min/sample)")
    print(f"resolvable period range: ~{2*STEP*MIN_PER_FRAME/60:.1f} h (Nyquist) "
          f".. ~{(n//2)*STEP*MIN_PER_FRAME/60:.1f} h (half the series)\n")

    fig, axs = plt.subplots(2, NPLANT, figsize=(15, 7))
    print(f"{'plant':>5} | {'axis':>5} | {'autocorrelation period':>36} | {'FFT period':>36}")
    for pid in range(NPLANT):
        x = lin_detrend(cen[:, pid, 0])
        y = lin_detrend(cen[:, pid, 1])
        for lbl, sig in (("x", x), ("y", y)):
            ac = autocorr(sig)
            pk = first_peak(ac)
            fp = fft_period(sig)
            print(f"{pid+1:>5} | {lbl:>5} | {fmt(pk):>36} | {fmt(fp):>36}")
            if lbl == "x":
                axs[0, pid].plot(frames, sig, color="#1f77b4", lw=1)
                axs[0, pid].set_title(f"plant {pid+1}: linearly detrended x(t)", fontsize=10)
                axs[0, pid].set_xlabel("frame"); axs[0, pid].axhline(0, color="k", lw=.5)
                lags = np.arange(len(ac))*STEP
                axs[1, pid].plot(lags, ac, color="#d62728", lw=1)
                axs[1, pid].axhline(0, color="k", lw=.5)
                if pk:
                    axs[1, pid].axvline(pk*STEP, color="g", ls="--",
                                        label=f"period ~ {pk*STEP*MIN_PER_FRAME/60:.1f} h")
                    axs[1, pid].legend(fontsize=8)
                axs[1, pid].set_title(f"plant {pid+1}: autocorrelation", fontsize=10)
                axs[1, pid].set_xlabel("lag (frames)")
    fig.suptitle("Oscillation period analysis - locate-anything tracking (Lima Bean, CC BY 3.0)\n"
                 "approximate time scale: 1 frame ~ 2.6 min (6 days / 3268 frames)", fontsize=12)
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "figures", "bean_period.png"), dpi=95)
    print("\nsaved figures/bean_period.png")

if __name__ == "__main__":
    main()
