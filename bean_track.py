"""
STEP 2: Assign locate-anything boxes to the 3 plants, track them, and plot.

Design notes:
 - Assignment: nearest fixed x-band (robust to missing/extra boxes). Band centers are
   estimated from the median across all frames and follow slow drift.
 - Tracked point: the box CENTER (stable). The apex (top-center) is also recorded.
 - Missing frames -> NaN -> interpolation; then drift is removed for oscillation analysis.

Input:  data/det_boxes/*.json   (produced by bean_detect.py)
Output: data/bean_tracks.npz, figures/bean_final_tracks.png
"""
import os, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
BOXDIR = os.path.join(BASE, "data", "det_boxes")
NPLANT = 3

F0 = 2000   # grid origin (first analysed frame)

def frames_avail(uniform=True):
    """Return available box files. Analysis needs uniform spacing, so detect the densest
    grid automatically (the SMALLEST gap between frames) and snap to it."""
    fs = sorted(int(os.path.basename(f)[1:6]) for f in glob.glob(os.path.join(BOXDIR, "f*.json")))
    if not fs or not uniform:
        return fs
    diffs = np.diff(fs)
    step = int(np.min(diffs)) if len(diffs) else 1
    fs = [f for f in fs if (f - F0) % step == 0]
    return fs

def load_boxes(fr):
    try:
        return [b["box"] for b in json.load(open(os.path.join(BOXDIR, f"f{fr:05d}.json")))["detections"]]
    except Exception:
        return []

def cxcy(b):
    x0, y0, x1, y1 = b
    return ((x0+x1)/2, (y0+y1)/2)

def estimate_bands(frames):
    """Estimate 3 band centers from all box center-x values (1D k-means-like)."""
    xs = [cxcy(b)[0] for fr in frames for b in load_boxes(fr)]
    xs = np.array(sorted(xs))
    if len(xs) < NPLANT:
        return np.linspace(200, 1000, NPLANT)
    # simple: split into 3 equal parts, take each part's median
    parts = np.array_split(xs, NPLANT)
    return np.array([np.median(p) for p in parts])

def assign(boxes, bands, maxdist=220):
    """Nearest box per plant (by band center); each box goes to at most one plant."""
    out = {}
    used = set()
    # rank every (plant, box) distance, then match greedily
    pairs = []
    for pid in range(NPLANT):
        for bi, b in enumerate(boxes):
            d = abs(cxcy(b)[0] - bands[pid])
            pairs.append((d, pid, bi))
    for d, pid, bi in sorted(pairs):
        if pid in out or bi in used or d > maxdist:
            continue
        out[pid] = boxes[bi]; used.add(bi)
    return out

def interp_nan(a):
    idx = np.arange(len(a)); good = ~np.isnan(a)
    if good.sum() >= 2:
        a = np.interp(idx, idx[good], a[good])
    return a

def reject_jumps(pts, thresh=35.0, win=7):
    """Reject isolated jumps using a moving MEDIAN.

    On frames where a box is missing, assignment picks the wrong box and the point jumps
    by 50-100 px. NOTE: rejecting by distance to the PREVIOUS point CASCADES (once the
    plant drifts away with growth, everything gets rejected) -> compare against the local
    median instead.
    """
    p = pts.copy()
    n = len(p)
    for k in (0, 1):
        col = p[:, k]
        med = np.full(n, np.nan)
        for i in range(n):
            a, b = max(0, i-win//2), min(n, i+win//2+1)
            seg = col[a:b]
            seg = seg[~np.isnan(seg)]
            if len(seg): med[i] = np.median(seg)
        bad = np.abs(col - med) > thresh
        p[bad, :] = np.nan     # drop the point if either coordinate is an outlier
    return p

def movavg(a, w):
    return np.convolve(a, np.ones(w)/w, mode="same")

def main():
    frames = frames_avail()
    if not frames:
        print("No detection boxes found in data/det_boxes/."); return
    bands = estimate_bands(frames)
    print(f"{len(frames)} frames, band centers (x): {np.round(bands,0)}")

    n = len(frames)
    cen = np.full((n, NPLANT, 2), np.nan)   # box center
    apx = np.full((n, NPLANT, 2), np.nan)   # top-center (apex)
    nbox = []
    for i, fr in enumerate(frames):
        boxes = load_boxes(fr); nbox.append(len(boxes))
        amap = assign(boxes, bands)
        for pid, b in amap.items():
            x0, y0, x1, y1 = b
            cen[i, pid] = cxcy(b)
            apx[i, pid] = ((x0+x1)/2, y0)
        # let the bands follow slow drift
        for pid, b in amap.items():
            bands[pid] = 0.9*bands[pid] + 0.1*cxcy(b)[0]

    print(f"boxes per frame: mean {np.mean(nbox):.2f}, frames with all 3: {sum(1 for k in nbox if k>=3)}/{n}")
    for pid in range(NPLANT):
        got = np.sum(~np.isnan(cen[:, pid, 0]))
        print(f"  plant {pid+1}: assigned in {got}/{n} frames")

    # clean: outlier rejection -> interpolation
    for pid in range(NPLANT):
        n_before = np.sum(~np.isnan(cen[:, pid, 0]))
        cen[:, pid] = reject_jumps(cen[:, pid])
        apx[:, pid] = reject_jumps(apx[:, pid], thresh=60)
        n_after = np.sum(~np.isnan(cen[:, pid, 0]))
        print(f"  plant {pid+1}: {n_before-n_after} outlier frames rejected")
        for k in (0, 1):
            cen[:, pid, k] = interp_nan(cen[:, pid, k])
            apx[:, pid, k] = interp_nan(apx[:, pid, k])
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    np.savez(os.path.join(BASE, "data", "bean_tracks.npz"),
             frames=np.array(frames), cen=cen, apx=apx)

    # ---- figures ----
    W = max(9, (len(frames)//8)|1)   # drift window (odd)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    cols = ["#d62728", "#2ca02c", "#ff7f0e"]
    for pid in range(NPLANT):
        c = cen[:, pid]
        axs[0, 0].plot(c[:, 0], c[:, 1], "-", color=cols[pid], lw=.9, label=f"plant {pid+1}")
        axs[0, 1].plot(frames, c[:, 0], "-", color=cols[pid], lw=.9, label=f"plant {pid+1}")
        # remove drift -> oscillation
        rx = c[:, 0] - movavg(c[:, 0], W)
        ry = c[:, 1] - movavg(c[:, 1], W)
        k = W//2
        axs[1, 0].plot(rx[k:-k or None], ry[k:-k or None], "-", color=cols[pid], lw=.9, label=f"plant {pid+1}")
        axs[1, 1].plot(np.array(frames)[k:-k or None], rx[k:-k or None], "-", color=cols[pid], lw=.9)
    axs[0, 0].invert_yaxis(); axs[0, 0].set_title("Box-center trajectory (raw, xy)"); axs[0, 0].legend(fontsize=8)
    axs[0, 1].set_title("Horizontal position vs frame (raw)"); axs[0, 1].set_xlabel("frame"); axs[0, 1].legend(fontsize=8)
    axs[1, 0].invert_yaxis(); axs[1, 0].set_title("Detrended oscillation (circumnutation)"); axs[1, 0].legend(fontsize=8)
    axs[1, 1].set_title("Horizontal component of oscillation vs frame"); axs[1, 1].set_xlabel("frame")
    fig.suptitle("Bean seedling motion tracking with locate-anything (Lima Bean, CC BY 3.0)", fontsize=13)
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "figures", "bean_final_tracks.png"), dpi=95)
    print("saved figures/bean_final_tracks.png")

    # oscillation metrics
    print("\nplant | osc. std-x | peak-to-peak x | total path (px)")
    for pid in range(NPLANT):
        c = cen[:, pid]
        rx = c[:, 0] - movavg(c[:, 0], W); k = W//2
        r = rx[k:-k or None]
        path = np.nansum(np.sqrt((np.diff(c, axis=0)**2).sum(1)))
        print(f"  {pid+1}   | {np.nanstd(r):10.1f} | {np.nanmax(r)-np.nanmin(r):14.1f} | {path:9.0f}")

if __name__ == "__main__":
    main()
