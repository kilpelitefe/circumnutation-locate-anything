"""
Cheap green-mask scan (NO locate-anything involved). Runs in ~27 s over the whole video.

Purpose:
 (1) find the usable frame range (exclude the intro underground germination and the outro
     logo card), so the expensive VLM is aimed only at frames that contain plants;
 (2) check whether the plants actually move horizontally (an oscillation worth tracking).

The 3 seedlings stand side by side -> split the frame into 3 vertical columns and follow
each column's green centroid. This is only a coarse pre-scan; the real per-plant tracking
is done by bean_detect.py + bean_track.py.

Output: figures/bean_scan.png
"""
import cv2, numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
VID = os.path.join(BASE, "data_wiki", "lima_bean.webm")

def green_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # green-yellow plant tissue: H ~25-95, with reasonable saturation and value
    m = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m

def main():
    cap = cv2.VideoCapture(VID)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {VID}\nDownload it first (see README).")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    area = np.zeros(n)
    cols = 3
    cx = np.full((n, cols), np.nan)   # each column's green centroid-x (within the column)
    cy = np.full((n, cols), np.nan)
    bounds = [(i*W//cols, (i+1)*W//cols) for i in range(cols)]
    fr = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        m = green_mask(img)
        area[fr] = m.sum()/255
        for j, (a, b) in enumerate(bounds):
            sub = m[:, a:b]
            if sub.sum() > 255*50:
                ys, xs = np.nonzero(sub)
                cx[fr, j] = xs.mean(); cy[fr, j] = ys.mean()
        fr += 1
    cap.release()

    # usable range: where the green area exceeds the threshold
    thr = area.max()*0.15
    good = np.where(area > thr)[0]
    lo, hi = (good[0], good[-1]) if len(good) else (0, n-1)
    print(f"total frames: {n}")
    print(f"max green area: {area.max():.0f} px, threshold (15%): {thr:.0f}")
    print(f"usable range (green > threshold): frames {lo} .. {hi}  ({hi-lo} frames)")
    # oscillation inside the range: std of each column's centroid-x after removing the trend
    print("\ncolumn | valid frames | centroid-x std (detrended) = oscillation amplitude")
    for j in range(cols):
        seg = cx[lo:hi, j]
        val = seg[~np.isnan(seg)]
        if len(val) > 60:
            # remove drift with a moving average
            k = 61; ker = np.ones(k)/k
            tr = np.convolve(seg, ker, mode="same")
            res = seg - tr
            r = res[~np.isnan(res)]
            print(f"  {j+1}    | {len(val):5d}        | std={np.nanstd(r):.1f} px, "
                  f"peak-to-peak ~{np.nanmax(r)-np.nanmin(r):.0f} px")

    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    ax[0].plot(area, color="green"); ax[0].axhline(thr, color="r", ls="--", lw=.8)
    ax[0].axvspan(lo, hi, color="green", alpha=.08)
    ax[0].set_title(f"Green area per frame - usable range {lo}-{hi}")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("green pixels")
    for j in range(cols):
        ax[1].plot(cx[:, j], lw=.7, label=f"plant {j+1} (column centroid-x)")
    ax[1].axvspan(lo, hi, color="green", alpha=.08)
    ax[1].set_title("Horizontal position of each plant per frame (oscillation = circumnutation)")
    ax[1].set_xlabel("frame"); ax[1].legend(fontsize=8)
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "figures", "bean_scan.png"), dpi=95)
    print("\nsaved figures/bean_scan.png")

if __name__ == "__main__":
    main()
