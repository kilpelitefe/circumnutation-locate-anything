"""
Treatment comparison: distilled water (seedlings 1-8) vs nutrient solution (seedlings 9-16).
Both groups are in the SAME video (top half / bottom half), so this is a clean paired
comparison. Each group is tracked with the classical-CV method, circumnutation metrics are
extracted, and the result is cross-checked against the hand-annotated ground truth.

This produced the project's biological finding: nutrient solution accelerates
circumnutation ~2x and increases amplitude/activity 5-7x. Both the automated and the human
tracking agree in direction, so it is not a tracking artifact. See RESULTS.md Part B.

Metrics (post-germination, on the detrended oscillation):
  - period T (hours)      : first autocorrelation peak
  - amplitude A (px)      : RMS radius of the detrended oscillation
  - net rotation (turns)  : total angle change / 2pi
  - path length L (px)    : summed distance between consecutive points (activity)

Data (Circumnutation Tracker sample set, Stolarz et al. 2014) is not redistributed here.
Point CT_DATA at the directory containing the video and .db files:
    set CT_DATA=C:\\path\\to\\Circumnutation Tracker samples

Output: figures/treatment_compare.png
"""
import sqlite3, cv2, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
CT_DATA = os.environ.get("CT_DATA", "")
VID = os.path.join(CT_DATA, "Video 1.avi")
DBS = {
    "distilled water":   os.path.join(CT_DATA, "distiled water.db"),   # (sic) misspelled upstream
    "nutrient solution": os.path.join(CT_DATA, "nutrient solution.db"),
}
H, W = 576, 768
HALF = 55
STEP_MIN = 5.0
GERM = 150   # frames before this are pre-germination and excluded

def db_info(db):
    """NOTE the coordinate convention: image_y = H - (yorigin + y); the DB stores Y-UP."""
    c = sqlite3.connect(db).cursor()
    org = {r[0]: (r[1], r[2], r[3]) for r in c.execute("SELECT id,name,xorigin,yorigin FROM plants")}
    gt = {pid: {} for pid in org}
    for pid, x, y, fr in c.execute("SELECT plantid,x,y,frame FROM samples"):
        _, ox, oy = org[pid]; gt[pid][fr] = (ox + x, H - (oy + y))
    return org, gt

def detect(crop, base):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY); b = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); th[g > 190] = 0
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [k for k in cs if cv2.contourArea(k) > 25]
    if not cs: return None
    bx, by = base
    k = max(cs, key=lambda c: cv2.contourArea(c)/(1+np.linalg.norm(c.reshape(-1,2)-[bx,by],axis=1).min()))
    M = cv2.moments(k); return np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])

def track(frames, org):
    tr = {pid: [] for pid in org}
    for img in frames:
        for pid, (_, ox, oy) in org.items():
            ix, iy = ox, H - oy
            x0 = int(max(0, ix-HALF)); y0 = int(max(0, iy-HALF))
            x1 = int(min(W, ix+HALF)); y1 = int(min(H, iy+HALF))
            r = detect(img[y0:y1, x0:x1], (ix-x0, iy-y0))
            tr[pid].append((np.nan, np.nan) if r is None else (x0+r[0], y0+r[1]))
    return {pid: np.array(v) for pid, v in tr.items()}

def clean(t, maxjump=30, smooth=3):
    t = t.copy(); n = len(t); last = None
    for i in range(n):
        if np.isnan(t[i, 0]): continue
        if last is not None and np.hypot(*(t[i]-last)) > maxjump: t[i] = np.nan
        else: last = t[i]
    idx = np.arange(n)
    for k in (0, 1):
        col = t[:, k]; good = ~np.isnan(col)
        if good.sum() >= 2: t[:, k] = np.interp(idx, idx[good], col[good])
    if smooth > 1:
        ker = np.ones(smooth)/smooth
        for k in (0, 1): t[:, k] = np.convolve(t[:, k], ker, mode="same")
    return t

def movavg(a, w): return np.convolve(a, np.ones(w)/w, mode="same")

def metrics(t):
    seg = clean(t)[GERM:]
    W2 = 96
    res = np.stack([seg[:,0]-movavg(seg[:,0],W2), seg[:,1]-movavg(seg[:,1],W2)], 1)[W2//2:-W2//2]
    ang = np.unwrap(np.arctan2(res[:,1], res[:,0]))
    turns = (ang[-1]-ang[0])/(2*np.pi)
    amp = np.sqrt((res**2).sum(1)).mean()                 # RMS radius
    sig = res[:,1]-res[:,1].mean()
    ac = np.correlate(sig, sig, "full")[len(sig)-1:]; ac /= ac[0] if ac[0] else 1
    peak = next((i for i in range(2, len(ac)-1) if ac[i]>ac[i-1] and ac[i]>ac[i+1] and ac[i]>0.2), None)
    per_h = peak*STEP_MIN/60 if peak else np.nan
    L = np.nansum(np.sqrt((np.diff(seg, axis=0)**2).sum(1)))  # path length
    return dict(period_h=per_h, amp_px=amp, turns=turns, path_px=L)

def main():
    if not CT_DATA or not os.path.exists(VID):
        sys.exit("Set CT_DATA to the 'Circumnutation Tracker samples' directory "
                 "(must contain 'Video 1.avi' and both .db files). See the module docstring.")
    cap = cv2.VideoCapture(VID); frames = []
    while True:
        ok, img = cap.read()
        if not ok: break
        frames.append(img)
    cap.release()
    print(f"{len(frames)} frames loaded\n")
    results = {}
    for treat, db in DBS.items():
        if not os.path.exists(db):
            print(f"{treat}: database not found, skipping"); continue
        org, gt = db_info(db)
        tr = track(frames, org)
        # validation error against the hand-annotated ground truth
        errs = [np.hypot(*(tr[pid][f]-gt[pid][f])) for pid in org for f in gt[pid]
                if not np.isnan(tr[pid][f][0])]
        m = {pid: metrics(tr[pid]) for pid in org}
        results[treat] = (org, tr, m, np.mean(errs))
        print(f"=== {treat} ===  (validation error mean {np.mean(errs):.1f} px)")
        print(f"{'seedling':>12} {'T (h)':>8} {'ampl (px)':>11} {'net turns':>10} {'path (px)':>10}")
        for pid in sorted(org):
            nm = org[pid][0]; d = m[pid]
            print(f"{nm:>12} {d['period_h']:>8.1f} {d['amp_px']:>11.1f} "
                  f"{d['turns']:>+10.2f} {d['path_px']:>10.0f}")
        arr = lambda k: np.array([m[pid][k] for pid in org])
        print(f"  MEAN: T={np.nanmean(arr('period_h')):.1f} h, amp={arr('amp_px').mean():.1f} px, "
              f"|turns|={np.abs(arr('turns')).mean():.2f}, path={arr('path_px').mean():.0f} px\n")

    if len(results) < 2:
        print("Need both treatments for the comparison figure."); return

    # comparison figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [t for t in DBS if t in results]
    for ax, key, title in zip(axs, ["period_h", "amp_px", "path_px"],
                              ["Period (hours)", "Amplitude (pixels)", "Path length (pixels)"]):
        data = [[results[t][2][pid][key] for pid in results[t][0]] for t in labels]
        data = [[v for v in dd if not np.isnan(v)] for dd in data]
        ax.boxplot(data, tick_labels=["distilled", "nutrient"])
        for i, dd in enumerate(data):
            ax.scatter(np.full(len(dd), i+1)+np.random.uniform(-.05, .05, len(dd)),
                       dd, c="#1f77b4", s=20, zorder=3)
        ax.set_title(title)
    fig.suptitle("Circumnutation: distilled water vs nutrient solution (automated CV tracking)",
                 fontsize=12)
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "figures", "treatment_compare.png"), dpi=95)
    print("saved figures/treatment_compare.png")

if __name__ == "__main__":
    main()
