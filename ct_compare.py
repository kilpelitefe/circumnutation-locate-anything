"""
Tedavi karsilastirmasi: distilled water (fide 1-8) vs nutrient solution (fide 9-16).
Ikisi de ayni videoda. Her grubu otomatik CV ile izle, circumnutation metriklerini
cikar, DB ground-truth ile dogrula, karsilastir.

Metrikler (cimlenme sonrasi, drift cikarilmis salinim uzerinden):
  - periyot T (saat)          : otokorelasyon ilk tepe
  - genlik A (piksel)         : detrend salinimin RMS yaricapi
  - net donme (tur)           : toplam aci degisimi / 2pi
  - yol uzunlugu L (piksel)   : ardisik noktalar arasi toplam mesafe (aktivite)
"""
import sqlite3, cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

VID = r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\Video 1.avi"
DBS = {
    "distilled water": r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\distiled water.db",
    "nutrient solution": r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\nutrient solution.db",
}
H, W = 576, 768
HALF = 55
STEP_MIN = 5.0
GERM = 150   # cimlenme oncesi atilacak kare sayisi

def db_info(db):
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
    amp = np.sqrt((res**2).sum(1)).mean()                 # RMS yaricap
    sig = res[:,1]-res[:,1].mean()
    ac = np.correlate(sig, sig, "full")[len(sig)-1:]; ac /= ac[0] if ac[0] else 1
    peak = next((i for i in range(2, len(ac)-1) if ac[i]>ac[i-1] and ac[i]>ac[i+1] and ac[i]>0.2), None)
    per_h = peak*STEP_MIN/60 if peak else np.nan
    L = np.nansum(np.sqrt((np.diff(seg, axis=0)**2).sum(1)))  # yol uzunlugu
    return dict(period_h=per_h, amp_px=amp, turns=turns, path_px=L)

def main():
    cap = cv2.VideoCapture(VID); frames = []
    while True:
        ok, img = cap.read()
        if not ok: break
        frames.append(img)
    cap.release()
    print(f"{len(frames)} kare yuklendi\n")
    results = {}
    for treat, db in DBS.items():
        org, gt = db_info(db)
        tr = track(frames, org)
        # dogrulama hatasi
        errs = [np.hypot(*(tr[pid][f]-gt[pid][f])) for pid in org for f in gt[pid] if not np.isnan(tr[pid][f][0])]
        m = {pid: metrics(tr[pid]) for pid in org}
        results[treat] = (org, tr, m, np.mean(errs))
        print(f"=== {treat} ===  (dogrulama hatasi ort {np.mean(errs):.1f}px)")
        print(f"{'fide':>12} {'T(saat)':>8} {'genlik(px)':>11} {'net_tur':>8} {'yol(px)':>9}")
        for pid in sorted(org):
            nm = org[pid][0]; d = m[pid]
            print(f"{nm:>12} {d['period_h']:>8.1f} {d['amp_px']:>11.1f} {d['turns']:>+8.2f} {d['path_px']:>9.0f}")
        arr = lambda k: np.array([m[pid][k] for pid in org])
        print(f"  ORTALAMA: T={np.nanmean(arr('period_h')):.1f}h amp={arr('amp_px').mean():.1f}px "
              f"|tur|={np.abs(arr('turns')).mean():.2f} yol={arr('path_px').mean():.0f}px\n")

    # karsilastirma figuru
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = list(DBS)
    for ax, key, title in zip(axs, ["period_h", "amp_px", "path_px"],
                              ["Periyot (saat)", "Genlik (piksel)", "Yol uzunlugu (piksel)"]):
        data = [[results[t][2][pid][key] for pid in results[t][0]] for t in labels]
        data = [[v for v in dd if not np.isnan(v)] for dd in data]
        ax.boxplot(data, tick_labels=["saf su", "besin"])
        for i, dd in enumerate(data):
            ax.scatter(np.full(len(dd), i+1)+np.random.uniform(-.05,.05,len(dd)), dd, c="#1f77b4", s=20, zorder=3)
        ax.set_title(title)
    fig.suptitle("Circumnutation: distilled water vs nutrient solution (otomatik CV takip)", fontsize=12)
    plt.tight_layout(); plt.savefig(r"test_out\treatment_compare.png", dpi=95)
    print("saved treatment_compare.png")

if __name__ == "__main__":
    main()
