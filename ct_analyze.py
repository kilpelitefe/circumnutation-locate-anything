"""
Izleri temizle (aykiri sicrama reddi + interpolasyon + hafif duzeltme),
DB ile hatayi tekrar olc, temiz circumnutation izlerini ciz ve
periyot + donme yonu analizini yap.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ct_track import load_origins_gt

STEP_MIN = 5.0   # kareler arasi dakika

def clean(track, maxjump=25.0, smooth=5):
    t = track.copy()
    n = len(t)
    # aykiri sicrama reddi: onceki gecerliye gore > maxjump -> NaN
    last = None
    for i in range(n):
        if np.isnan(t[i, 0]):
            continue
        if last is not None and np.hypot(*(t[i]-last)) > maxjump:
            t[i] = np.nan
        else:
            last = t[i]
    # NaN'leri lineer interpolasyonla doldur
    idx = np.arange(n)
    for k in (0, 1):
        col = t[:, k]; good = ~np.isnan(col)
        if good.sum() >= 2:
            t[:, k] = np.interp(idx, idx[good], col[good])
    # hareketli ortalama ile hafif duzeltme
    if smooth > 1:
        ker = np.ones(smooth)/smooth
        for k in (0, 1):
            t[:, k] = np.convolve(t[:, k], ker, mode="same")
    return t

def rotation_and_period(t):
    """merkeze gore aci -> ortalama donme yonu (CW/CCW) ve baskin periyot (kare)."""
    c = t - t.mean(axis=0)
    ang = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
    net = ang[-1] - ang[0]                 # toplam aci degisimi (rad)
    turns = net/(2*np.pi)
    direction = "CCW (saat yonu tersi)" if net > 0 else "CW (saat yonu)"
    # periyot: y(t)'nin otokorelasyonundan ilk tepe
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
    """yavas buyume driftini (hareketli ort) cikar -> saf circumnutation salinimi."""
    r = np.empty_like(t)
    r[:, 0] = t[:, 0] - movavg(t[:, 0], w)
    r[:, 1] = t[:, 1] - movavg(t[:, 1], w)
    return r

if __name__ == "__main__":
    org, gt = load_origins_gt()
    d = np.load(r"test_out\cv_tracks.npz")
    raw = {pid: d[f"p{pid}"] for pid in org}
    clean_tr = {pid: clean(raw[pid], maxjump=30, smooth=3) for pid in org}

    # cimlenme sonrasi (>=150) + drift cikar
    W = 96          # ~8h pencere; ~5h periyodu korur, driftti siler
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    print("bitki | net_tur | yon                    | periyot")
    rows = []
    for i, pid in enumerate(sorted(org)):
        ax = axs[i//4, i%4]
        seg = clean_tr[pid][150:]
        res = detrend(seg, W)
        # kenar etkisini at (pencere yarisi)
        res = res[W//2:-W//2]
        turns, direction, peak = rotation_and_period(res)
        ax.plot(res[:, 0], res[:, 1], "-", color="#1f77b4", lw=.8)
        ax.plot(0, 0, "k+")
        per_h = f"{peak*STEP_MIN/60:.1f}h" if peak else "—"
        ax.set_title(f"seedling {pid}\n{turns:+.1f} tur, T={per_h}", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        rows.append((pid, turns, direction, per_h))
        print(f"  {pid}   | {turns:+5.2f}  | {direction:22s} | {per_h}")
    fig.suptitle("Circumnutation salinimlari (drift cikarilmis, cimlenme sonrasi) — distilled water",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(r"test_out\circumnutation_detrended.png", dpi=95)
    print("\nsaved circumnutation_detrended.png")
