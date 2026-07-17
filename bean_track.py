"""
ADIM 2 (v2): locate-anything kutularini 3 bitkiye SAGLAM ata + izle + analiz.

Duzeltmeler:
 - Atama: sabit x-bantlarina en yakinlik (eksik/fazla kutuya dayanikli).
   Bant merkezleri tum karelerin medyanindan kestiriliyor, yavas drift takibi var.
 - Izlenen nokta: kutu MERKEZI (stabil). Apex (ust-orta) da kaydediliyor.
 - Eksik kareler NaN -> interpolasyon; sonra drift cikarilip salinim analizi.
"""
import os, json, glob, cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
BOXDIR   = os.path.join(BASE, "data_wiki", "det_boxes")
FRAMEDIR = os.path.join(BASE, "data_wiki", "det_frames")
NPLANT = 3

F0 = 2000   # izgara baslangici

def frames_avail(uniform=True):
    """Mevcut kutu dosyalarini dondur. Analiz duzgun araliklama ister, o yuzden
    en yogun izgarayi otomatik bul (kareler arasi EN KUCUK fark) ve ona oturt."""
    fs = sorted(int(os.path.basename(f)[1:6]) for f in glob.glob(os.path.join(BOXDIR, "f*.json")))
    if not fs or not uniform:
        return fs
    diffs = np.diff(fs)
    step = int(np.min(diffs)) if len(diffs) else 1
    fs = [f for f in fs if (f - F0) % step == 0]
    return fs

def detect_step(fs):
    d = np.diff(fs)
    return int(np.median(d)) if len(d) else 1

def load_boxes(fr):
    try:
        return [b["box"] for b in json.load(open(os.path.join(BOXDIR, f"f{fr:05d}.json")))["detections"]]
    except Exception:
        return []

def cxcy(b):
    x0, y0, x1, y1 = b
    return ((x0+x1)/2, (y0+y1)/2)

def estimate_bands(frames):
    """tum karelerdeki kutu merkez-x'lerinden 3 bant merkezi kestir (1B k-means benzeri)."""
    xs = [cxcy(b)[0] for fr in frames for b in load_boxes(fr)]
    xs = np.array(sorted(xs))
    if len(xs) < NPLANT:
        return np.linspace(200, 1000, NPLANT)
    # basit: 3 esit parcaya bol, her parcanin medyani
    parts = np.array_split(xs, NPLANT)
    return np.array([np.median(p) for p in parts])

def assign(boxes, bands, maxdist=220):
    """her bitkiye en yakin kutu (bant merkezine gore), tek kutu tek bitkiye."""
    out = {}
    used = set()
    # once her (bitki,kutu) mesafesini sirala, aciz-gozlu eslestir
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
    """Izole sicrama reddi (hareketli MEDYAN ile).
    Eksik kutu olan karelerde atama yanlis kutuyu secip 50-100px sicratiyor.
    Not: ardisik-noktaya gore reddetme ZINCIRLEME yapiyor (bitki buyume ile
    kayinca her sey reddediliyor) -> yerel medyandan sapmaya bak."""
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
        p[bad, :] = np.nan     # koordinatlardan biri aykiriysa noktayi at
    return p

def movavg(a, w):
    return np.convolve(a, np.ones(w)/w, mode="same")

def main():
    frames = frames_avail()
    if not frames:
        print("kutu yok."); return
    bands = estimate_bands(frames)
    print(f"{len(frames)} kare, bant merkezleri (x): {np.round(bands,0)}")

    n = len(frames)
    cen = np.full((n, NPLANT, 2), np.nan)   # kutu merkezi
    apx = np.full((n, NPLANT, 2), np.nan)   # ust-orta (apex)
    nbox = []
    for i, fr in enumerate(frames):
        boxes = load_boxes(fr); nbox.append(len(boxes))
        amap = assign(boxes, bands)
        for pid, b in amap.items():
            x0, y0, x1, y1 = b
            cen[i, pid] = cxcy(b)
            apx[i, pid] = ((x0+x1)/2, y0)
        # bantlari yavasca guncelle (drift takibi)
        for pid, b in amap.items():
            bands[pid] = 0.9*bands[pid] + 0.1*cxcy(b)[0]

    print(f"kare basina kutu: ort {np.mean(nbox):.2f}, tam-3 olan: {sum(1 for k in nbox if k>=3)}/{n}")
    for pid in range(NPLANT):
        got = np.sum(~np.isnan(cen[:, pid, 0]))
        print(f"  bitki {pid+1}: {got}/{n} karede atandi")

    # temizle: aykiri sicrama reddi -> interpolasyon
    for pid in range(NPLANT):
        n_before = np.sum(~np.isnan(cen[:, pid, 0]))
        cen[:, pid] = reject_jumps(cen[:, pid])
        apx[:, pid] = reject_jumps(apx[:, pid], thresh=60)
        n_after = np.sum(~np.isnan(cen[:, pid, 0]))
        print(f"  bitki {pid+1}: {n_before-n_after} aykiri kare reddedildi")
        for k in (0, 1):
            cen[:, pid, k] = interp_nan(cen[:, pid, k])
            apx[:, pid, k] = interp_nan(apx[:, pid, k])
    np.savez(os.path.join(BASE, "data_wiki", "bean_tracks.npz"),
             frames=np.array(frames), cen=cen, apx=apx)

    # ---- gorseller ----
    W = max(9, (len(frames)//8)|1)   # drift penceresi (tek sayi)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    cols = ["#d62728", "#2ca02c", "#ff7f0e"]
    for pid in range(NPLANT):
        c = cen[:, pid]
        axs[0, 0].plot(c[:, 0], c[:, 1], "-", color=cols[pid], lw=.9, label=f"bitki {pid+1}")
        axs[0, 1].plot(frames, c[:, 0], "-", color=cols[pid], lw=.9, label=f"bitki {pid+1}")
        # drift cikar -> salinim
        rx = c[:, 0] - movavg(c[:, 0], W)
        ry = c[:, 1] - movavg(c[:, 1], W)
        k = W//2
        axs[1, 0].plot(rx[k:-k or None], ry[k:-k or None], "-", color=cols[pid], lw=.9, label=f"bitki {pid+1}")
        axs[1, 1].plot(np.array(frames)[k:-k or None], rx[k:-k or None], "-", color=cols[pid], lw=.9)
    axs[0, 0].invert_yaxis(); axs[0, 0].set_title("Kutu merkezi trajektori (ham, xy)"); axs[0, 0].legend(fontsize=8)
    axs[0, 1].set_title("Yatay konum / kare (ham)"); axs[0, 1].set_xlabel("kare"); axs[0, 1].legend(fontsize=8)
    axs[1, 0].invert_yaxis(); axs[1, 0].set_title("Drift cikarilmis salinim (circumnutation)"); axs[1, 0].legend(fontsize=8)
    axs[1, 1].set_title("Salinim yatay bileseni / kare"); axs[1, 1].set_xlabel("kare")
    fig.suptitle("locate-anything ile fasulye fidesi hareket takibi (Lima Bean, CC BY 3.0)", fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "test_out", "bean_final_tracks.png"), dpi=95)
    print("saved test_out/bean_final_tracks.png")

    # salinim metrikleri
    print("\nbitki | salinim std-x | tepe-tepe-x | toplam yol (px)")
    for pid in range(NPLANT):
        c = cen[:, pid]
        rx = c[:, 0] - movavg(c[:, 0], W); k = W//2
        r = rx[k:-k or None]
        path = np.nansum(np.sqrt((np.diff(c, axis=0)**2).sum(1)))
        print(f"  {pid+1}   | {np.nanstd(r):11.1f} | {np.nanmax(r)-np.nanmin(r):10.1f} | {path:9.0f}")

if __name__ == "__main__":
    main()
