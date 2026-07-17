"""
Ucuz yesil-maske taramasi (locate-anything YOK).
Amac: (1) kullanilabilir kare araligini bul (intro toprak-alti / outro logo disla),
(2) bitkilerin yatay salinimi (circumnutation) var mi gor.
3 fide yan yana -> kareyi 3 dikey suutna bol, her birinin yesil agirlik merkezini izle.
"""
import cv2, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

VID = "data_wiki/lima_bean.webm"
BASE = os.path.dirname(os.path.abspath(__file__))

def green_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # yesil-sari bitki: H ~25-95, doygunluk ve parlaklik makul
    m = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m

def main():
    cap = cv2.VideoCapture(VID)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    area = np.zeros(n)
    cols = 3
    cx = np.full((n, cols), np.nan)   # her sutunun yesil merkez-x'i (sutun-ici)
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

    # kullanilabilir aralik: yesil alan esigin ustunde
    thr = area.max()*0.15
    good = np.where(area > thr)[0]
    lo, hi = (good[0], good[-1]) if len(good) else (0, n-1)
    print(f"toplam kare: {n}")
    print(f"yesil alan max: {area.max():.0f} px, esik(%15): {thr:.0f}")
    print(f"kullanilabilir aralik (yesil>esik): frame {lo} .. {hi}  ({hi-lo} kare)")
    # aralik icinde salinim: her sutun merkez-x'in std'si (trendden arindirilmis)
    print("\nsutun | gecerli kare | merkez-x std (drift cikarilmis) = salinim genligi")
    for j in range(cols):
        seg = cx[lo:hi, j]
        val = seg[~np.isnan(seg)]
        if len(val) > 60:
            # hareketli ortalama ile drift cikar
            k = 61; ker = np.ones(k)/k
            tr = np.convolve(seg, ker, mode="same")
            res = seg - tr
            r = res[~np.isnan(res)]
            print(f"  {j+1}   | {len(val):5d}       | std={np.nanstd(r):.1f} px, tepe-tepe~{np.nanmax(r)-np.nanmin(r):.0f} px")

    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    ax[0].plot(area, color="green"); ax[0].axhline(thr, color="r", ls="--", lw=.8)
    ax[0].axvspan(lo, hi, color="green", alpha=.08)
    ax[0].set_title(f"Yesil alan / kare — kullanilabilir aralik {lo}-{hi}"); ax[0].set_xlabel("kare"); ax[0].set_ylabel("yesil piksel")
    for j in range(cols):
        ax[1].plot(cx[:, j], lw=.7, label=f"bitki {j+1} (sutun merkez-x)")
    ax[1].axvspan(lo, hi, color="green", alpha=.08)
    ax[1].set_title("Her bitkinin yatay konumu / kare (salinim = circumnutation)"); ax[1].set_xlabel("kare"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("test_out/bean_scan.png", dpi=95)
    print("\nsaved test_out/bean_scan.png")

if __name__ == "__main__":
    main()
