"""
Periyot analizi: locate-anything ile izlenen fasulye fidelerinin salinim periyodu.

Yontem:
 - Buyume kaymasini DOGRUSAL cikar (hareketli ortalama yavas salinimlari da silerdi;
   dogrusal detrend tum periyotlari korur).
 - Otokorelasyon -> ilk anlamli tepe = baskin periyot.
 - Capraz kontrol: FFT guc spektrumu -> baskin frekans.

Zaman olcegi (YAKLASIK): Commons aciklamasi "6 gunde >1600 foto" diyor. Bitki
goruntusu ~3268 kareye yayiliyor -> 1 kare ~= 2.6 dk; ornekleme 15 kare ~= 40 dk.
Video farkli sahneler icerdigi icin gercek-zaman cevrimi kesin DEGIL, yaklasiktir.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
MIN_PER_FRAME = 144*60/3268    # ~2.64 dk/kare (6 gun / 3268 bitki karesi) - YAKLASIK
NPLANT = 3
STEP = 15                      # kayitli karelerden otomatik belirlenir (main icinde)

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
    f = np.fft.rfftfreq(n, d=1.0)          # ornek basina cevrim
    p = np.abs(np.fft.rfft(x*np.hanning(n)))**2
    p[0] = 0
    k = p.argmax()
    return (1/f[k]) if f[k] > 0 else None   # ornek cinsinden periyot

def fmt(samples):
    if samples is None: return "—"
    fr = samples*STEP
    hrs = fr*MIN_PER_FRAME/60
    return f"{samples:.1f} ornek = {fr:.0f} kare ~ {hrs:.1f} sa"

def main():
    global STEP
    d = np.load(os.path.join(BASE, "data_wiki", "bean_tracks.npz"))
    frames, cen = d["frames"], d["cen"]
    n = len(frames)
    # ornekleme adimini kayitli karelerden otomatik belirle
    if n > 1:
        STEP = int(np.median(np.diff(frames)))
    print(f"{n} ornek, adim {STEP} kare (~{STEP*MIN_PER_FRAME:.0f} dk/ornek)")
    print(f"cozulebilir periyot araligi: ~{2*STEP*MIN_PER_FRAME/60:.1f} sa (Nyquist) "
          f".. ~{(n//2)*STEP*MIN_PER_FRAME/60:.1f} sa (seri yarisi)\n")

    fig, axs = plt.subplots(2, NPLANT, figsize=(15, 7))
    print(f"{'bitki':>5} | {'eksen':>5} | {'otokorelasyon periyodu':>34} | {'FFT periyodu':>32}")
    for pid in range(NPLANT):
        x = lin_detrend(cen[:, pid, 0])
        y = lin_detrend(cen[:, pid, 1])
        for lbl, sig in (("x", x), ("y", y)):
            ac = autocorr(sig)
            pk = first_peak(ac)
            fp = fft_period(sig)
            print(f"{pid+1:>5} | {lbl:>5} | {fmt(pk):>34} | {fmt(fp):>32}")
            if lbl == "x":
                axs[0, pid].plot(frames, sig, color="#1f77b4", lw=1)
                axs[0, pid].set_title(f"bitki {pid+1}: dogrusal-detrend x(t)", fontsize=10)
                axs[0, pid].set_xlabel("kare"); axs[0, pid].axhline(0, color="k", lw=.5)
                lags = np.arange(len(ac))*STEP
                axs[1, pid].plot(lags, ac, color="#d62728", lw=1)
                axs[1, pid].axhline(0, color="k", lw=.5)
                if pk:
                    axs[1, pid].axvline(pk*STEP, color="g", ls="--",
                                        label=f"periyot ≈ {pk*STEP*MIN_PER_FRAME/60:.1f} sa")
                    axs[1, pid].legend(fontsize=8)
                axs[1, pid].set_title(f"bitki {pid+1}: otokorelasyon", fontsize=10)
                axs[1, pid].set_xlabel("gecikme (kare)")
    fig.suptitle("Salinim periyot analizi — locate-anything takibi (Lima Bean, CC BY 3.0)\n"
                 "zaman olcegi yaklasik: 1 kare ≈ 2.6 dk (6 gun / 3268 kare)", fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "test_out", "bean_period.png"), dpi=95)
    print("\nsaved test_out/bean_period.png")

if __name__ == "__main__":
    main()
