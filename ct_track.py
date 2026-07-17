"""
Circumnutation apex izleyici (klasik CV).
Yontem: her bitki icin origin cevresinden crop -> Otsu ters esikleme ->
en buyuk koyu kontur (fide govdesi) -> centroid = izlenen nokta.
DB (elle takip) ile ~9px ortalama hata. ~10 sn / 757 kare / 8 bitki.

Cikti: test_out/cv_tracks.npz  (pid -> Nx2 goruntu-koordinat dizisi + frame)
"""
import sqlite3, cv2, numpy as np

DB  = r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\distiled water.db"
VID = r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\Video 1.avi"
H, W = 576, 768
HALF = 55

def load_origins_gt():
    con = sqlite3.connect(DB); c = con.cursor()
    org = {r[0]: (r[1], r[2]) for r in c.execute("SELECT id,xorigin,yorigin FROM plants")}
    gt = {pid: {} for pid in org}
    for pid, x, y, fr in c.execute("SELECT plantid,x,y,frame FROM samples"):
        ox, oy = org[pid]; gt[pid][fr] = (ox + x, H - (oy + y))
    return org, gt

def detect_centroid(crop, base):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th[g > 190] = 0                                   # parlak etiket yazisini ele
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [k for k in cs if cv2.contourArea(k) > 25]
    if not cs:
        return None
    bx, by = base
    # buyuk & tabana yakin konturu sec (komsu fideyi disla)
    k = max(cs, key=lambda c: cv2.contourArea(c) /
            (1 + np.linalg.norm(c.reshape(-1, 2) - [bx, by], axis=1).min()))
    M = cv2.moments(k)
    return np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])

def track_all():
    org, gt = load_origins_gt()
    cap = cv2.VideoCapture(VID); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks = {pid: [] for pid in org}
    for fr in range(nfr):
        ok, img = cap.read()
        if not ok: break
        for pid, (ox, oy) in org.items():
            ix, iy = ox, H - oy
            x0 = int(max(0, ix-HALF)); y0 = int(max(0, iy-HALF))
            x1 = int(min(W, ix+HALF)); y1 = int(min(H, iy+HALF))
            r = detect_centroid(img[y0:y1, x0:x1], (ix-x0, iy-y0))
            tracks[pid].append((np.nan, np.nan) if r is None else (x0+r[0], y0+r[1]))
    cap.release()
    return org, gt, {pid: np.array(v) for pid, v in tracks.items()}

if __name__ == "__main__":
    org, gt, tracks = track_all()
    np.savez(r"test_out\cv_tracks.npz",
             **{f"p{pid}": tracks[pid] for pid in tracks})
    print("izlendi:", {pid: len(v) for pid, v in tracks.items()})
