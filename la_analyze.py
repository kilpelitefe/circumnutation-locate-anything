"""
2. ADIM: locate-anything kutularini bitkilere ata ve DB ile dogrula.
Her fide origin'ine en yakin kutu-merkezini ata (max yaricap icinde).
Kutu merkezi = o fidenin izlenen noktasi.
"""
import os, json, glob, sqlite3
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
BOXDIR = os.path.join(BASE, "test_out", "la_boxes")
H, W = 576, 768
MAXR = 55   # atama icin max origin-kutu mesafesi (px)

DBS = {
    "distilled water":   r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\distiled water.db",
    "nutrient solution": r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\nutrient solution.db",
}

def load_db(db):
    c = sqlite3.connect(db).cursor()
    org = {r[0]: (r[1], r[2], r[3]) for r in c.execute("SELECT id,name,xorigin,yorigin FROM plants")}
    gt = {pid: {} for pid in org}
    for pid, x, y, fr in c.execute("SELECT plantid,x,y,frame FROM samples"):
        _, ox, oy = org[pid]; gt[pid][fr] = (ox + x, H - (oy + y))
    return org, gt

def box_centers(fr):
    p = os.path.join(BOXDIR, f"f{fr:04d}.json")
    if not os.path.exists(p): return None
    dets = json.load(open(p))["detections"]
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        out.append(((x0+x1)/2, (y0+y1)/2))
    return np.array(out) if out else np.empty((0, 2))

def assign(origin_img, centers, prev=None):
    """origin'e (ya da onceki noktaya) en yakin kutu-merkezi, MAXR icinde."""
    if centers is None or len(centers) == 0: return None
    ref = prev if prev is not None else origin_img
    d = np.linalg.norm(centers - ref, axis=1)
    i = d.argmin()
    return centers[i] if d[i] <= MAXR else None

def main():
    frames = sorted(int(os.path.basename(f)[1:5]) for f in glob.glob(os.path.join(BOXDIR, "f*.json")))
    print(f"islenen kareler: {frames}\n")
    for treat, db in DBS.items():
        org, gt = load_db(db)
        errs, hits, tot = [], 0, 0
        for pid, (nm, ox, oy) in org.items():
            oimg = np.array([ox, H - oy])
            for fr in frames:
                g = gt[pid].get(fr)
                if g is None: continue
                tot += 1
                p = assign(oimg, box_centers(fr))
                if p is None: continue
                hits += 1
                errs.append(np.hypot(p[0]-g[0], p[1]-g[1]))
        cov = 100*hits/tot if tot else 0
        print(f"{treat:20s}: kapsam {hits}/{tot} ({cov:.0f}%), "
              f"hata ort {np.mean(errs):.1f}px medyan {np.median(errs):.1f}" if errs
              else f"{treat}: hic atama yok")

if __name__ == "__main__":
    main()
