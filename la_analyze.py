"""
STEP 2 of the locate-anything EVALUATION: assign boxes to plants and score against the
hand-annotated ground truth.

For each seedling, take the box center nearest to its origin (within a maximum radius);
that box center is the plant's tracked point. Coverage = fraction of (plant, frame) pairs
where a box was found at all.

This is the script that produced the headline evaluation numbers: coverage 27% (distilled)
/ 19% (nutrient), error 18-27 px. See RESULTS.md Part B.

Data (Circumnutation Tracker sample set, Stolarz et al. 2014) is not redistributed here.
Point CT_DATA at the directory containing the .db files:
    set CT_DATA=C:\\path\\to\\Circumnutation Tracker samples

Input: data/la_boxes/*.json  (produced by la_detect.py)
"""
import os, json, glob, sqlite3, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
BOXDIR = os.path.join(BASE, "data", "la_boxes")
CT_DATA = os.environ.get("CT_DATA", "")
H, W = 576, 768
MAXR = 55   # max origin-to-box distance for assignment (px)

DBS = {
    "distilled water":   "distiled water.db",     # (sic) original filename is misspelled
    "nutrient solution": "nutrient solution.db",
}

def load_db(db):
    """Ground truth. NOTE the coordinate convention: the DB stores y as Y-UP offsets from
    the plant origin, so image_y = H - (yorigin + y)."""
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
    """Box center nearest to the origin (or to the previous point), within MAXR."""
    if centers is None or len(centers) == 0: return None
    ref = prev if prev is not None else origin_img
    d = np.linalg.norm(centers - ref, axis=1)
    i = d.argmin()
    return centers[i] if d[i] <= MAXR else None

def main():
    if not CT_DATA or not os.path.isdir(CT_DATA):
        sys.exit("Set CT_DATA to the 'Circumnutation Tracker samples' directory "
                 "(contains the .db files). See the module docstring.")
    frames = sorted(int(os.path.basename(f)[1:5]) for f in glob.glob(os.path.join(BOXDIR, "f*.json")))
    if not frames:
        sys.exit(f"No detection boxes in {BOXDIR}. Run la_detect.py first.")
    print(f"frames processed: {frames}\n")
    for treat, fname in DBS.items():
        db = os.path.join(CT_DATA, fname)
        if not os.path.exists(db):
            print(f"{treat:20s}: {fname} not found, skipping"); continue
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
        if errs:
            cov = 100*hits/tot if tot else 0
            print(f"{treat:20s}: coverage {hits}/{tot} ({cov:.0f}%), "
                  f"error mean {np.mean(errs):.1f} px, median {np.median(errs):.1f} px")
        else:
            print(f"{treat:20s}: nothing assigned")

if __name__ == "__main__":
    main()
