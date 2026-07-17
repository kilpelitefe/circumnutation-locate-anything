"""
Classical-CV seedling tracker (the reference method for Part B).

Method: for each plant, crop a window around its known origin -> Otsu inverse threshold ->
pick the largest dark contour (the seedling body) -> its centroid is the tracked point.

Achieves ~9 px mean error against the hand-annotated DB, over 757 frames x 8 plants in
~10 seconds — which is what put locate-anything's 19-27% coverage / 18-27 px error /
~17 hours into context. See RESULTS.md Part B.

Data (Circumnutation Tracker sample set, Stolarz et al. 2014) is not redistributed here.
Point CT_DATA at the directory containing the video and .db files:
    set CT_DATA=C:\\path\\to\\Circumnutation Tracker samples

Output: data/cv_tracks.npz  (per plant: Nx2 array of image coordinates)
"""
import sqlite3, cv2, os, sys, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CT_DATA = os.environ.get("CT_DATA", "")
DB  = os.path.join(CT_DATA, "distiled water.db")   # (sic) original filename is misspelled
VID = os.path.join(CT_DATA, "Video 1.avi")
H, W = 576, 768
HALF = 55        # crop half-size; 55 was optimal (40/45/50/55 were swept)

def _check_data():
    if not CT_DATA or not os.path.exists(DB) or not os.path.exists(VID):
        sys.exit("Set CT_DATA to the 'Circumnutation Tracker samples' directory "
                 "(must contain 'Video 1.avi' and 'distiled water.db'). "
                 "See the module docstring.")

def load_origins_gt():
    """Plant origins and ground truth.

    NOTE the coordinate convention: the DB stores y as Y-UP offsets from the plant origin,
    so image_y = H - (yorigin + y). This was verified by overlaying DB points on frames.
    """
    _check_data()
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
    th[g > 190] = 0                                   # drop the bright burned-in label text
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [k for k in cs if cv2.contourArea(k) > 25]
    if not cs:
        return None
    bx, by = base
    # Prefer a contour that is both large and close to the base, which excludes the
    # neighbouring seedling. NOTE: "farthest contour point from the base" (i.e. the apex)
    # was tried and is much worse (59-79 px) — it jumps to neighbours and noise.
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
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    np.savez(os.path.join(BASE, "data", "cv_tracks.npz"),
             **{f"p{pid}": tracks[pid] for pid in tracks})
    print("tracked:", {pid: len(v) for pid, v in tracks.items()})
    print("saved data/cv_tracks.npz")
