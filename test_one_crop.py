"""
Smoke test: run locate-anything-cli on a single per-plant crop.

This is the experiment that established the first negative result: on the grayscale
Circumnutation Tracker footage the model returns 0 detections for an isolated small plant
in a crop, regardless of prompt or preprocessing (upscaling, CLAHE and contrast stretching
were also tried). That is why the evaluation moved to full-frame detection (la_detect.py).
See RESULTS.md Part B.

Data (Circumnutation Tracker sample set, Stolarz et al. 2014) is not redistributed here.
Point CT_DATA at the directory containing 'Video 1.avi':
    set CT_DATA=C:\\path\\to\\Circumnutation Tracker samples

Configuration (environment variables):
    LA_CLI    path to locate-anything-cli.exe
    LA_MODEL  path to the .gguf model (relative to this repo)
    CT_DATA   Circumnutation Tracker samples directory
"""
import cv2, os, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))
CT_DATA = os.environ.get("CT_DATA", "")
VIDEO = os.path.join(CT_DATA, "Video 1.avi")
CLI = os.environ.get("LA_CLI",
      os.path.join(BASE, "build", "examples", "cli", "Release", "locate-anything-cli.exe"))
MODEL_REL = os.environ.get("LA_MODEL", "models/locate-anything-q8_0.gguf")
OUTDIR = os.path.join(BASE, "figures", "smoke_test")

H, W = 576, 768
# seedling 7 origin (Y-UP coordinates as stored in the DB): xorigin, yorigin
ox, oy = 425.739, 370.643
iy = H - oy   # convert to image coordinates (y-down)
ix = ox
HALF = 90
FRAME = 400

def make_crop(frame_idx):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}")
    x0 = int(max(0, ix - HALF)); y0 = int(max(0, iy - HALF))
    x1 = int(min(W, ix + HALF)); y1 = int(min(H, iy + HALF))
    crop = img[y0:y1, x0:x1]
    path = os.path.join(OUTDIR, f"crop_s7_f{frame_idx}.png")
    cv2.imencode(".png", crop)[1].tofile(path)     # Unicode-safe write (see README)
    return path, (x0, y0, x1, y1)

def run_detect(image_path, prompt, mode="hybrid"):
    # Non-ASCII path pitfall: passing an ABSOLUTE path to the CLI corrupts non-ASCII
    # characters via MSVC's ANSI argv, so use cwd=BASE plus RELATIVE ASCII paths.
    rel_img = os.path.relpath(image_path, BASE).replace("\\", "/")
    out_json = "figures/smoke_test/det.json"
    ann = "figures/smoke_test/annotated.png"
    cmd = [CLI, "detect",
           "--model", MODEL_REL, "--input", rel_img,
           "--prompt", prompt, "--output", out_json, "--annotated", ann,
           "--mode", mode]
    print(">>>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    print("stdout:", r.stdout[-2000:])
    print("stderr:", r.stderr[-2000:])
    jp = os.path.join(BASE, out_json)
    if os.path.exists(jp):
        with open(jp) as f:
            print("JSON:", f.read())
    return r.returncode

if __name__ == "__main__":
    if not CT_DATA or not os.path.exists(VIDEO):
        sys.exit("Set CT_DATA to the 'Circumnutation Tracker samples' directory "
                 "(must contain 'Video 1.avi'). See the module docstring.")
    if not os.path.exists(os.path.join(BASE, MODEL_REL)):
        sys.exit(f"Model not found: {MODEL_REL} (see README for the download)")
    os.makedirs(OUTDIR, exist_ok=True)
    path, box = make_crop(FRAME)
    print("crop:", path, "region:", box)
    # open-vocabulary prompt, using the upstream template form
    prompt = "Locate all the instances that matches the following description: the seedling plant."
    run_detect(path, prompt)
