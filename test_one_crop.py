"""
Tek bir bitki kirpintisi uzerinde locate-anything-cli'i test eder.
Model indikten sonra: python test_one_crop.py
Amac: LocateAnything-3B'nin bu gri time-lapse goruntude fide ucunu
tespit edip etmedigini gormek (tam pipeline'a gecmeden once dogrulama).
"""
import cv2, os, subprocess, json, sys

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO = r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\Video 1.avi"
CLI = os.path.join(BASE, "build", "examples", "cli", "Release", "locate-anything-cli.exe")
MODEL = os.path.join(BASE, "models", "locate-anything-q8_0.gguf")
OUTDIR = os.path.join(BASE, "test_out")
os.makedirs(OUTDIR, exist_ok=True)

H, W = 576, 768
# fide 7 origin (y-yukari koordinat): xorigin, yorigin
ox, oy = 425.739, 370.643
iy = H - oy   # goruntu koordinatina cevir (y-asagi)
ix = ox
HALF = 90
FRAME = 400

def make_crop(frame_idx):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"frame {frame_idx} okunamadi")
    x0 = int(max(0, ix - HALF)); y0 = int(max(0, iy - HALF))
    x1 = int(min(W, ix + HALF)); y1 = int(min(H, iy + HALF))
    crop = img[y0:y1, x0:x1]
    path = os.path.join(OUTDIR, f"crop_s7_f{frame_idx}.png")
    cv2.imwrite(path, crop)
    return path, (x0, y0, x1, y1)

def run_detect(image_path, prompt, mode="hybrid"):
    # ONEMLI: yol icinde Turkce karakter (Masaustu) oldugu icin CLI'a MUTLAK
    # yol verince MSVC ANSI argv bozuyor. Bu yuzden cwd=BASE ve GORELI ASCII
    # yollar kullaniyoruz.
    rel_img = os.path.relpath(image_path, BASE).replace("\\", "/")
    out_json = "test_out/det.json"
    ann = "test_out/annotated.png"
    cmd = [CLI, "detect",
           "--model", "models/locate-anything-q8_0.gguf", "--input", rel_img,
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
    if not os.path.exists(MODEL):
        sys.exit(f"Model henuz yok: {MODEL}")
    path, box = make_crop(FRAME)
    print("crop:", path, "region:", box)
    # Denenecek promptlar - open-vocabulary
    prompt = "Locate all the instances that matches the following description: the seedling plant."
    run_detect(path, prompt)
