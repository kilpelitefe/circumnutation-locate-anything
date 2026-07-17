"""
1. ADIM: locate-anything ile tam-kare tespit.
Belirtilen karelerde 'small plant' tespiti calistirir, kutulari JSON kaydeder.
Pahali (~80 sn/kare CPU) -> tespitleri diske yazar, analiz ayri adimda.
"""
import cv2, subprocess, os, json, sys

BASE = os.path.dirname(os.path.abspath(__file__))
VID  = r"C:\Users\kilpe\AppData\Local\Temp\claude\C--Users-kilpe-OneDrive-Masa-st--AOI\38061a1a-d683-492d-9fc8-60bd22ef4dc2\scratchpad\extracted\circumnutation\Circumnutation Tracker samples\Video 1.avi"
CLI   = os.path.join(BASE, "build", "examples", "cli", "Release", "locate-anything-cli.exe")
MODEL = "models/locate-anything-q8_0.gguf"     # goreli ASCII (Turkce yol tuzagi)
FRAMEDIR = "test_out/la_frames"
BOXDIR   = "test_out/la_boxes"
PROMPT = "small plant"

def detect_frames(frame_list):
    os.makedirs(os.path.join(BASE, FRAMEDIR), exist_ok=True)
    os.makedirs(os.path.join(BASE, BOXDIR), exist_ok=True)
    cap = cv2.VideoCapture(VID)
    for i, fr in enumerate(frame_list):
        boxpath = os.path.join(BASE, BOXDIR, f"f{fr:04d}.json")
        if os.path.exists(boxpath):
            print(f"[{i+1}/{len(frame_list)}] frame {fr}: zaten var, atla"); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            print(f"frame {fr} okunamadi"); continue
        imgrel = f"{FRAMEDIR}/f{fr:04d}.png"
        # Turkce yol (Masaustu) tuzagi: cv2.imwrite mutlak ANSI yolda bozuluyor.
        # imencode + numpy.tofile Unicode-guvenli yazar.
        cv2.imencode(".png", img)[1].tofile(os.path.join(BASE, imgrel))
        cmd = [CLI, "detect", "--model", MODEL, "--input", imgrel,
               "--prompt", PROMPT, "--mode", "fast", "--output", f"{BOXDIR}/f{fr:04d}.json"]
        r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
        try:
            n = len(json.load(open(boxpath))["detections"])
        except Exception:
            n = "?"
        print(f"[{i+1}/{len(frame_list)}] frame {fr}: {n} kutu", flush=True)
    cap.release()

if __name__ == "__main__":
    # argv: baslangic bitis adim  (or. 0 757 5) ya da bos -> dogrulama karesi seti
    if len(sys.argv) == 4:
        a, b, s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        frames = list(range(a, b, s))
    else:
        frames = [200, 300, 400, 500, 600, 700]   # dogrulama seti
    print(f"{len(frames)} kare tespit edilecek (~{len(frames)*80/60:.0f} dk)")
    detect_frames(frames)
    print("bitti.")
