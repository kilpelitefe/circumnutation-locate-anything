"""
Fasulye videosunda locate-anything tam-kare tespit (ADIM 1).
Belirtilen karelerde 'seedling' tespiti -> kutulari JSON kaydet.
~80 sn/kare. Turkce yol tuzagi icin: kare yazma imencode+tofile (Unicode),
CLI'a goreli ASCII yol + cwd=repo.
"""
import cv2, subprocess, os, json, sys

BASE  = os.path.dirname(os.path.abspath(__file__))
VID   = "data_wiki/lima_bean.webm"
CLI   = os.path.join(BASE, "build", "examples", "cli", "Release", "locate-anything-cli.exe")
MODEL = "models/locate-anything-q8_0.gguf"
FRAMEDIR = "data_wiki/det_frames"
BOXDIR   = "data_wiki/det_boxes"
PROMPT = "seedling"

def detect(frame_list):
    os.makedirs(os.path.join(BASE, FRAMEDIR), exist_ok=True)
    os.makedirs(os.path.join(BASE, BOXDIR), exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(BASE, VID))
    for i, fr in enumerate(frame_list):
        boxpath = os.path.join(BASE, BOXDIR, f"f{fr:05d}.json")
        if os.path.exists(boxpath):
            print(f"[{i+1}/{len(frame_list)}] frame {fr}: var, atla", flush=True); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            print(f"frame {fr} okunamadi"); continue
        imgrel = f"{FRAMEDIR}/f{fr:05d}.png"
        cv2.imencode(".png", img)[1].tofile(os.path.join(BASE, imgrel))   # Unicode-guvenli
        cmd = [CLI, "detect", "--model", MODEL, "--input", imgrel,
               "--prompt", PROMPT, "--mode", "fast", "--output", f"{BOXDIR}/f{fr:05d}.json"]
        subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
        try:
            n = len(json.load(open(boxpath))["detections"])
        except Exception:
            n = "?"
        print(f"[{i+1}/{len(frame_list)}] frame {fr}: {n} kutu", flush=True)
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        a, b, s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        frames = list(range(a, b, s))
    else:
        frames = list(range(2000, 3325, 130))   # ~11 kare dogrulama seti
    print(f"{len(frames)} kare tespit edilecek (~{len(frames)*80/60:.0f} dk): {frames}", flush=True)
    detect(frames)
    print("bitti.", flush=True)
