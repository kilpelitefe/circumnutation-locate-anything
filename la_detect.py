"""
STEP 1 of the locate-anything EVALUATION: full-frame detection on the grayscale
Circumnutation Tracker video.

This is the run that showed locate-anything is unreliable on this image type (grayscale,
top-down, low contrast): coverage 19-27%, and with greedier prompts the model detects the
text labels burned into the video rather than the plants. See RESULTS.md Part B.

Expensive (~80 s/frame on CPU) -> detections are written to disk; scoring is a separate
step (la_analyze.py). Already-detected frames are skipped, so runs can be resumed.

Data (Circumnutation Tracker sample set, Stolarz et al. 2014) is not redistributed here.
Point CT_DATA at the directory containing 'Video 1.avi':
    set CT_DATA=C:\\path\\to\\Circumnutation Tracker samples

Configuration (environment variables):
    LA_CLI    path to locate-anything-cli.exe
    LA_MODEL  path to the .gguf model (relative to this repo)
    CT_DATA   Circumnutation Tracker samples directory

Usage:
    python la_detect.py                # 6-frame validation set
    python la_detect.py 0 757 5        # start stop step

Output: data/la_boxes/*.json
"""
import cv2, subprocess, os, json, sys

BASE = os.path.dirname(os.path.abspath(__file__))
CT_DATA = os.environ.get("CT_DATA", "")
VID = os.path.join(CT_DATA, "Video 1.avi")
CLI = os.environ.get("LA_CLI",
      os.path.join(BASE, "build", "examples", "cli", "Release", "locate-anything-cli.exe"))
MODEL = os.environ.get("LA_MODEL", "models/locate-anything-q8_0.gguf")  # relative ASCII path
FRAMEDIR = "data_wiki/la_frames"   # gitignored: regenerable from the video
BOXDIR   = "data/la_boxes"         # committed: the expensive output
PROMPT = "small plant"

def detect_frames(frame_list):
    if not CT_DATA or not os.path.exists(VID):
        sys.exit("Set CT_DATA to the 'Circumnutation Tracker samples' directory "
                 "(must contain 'Video 1.avi'). See the module docstring.")
    if not os.path.exists(CLI):
        sys.exit(f"locate-anything CLI not found: {CLI}\n"
                 f"Build it (see README) or set the LA_CLI environment variable.")
    os.makedirs(os.path.join(BASE, FRAMEDIR), exist_ok=True)
    os.makedirs(os.path.join(BASE, BOXDIR), exist_ok=True)
    cap = cv2.VideoCapture(VID)
    for i, fr in enumerate(frame_list):
        boxpath = os.path.join(BASE, BOXDIR, f"f{fr:04d}.json")
        if os.path.exists(boxpath):
            print(f"[{i+1}/{len(frame_list)}] frame {fr}: already done, skipping"); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            print(f"frame {fr}: could not read"); continue
        imgrel = f"{FRAMEDIR}/f{fr:04d}.png"
        # Non-ASCII path pitfall: cv2.imwrite silently writes a corrupt file to an absolute
        # ANSI path. imencode + numpy.tofile is Unicode-safe. (See README.)
        cv2.imencode(".png", img)[1].tofile(os.path.join(BASE, imgrel))
        cmd = [CLI, "detect", "--model", MODEL, "--input", imgrel,
               "--prompt", PROMPT, "--mode", "fast", "--output", f"{BOXDIR}/f{fr:04d}.json"]
        subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
        try:
            n = len(json.load(open(boxpath))["detections"])
        except Exception:
            n = "?"
        print(f"[{i+1}/{len(frame_list)}] frame {fr}: {n} boxes", flush=True)
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        a, b, s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        frames = list(range(a, b, s))
    else:
        frames = [200, 300, 400, 500, 600, 700]   # validation set
    print(f"{len(frames)} frames to detect (~{len(frames)*80/60:.0f} min)")
    detect_frames(frames)
    print("done.")
