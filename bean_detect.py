"""
STEP 1: Full-frame locate-anything detection on the bean video.

Runs the `seedling` prompt over the given frames and stores the boxes as JSON.
Expensive (~80 s/frame on CPU), so detections are written to disk and the analysis
(bean_track.py / bean_period.py) reads them back — the detection never has to be repeated.
Already-detected frames are skipped, so an interrupted run can simply be restarted.

Usage:
    python bean_detect.py                  # ~11-frame validation set
    python bean_detect.py 2000 3325 5      # start stop step

Configuration (environment variables):
    LA_CLI    path to locate-anything-cli.exe   (see README for the build)
    LA_MODEL  path to the .gguf model           (relative to this repo)

NOTE on non-ASCII paths (see README "Technical notes"): the CLI receives a RELATIVE ASCII
path plus cwd=repo, because MSVC's ANSI argv corrupts non-ASCII characters in absolute
paths. Frames are written with imencode+tofile because cv2.imwrite silently fails on such
paths.

Output: data/det_boxes/*.json
"""
import cv2, subprocess, os, json, sys

BASE  = os.path.dirname(os.path.abspath(__file__))
VID   = "data_wiki/lima_bean.webm"
CLI   = os.environ.get("LA_CLI",
        os.path.join(BASE, "build", "examples", "cli", "Release", "locate-anything-cli.exe"))
MODEL = os.environ.get("LA_MODEL", "models/locate-anything-q8_0.gguf")
FRAMEDIR = "data_wiki/det_frames"     # gitignored: regenerable from the video
BOXDIR   = "data/det_boxes"           # committed: the expensive output
PROMPT = "seedling"

def detect(frame_list):
    if not os.path.exists(CLI):
        sys.exit(f"locate-anything CLI not found: {CLI}\n"
                 f"Build it (see README) or set the LA_CLI environment variable.")
    os.makedirs(os.path.join(BASE, FRAMEDIR), exist_ok=True)
    os.makedirs(os.path.join(BASE, BOXDIR), exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(BASE, VID))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {VID}\nDownload it first (see README).")
    for i, fr in enumerate(frame_list):
        boxpath = os.path.join(BASE, BOXDIR, f"f{fr:05d}.json")
        if os.path.exists(boxpath):
            print(f"[{i+1}/{len(frame_list)}] frame {fr}: already done, skipping", flush=True)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            print(f"frame {fr}: could not read"); continue
        imgrel = f"{FRAMEDIR}/f{fr:05d}.png"
        cv2.imencode(".png", img)[1].tofile(os.path.join(BASE, imgrel))   # Unicode-safe write
        cmd = [CLI, "detect", "--model", MODEL, "--input", imgrel,
               "--prompt", PROMPT, "--mode", "fast", "--output", f"{BOXDIR}/f{fr:05d}.json"]
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
        frames = list(range(2000, 3325, 130))   # ~11-frame validation set
    print(f"{len(frames)} frames to detect (~{len(frames)*80/60:.0f} min): {frames}", flush=True)
    detect(frames)
    print("done.", flush=True)
