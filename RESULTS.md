# Results — Plant Motion Tracking with locate-anything

Studying plant movement through point-based localization with **locate-anything**
(NVIDIA LocateAnything-3B, C++/ggml port).

**Two datasets were used:**
- **A) Lima Bean (MAIN RESULT)** — Wikimedia Commons, color, side view → locate-anything **WORKS** ✅
- **B) Circumnutation Tracker sample** — grayscale, top view → locate-anything **FAILS** ❌ (evaluation finding)

---

# PART A — Bean seedling motion tracking with locate-anything (MAIN RESULT)

**Data:** Wikimedia Commons "Lima Bean Time Lapse", **David Marvin, CC BY 3.0**
(`data_wiki/lima_bean.webm`, 3567 frames, 1280×720, color, side view, 3 bean seedlings).
Attribution required: David Marvin, CC BY 3.0, via Wikimedia Commons.

**Usable range:** determined by a cheap green-mask scan (`bean_scan.py`, 27 s):
frames **2000–3324** = mature above-ground seedling phase (before it: underground
germination; after it: an outro logo card).

**Pipeline:**
1. `bean_detect.py` — full-frame locate-anything detection, prompt `seedling`, fast mode
   (~80 s/frame; 265 frames ≈ 3.8 h)
2. `bean_track.py` — assign boxes to 3 plants via fixed x-bands → track box center →
   moving-median outlier rejection → interpolation → detrend → oscillation
3. `bean_period.py` — linear detrend → autocorrelation + FFT

**locate-anything performance (excellent on THIS data) — dense run, 265 frames:**

| Metric | Value |
|---|---|
| Frames with exactly 3 boxes | **155 / 169** new frames (~2.9 boxes/frame avg) |
| Per-plant assignment coverage | **plant 1: 99%, plant 2: 92%, plant 3: 99%** |
| Identity stability | Three plants in separate x-bands (~293 / 573 / 959), no swapping |
| Outlier frames rejected | 6 / 7 / 1 |

**Measured motion (box center, detrended):**

| Plant | Oscillation std-x | Peak-to-peak x | Total path |
|---|---|---|---|
| 1 | **18.1 px** | 98.2 px | 1217 px |
| 2 | 10.1 px | 71.8 px | 882 px |
| 3 | 5.6 px | 29.2 px | 649 px |

Plant 1 is the most active, plant 3 the most static (~3× difference). Plant 1 shows a
pronounced swing between frames 2700–2900 (x: 295→440→235).

The sparse (89-frame) and dense (265-frame) runs produced **the same ranking and similar
amplitudes** (16.7 / 8.8 / 5.4 → 18.1 / 10.1 / 5.6) → the amplitude measurement is robust
and independent of sampling density.

**Figures:** `figures/bean_final_tracks.png` ⭐ (trajectories + oscillation),
`figures/bean_detect.png` (detection example), `figures/bean_track_montage.png`
(frame-by-frame validation), `figures/bean_scan.png` (range determination).

## Period analysis (`bean_period.py`) — dense sampling result

Method: linear detrend (removes growth drift) → autocorrelation + FFT.
**Sampling: every 5th frame = 265 samples** (~3.8 h of locate-anything compute).

Time scale is **APPROXIMATE**: the Commons description says "over a six day period, taking
more than 1,600 photos"; the plant footage spans ~3268 frames → 1 frame ≈ 2.6 min →
**13 min/sample**, observation window ≈ 58 h.
**Resolvable range: 0.4 h (Nyquist) … 29 h** → easily strong enough to see a 1–3 h
circumnutation.

| Plant | Autocorrelation (x) | FFT | Interpretation |
|---|---|---|---|
| 1 | 26.9 h | 19.5 h | circadian scale |
| 2 | 51.6 h (x) / 24.9 h (y) | 58.4 / 29.2 h | x = series length (artifact), y circadian |
| 3 | no peak | 29.2 h | weak |

### ⭐ MAIN FINDING (negative result, ROBUST)

**No 1–3 h circumnutation was detected** — and the sampling was strong enough to see it
(13 min; Nyquist 26 min).

Hypothesis test: *"the box center averages the whole plant and may be damping the tip's
oscillation"* → **the apex (box top-center) was analyzed separately and produced identical
periods** (27.1 vs 26.9 h; FFT 19.5 h in both) → hypothesis refuted. Fast circumnutation is
genuinely absent.

This is not "we couldn't measure it" but **"we looked with sufficient resolution and it
wasn't there"** — a much stronger statement.

### Secondary finding (coarse, weak evidence)

All three plants show a slow ~20–29 h oscillation = **circadian scale** (the video spans a
6 day/night cycle → daily leaf movement is plausible). However:
- The window contains only ~2 cycles.
- The FFT values "19.5 h" and "29.2 h" are **adjacent frequency bins** (265/3 and 265/2
  samples) → period resolution is coarse at this scale; the two cannot be distinguished.
- Plant 2's 51–58 h in x equals the series length itself = artifact, not a cycle.

**Honest summary: we can say "a slow, circadian-scale oscillation is present"; we cannot
give a precise period.**

**Why no circumnutation?** Unknown. Possibilities: the seedlings may have passed their
active circumnutation phase; the video may be edited with scenes joined together (which
breaks a continuous time axis); the 2D projection of 3D circular motion in a side view may
be weak.

**Figure:** `figures/bean_period.png` (detrended signals + autocorrelation)

**General limitation:** this is a *growth* time-lapse; the signal mixes growth + nutation +
leaf movement. It is not a controlled circumnutation experiment, so clean circular loops
(like those in Part B) should not be expected.

---

# PART B — Circumnutation Tracker sample data (locate-anything EVALUATION)

Video: `Video 1.avi` (757 frames, 768×576, grayscale, top view, 16 seedlings, 5 min
interval). Top 8 seedlings = distilled water, bottom 8 = nutrient solution. Same video.

## 1. locate-anything (VLM) evaluation

**Unreliable on this data.**

| Attempt | Result |
|---|---|
| Per-plant small crop (incl. upscaling / CLAHE / contrast stretch) | **0 detections** — the model never finds an isolated small plant in a crop |
| Full frame, bare `small plant` prompt | Low/inconsistent coverage (0–17 boxes per frame) |
| Full frame, template `Locate all the instances…: seedling.` | Better (frame 500: 1→15) but some frames still 0 |
| Full frame `plant` / `leaf` / `sprout` | 26 boxes (the cap) — but it detects the **text labels** burned into the video |

**Validation (6 frames, against the manually annotated ground-truth DB):**
- Coverage: distilled 27%, nutrient 19% (i.e. **no detection in ~3/4** of plant-frame pairs)
- Error (when found): 18–27 px
- Speed: ~80 s/frame (CPU) → ~17 h for the whole video
- Boxes frequently merge two seedlings; the box center is offset from the growing tip

Visual evidence: `figures/la_vs_truth.png` (red = LA box, yellow = center, green = true tip)
Raw detections: `data/la_boxes/*.json`

**Why it fails here:** locate-anything was trained on natural images (COCO-style: color,
sharp, recognizable objects). This dataset is the opposite: grayscale, top-down, low
contrast, and the target ("growing tip") is an abstract botanical concept. The model also
mistakes the embedded digits and labels for objects. This finding is valuable in its own
right because it maps the tool's **limits of applicability**.

## 2. Classical CV tracking (reference method)

Crop from origin → Otsu inverse threshold → centroid of the largest dark contour.

| Method | Coverage | Error | Runtime (757 frames × 8 plants) |
|---|---|---|---|
| locate-anything | 19–27% | 18–27 px | ~17 hours |
| Classical CV | 100% | **9.3 px** | **10 seconds** |

Figure: `figures/validation_all8.png`

## 3. Biological finding: treatment comparison

Circumnutation metrics (post-germination, detrended oscillation):

| Metric | Distilled water (CV) | Distilled (human/DB) | Nutrient (CV) | Nutrient (human/DB) |
|---|---|---|---|---|
| Period | 7.1 h | 6.2 h | **3.9 h** | **3.9 h** |
| Amplitude | 3.9 px | 3.1 px | **15.0 px** | **21.5 px** |
| Net rotation | 3.0 turns | 3.7 turns | **8.3 turns** | **9.7 turns** |
| Path length | 377 px | 314 px | **1589 px** | **1996 px** |

**Conclusion:** nutrient solution markedly **accelerates** circumnutation (period ~6–7 h →
~4 h) and increases amplitude/activity **5–7×**. Both the automated CV and the human
tracking agree in direction → a real biological effect, not a tracking artifact.

Figures: `figures/treatment_compare.png`, `figures/circumnutation_detrended.png`

### Limitations (Part B)
- n = 8 per group; no formal statistical test yet (though the effect is large).
- The first ~150 frames (pre-germination) were excluded from analysis.
- Tracking of the nutrient group is slightly noisier (14.5 px vs 9.3 px), but the measured
  effect far exceeds this.

---

## Files

**Code (repo root):**
- `bean_scan.py` / `bean_detect.py` / `bean_track.py` / `bean_period.py` — main pipeline (Part A)
- `la_detect.py` / `la_analyze.py` / `test_one_crop.py` — locate-anything evaluation (Part B)
- `ct_track.py` / `ct_analyze.py` / `ct_compare.py` — classical CV reference + treatment comparison

**Key figures (`figures/`):**
- `bean_final_tracks.png` ⭐ — trajectories + oscillation (Part A)
- `bean_period.png` — period analysis
- `la_vs_truth.png` ⭐ — locate-anything boxes vs ground truth (Part B evaluation)
- `treatment_compare.png` ⭐ — distilled water vs nutrient solution (biological finding)
- `validation_all8.png` — classical CV validation
- `circumnutation_detrended.png` — oscillation loops
