# Plate Reader — Production-Grade License Plate Recognition

![CI](https://github.com/YOUR_USERNAME/plate-reader/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

A two-stage pipeline (YOLOv8 detection + EasyOCR recognition) hardened with
the pieces that separate a portfolio toy from something a senior engineer
would actually sign off on:

| Stage | What it does | File |
|---|---|---|
| 1. Detection | YOLOv8 locates plate bounding boxes | `pipeline.py` |
| 2. Perspective correction | Warps skewed plates flat before OCR | `pipeline.py` |
| 3. Recognition | EasyOCR reads Arabic + English text | `pipeline.py` |
| 4. Confidence fallback | Retries low-confidence reads with a CLAHE + upscaled crop | `pipeline.py` |
| 5. Format validation | Scores reads against Egyptian plate grammar (digits + letters) | `pipeline.py` |
| 6. Multi-frame tracking | IOU tracker + weighted majority vote across frames (video/camera) | `pipeline.py` |

Everything above lives in **one engine file** (`pipeline.py`) with no UI code,
so it's reused unchanged by three different front ends:

```
pipeline.py  ← shared engine, stages 1-6
  ├── plate_reader.py   Tkinter desktop app (image / video / live camera)
  ├── api.py             Flask REST API (POST /process)
  └── evaluate.py         Metrics harness — not part of the runtime, but proves it works
train_detector.py         Fine-tunes YOLOv8 on your own plate dataset
Dockerfile                 Containerizes api.py for deployment
```

## Why this structure (not one giant file)

A single script that draws a GUI, calls a detector, runs OCR, tracks objects,
serves an API, *and* trains a model would be unreadable and untestable.
Splitting by responsibility means:
- `pipeline.py` has zero UI dependencies — it's directly unit-testable and
  reusable from a desktop app, a web API, or a Jupyter notebook.
- `evaluate.py` can score `pipeline.py` without ever importing Tkinter or Flask.
- `train_detector.py` only needs `ultralytics` — it doesn't pull in Flask,
  Pillow, or the tracker at all.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Use the generic model, or train your own

Out of the box `pipeline.py` loads `yolov8n.pt` — trained on COCO, so it has
**no concept of "license plate"** as a class. Detection will be weak or
random until you either:

**a) Fine-tune your own detector** (recommended — this is what actually
makes the demo impressive):
```bash
# get a plate dataset from Roboflow Universe in YOLOv8 format, unzip to ./dataset
python train_detector.py --data dataset/data.yaml --epochs 80
# then:
export PLATE_MODEL_PATH=runs/detect/train/weights/best.pt
```

**b) Or point at a plate model you already trained** in an earlier project:
```bash
export PLATE_MODEL_PATH=/path/to/your_model.pt
```

## 3. Run whichever front end you need

**Desktop app:**
```bash
python plate_reader.py
```
Open Image / Open Video / Start Camera. Video and camera modes run through
the tracker — each physical plate gets one voted-on reading (shown as
`#track_id PLATE_TEXT`) instead of a new flickering guess every frame.

**API (for a web front end, or to demo deployment skills):**
```bash
python api.py
# POST an image or video to http://localhost:5000/process as multipart "file"
```

**Docker:**
```bash
docker build -t plate-reader-api .
docker run -p 5000:5000 -e PLATE_MODEL_PATH=/app/best.pt plate-reader-api
```
(Copy your `best.pt` next to the Dockerfile and uncomment the two lines in
`Dockerfile` that copy it in and set the env var.)

## 4. Prove it works — run the evaluation

Build a small labeled test set (10-30 images is enough for a portfolio
number) — see the format documented at the top of `evaluate.py` — then:

```bash
python evaluate.py --test-dir test_set
```

This reports detection precision/recall/F1 at IoU≥0.5, OCR character-level
accuracy, and end-to-end exact-match plate accuracy. Put these numbers in
your README/portfolio instead of "works great" — a real metric is what
makes a senior engineer trust the project.

## Design notes worth mentioning in an interview

- **Tracking uses a custom greedy IOU matcher**, not ByteTrack/DeepSORT —
  intentional: for a single fixed camera with modest frame-to-frame motion,
  a lightweight tracker is enough and avoids a heavy dependency. The
  `PlateTracker` class in `pipeline.py` is a natural swap point if you later
  need re-identification across occlusion.
- **Confidence fallback, not a blanket second pass** — the CLAHE + upscale
  retry only fires when the first OCR pass is unsure (`OCR_RETRY_THRESHOLD`),
  so you're not paying the extra compute cost on every frame.
- **Format validation informs, it doesn't gate** — Egyptian plates have
  edge cases (trucks, motorcycles, older formats), so `validate_plate_format`
  scores a read rather than silently dropping anything that doesn't match a
  strict private-car pattern. The tracker weighs validated reads higher when
  voting, without throwing away legitimate reads that fall outside the
  common pattern.

## Results

*(Fill this in once you've trained on your dataset and run `evaluate.py` —
a real number here is worth more than any amount of polish elsewhere.)*

| Detection mAP50 | 0.976 | License Plate Recognition v4 (val split) |
| Detection mAP50-95 | 0.667 | License Plate Recognition v4 (val split) |

## Tests

```bash
pip install pytest opencv-python-headless numpy
pytest tests/ -v
```

Tests cover the pure logic (IOU, format validation, tracker vote weighting,
edit distance) — they don't require the YOLO/EasyOCR model downloads, so
they run in seconds and are what CI runs on every push.

## Project status

- [x] Detection + OCR pipeline
- [x] Perspective correction
- [x] Confidence-triggered OCR retry
- [x] Egyptian plate format validation
- [x] Multi-frame tracking with majority vote
- [x] Desktop app, REST API, Docker packaging
- [x] Unit tests + CI
- [X] Custom-trained detector (train on your own dataset — see above)
- [ ] Evaluation numbers filled in from a real test set

## License

MIT — see [LICENSE](LICENSE).

## Pre-trained weights

Download the fine-tuned detector from [Releases](https://github.com/aidev-ahmedamr/License_plate_recognition/releases/tag/v1.0-model) — YOLOv8n trained on the License Plate Recognition v4 dataset (10 epochs, mAP50: 0.976, mAP50-95: 0.667).
