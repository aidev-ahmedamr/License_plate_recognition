"""
api.py — Minimal REST API over the pipeline (for Docker / deployment demos)
================================================================================
POST /process   multipart form, field "file" = image or video
GET  /health

Run locally:
    pip install -r requirements.txt
    python api.py

Or containerized — see Dockerfile.
"""

import logging
import tempfile
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import pipeline

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_SCAN_INTERVAL_SECONDS = 0.5

UPLOAD_DIR = Path(tempfile.gettempdir()) / "plate_api_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("plate_api")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    job_id = uuid.uuid4().hex[:8]
    saved_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(saved_path)

    started = time.time()
    try:
        if ext in ALLOWED_IMAGE_EXT:
            frame = cv2.imread(str(saved_path))
            if frame is None:
                return jsonify({"error": "Could not read image file."}), 400
            results, _ = pipeline.process_frame(frame)
            payload = [
                {
                    "plate_text": r["plate_text"],
                    "detection_confidence": r["detection_confidence"],
                    "ocr_confidence": r["ocr_confidence"],
                    "format_valid": r["format_valid"],
                    "retried_with_enhancement": r["retried_with_enhancement"],
                    "bbox": r["bbox"],
                }
                for r in results
            ]
            media_type = "image"

        elif ext in ALLOWED_VIDEO_EXT:
            payload = _process_video(str(saved_path))
            media_type = "video"
        else:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("Pipeline failed for job %s", job_id)
        return jsonify({"error": f"Processing failed: {exc}"}), 500
    finally:
        saved_path.unlink(missing_ok=True)

    elapsed = round(time.time() - started, 2)
    log.info("Job %s (%s) done in %ss — %d plate(s)", job_id, media_type, elapsed, len(payload))
    return jsonify({
        "media_type": media_type,
        "processing_seconds": elapsed,
        "plate_count": len(payload),
        "results": payload,
    })


def _process_video(path: str):
    """Runs the tracked pipeline over the whole video, returns one
    voted-on reading per confirmed track."""
    tracker = pipeline.PlateTracker()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    stride = max(1, int(fps * VIDEO_SCAN_INTERVAL_SECONDS))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            pipeline.process_frame_tracked(frame, tracker)
        frame_idx += 1
    cap.release()

    final_tracks = tracker.update([])  # flush; returns current (unmatched-decayed) tracks
    confirmed = [t for t in final_tracks if t.is_confirmed and t.best_text[0]]
    results = []
    for t in confirmed:
        text, agreement = t.best_text
        results.append({
            "plate_text": text,
            "agreement": round(agreement, 3),
            "detection_confidence": round(t.best_det_conf, 3),
            "track_hits": t.hits,
        })
    return results


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
