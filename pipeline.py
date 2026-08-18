"""
pipeline.py — License Plate Recognition Engine
=================================================
Two-stage detect + read pipeline, hardened for production use:

  Stage 1  Detection            YOLOv8 (swap in your own fine-tuned model)
  Stage 2  Perspective fix      Rectify skewed plates before OCR
  Stage 3  Recognition          EasyOCR (ar + en)
  Stage 4  Confidence fallback  Re-attempt with enhanced crop if OCR is unsure
  Stage 5  Format validation    Score reads against Egyptian plate grammar
  Stage 6  Multi-frame tracking IOU tracker + weighted majority vote per plate

This file has no UI code — it's imported by plate_reader.py (desktop),
api.py (web/Docker), and evaluate.py (metrics).
"""

import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get("PLATE_MODEL_PATH", "yolov8n.pt")
CONF_THRESHOLD = float(os.environ.get("PLATE_CONF_THRESHOLD", 0.35))
OCR_RETRY_THRESHOLD = 0.55   # below this OCR confidence, retry with an enhanced crop
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_MISSES = 8
TRACK_MIN_HITS_TO_REPORT = 2

# Egyptian private-plate letters (the subset actually used on plates —
# excludes letters visually ambiguous with digits/each other)
EGYPT_PLATE_LETTERS = set("أبتحدرسصطعفقلمنهوي")

_detector = None
_reader = None


def get_detector():
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO(MODEL_PATH)
    return _detector


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["ar", "en"], gpu=False)
    return _reader


# --------------------------------------------------------------------------- #
# Stage 1 — Detection
# --------------------------------------------------------------------------- #
def detect_plates(frame: np.ndarray):
    """Returns list of {bbox: (x1,y1,x2,y2), det_conf, crop}."""
    model = get_detector()
    detections = []
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            det_conf = float(box.conf[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            detections.append({"bbox": (x1, y1, x2, y2), "det_conf": det_conf, "crop": crop})
    return detections


# --------------------------------------------------------------------------- #
# Stage 2 — Perspective correction
# --------------------------------------------------------------------------- #
def rectify_plate(crop: np.ndarray) -> np.ndarray:
    """Find the plate's quadrilateral inside the crop and warp it flat.
    Falls back to the original crop if no clean quad is found."""
    if crop.size == 0:
        return crop
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop

        h, w = crop.shape[:2]
        frame_area = h * w
        best_quad = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.25 * frame_area:  # too small to be the plate itself
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_quad = approx.reshape(4, 2)
                best_area = area

        if best_quad is None:
            return crop

        pts = order_quad_points(best_quad.astype("float32"))
        (tl, tr, br, bl) = pts
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 20 or height < 10:
            return crop

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(crop, matrix, (width, height))
        return warped
    except Exception:
        return crop


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# --------------------------------------------------------------------------- #
# Stage 3 & 4 — OCR with confidence-triggered fallback
# --------------------------------------------------------------------------- #
def _enhance_for_retry(crop: np.ndarray) -> np.ndarray:
    """CLAHE contrast boost + 2x upscale — a cheap 'super-resolution' fallback
    for crops that are small, low-contrast, or motion-blurred."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray


def _basic_preprocess(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)
    return gray


def _run_ocr(reader, image: np.ndarray):
    results = reader.readtext(image)
    if not results:
        return "", 0.0
    text = " ".join(r[1] for r in results).strip().upper()
    confidence = float(np.mean([r[2] for r in results]))
    return text, confidence


def read_plate_text(crop: np.ndarray):
    """Stage 3+4: rectify, OCR, and retry with an enhanced crop if unsure.
    Returns (text, confidence, was_retried)."""
    reader = get_reader()
    rectified = rectify_plate(crop)

    text, conf = _run_ocr(reader, _basic_preprocess(rectified))
    if conf >= OCR_RETRY_THRESHOLD or text == "":
        if text != "" or conf >= OCR_RETRY_THRESHOLD:
            return text, conf, False

    # Low confidence (or nothing found) — retry with the enhanced version
    retry_text, retry_conf = _run_ocr(reader, _enhance_for_retry(rectified))
    if retry_conf > conf:
        return retry_text, retry_conf, True
    return text, conf, False


# --------------------------------------------------------------------------- #
# Stage 5 — Format validation (Egyptian plate grammar)
# --------------------------------------------------------------------------- #
_DIGIT_RUN = re.compile(r"\d{1,4}")


def validate_plate_format(text: str) -> dict:
    """Scores a raw OCR string against Egyptian plate conventions.
    Doesn't hard-reject (trucks/motorcycles/older formats vary) — just
    reports whether it looks well-formed, which the caller can weight."""
    if not text:
        return {"valid": False, "reason": "empty"}

    stripped = text.replace(" ", "")
    digits = "".join(ch for ch in stripped if ch.isdigit())
    letters = "".join(ch for ch in stripped if ch in EGYPT_PLATE_LETTERS)
    unexpected = [ch for ch in stripped if not (ch.isdigit() or ch in EGYPT_PLATE_LETTERS)]

    has_digit_run = bool(_DIGIT_RUN.search(digits))
    letter_count_ok = 1 <= len(letters) <= 3
    digit_count_ok = 1 <= len(digits) <= 4
    clean = len(unexpected) == 0

    valid = has_digit_run and letter_count_ok and digit_count_ok and clean
    return {
        "valid": valid,
        "digits": digits,
        "letters": letters,
        "unexpected_chars": unexpected,
    }


# --------------------------------------------------------------------------- #
# Stage 6 — Multi-frame tracking + weighted majority vote
# --------------------------------------------------------------------------- #
def iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: tuple
    hits: int = 1
    misses: int = 0
    votes: dict = field(default_factory=dict)   # text -> cumulative weighted confidence
    best_det_conf: float = 0.0

    def add_vote(self, text: str, ocr_conf: float, det_conf: float, format_valid: bool):
        if not text:
            return
        weight = ocr_conf * (1.3 if format_valid else 1.0)
        self.votes[text] = self.votes.get(text, 0.0) + weight
        self.best_det_conf = max(self.best_det_conf, det_conf)

    @property
    def best_text(self):
        if not self.votes:
            return "", 0.0
        text = max(self.votes, key=self.votes.get)
        total = sum(self.votes.values())
        return text, self.votes[text] / total if total else 0.0

    @property
    def is_confirmed(self):
        return self.hits >= TRACK_MIN_HITS_TO_REPORT


class PlateTracker:
    """Lightweight IOU tracker. Swap for ByteTrack/DeepSORT if you need
    re-identification across occlusion — this greedy IOU matcher is enough
    for a single camera feed with modest frame-to-frame motion."""

    def __init__(self):
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list):
        """detections: list of {bbox, det_conf, plate_text, ocr_conf, format_valid}
        Returns the list of currently active Track objects."""
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        for tid, track in self._tracks.items():
            best_iou, best_det_idx = 0.0, -1
            for di in unmatched_dets:
                score = iou(track.bbox, detections[di]["bbox"])
                if score > best_iou:
                    best_iou, best_det_idx = score, di
            if best_iou >= TRACK_IOU_THRESHOLD and best_det_idx != -1:
                det = detections[best_det_idx]
                track.bbox = det["bbox"]
                track.hits += 1
                track.misses = 0
                track.add_vote(det["plate_text"], det["ocr_conf"], det["det_conf"], det["format_valid"])
                unmatched_dets.remove(best_det_idx)
                matched_track_ids.add(tid)

        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.misses += 1

        for di in unmatched_dets:
            det = detections[di]
            track = Track(track_id=self._next_id, bbox=det["bbox"])
            track.add_vote(det["plate_text"], det["ocr_conf"], det["det_conf"], det["format_valid"])
            self._tracks[self._next_id] = track
            self._next_id += 1

        self._tracks = {tid: t for tid, t in self._tracks.items() if t.misses <= TRACK_MAX_MISSES}
        return list(self._tracks.values())


# --------------------------------------------------------------------------- #
# Full pipeline entry points
# --------------------------------------------------------------------------- #
def process_frame(frame: np.ndarray):
    """Runs stages 1-5 on a single frame (no tracking).
    Returns (results, annotated_frame). Used for single-image mode."""
    annotated = frame.copy()
    results = []
    for p in detect_plates(frame):
        text, ocr_conf, retried = read_plate_text(p["crop"])
        validation = validate_plate_format(text)
        x1, y1, x2, y2 = p["bbox"]
        color = (48, 194, 242) if validation["valid"] else (140, 140, 140)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = text if text else "?"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if text:
            results.append({
                "plate_text": text,
                "detection_confidence": round(p["det_conf"], 3),
                "ocr_confidence": round(ocr_conf, 3),
                "format_valid": validation["valid"],
                "retried_with_enhancement": retried,
                "bbox": p["bbox"],
            })
    return results, annotated


def process_frame_tracked(frame: np.ndarray, tracker: PlateTracker):
    """Stages 1-6: detect, read, validate, then feed into the tracker.
    Returns (confirmed_tracks, annotated_frame). Used for video/camera mode."""
    annotated = frame.copy()
    detections = []
    for p in detect_plates(frame):
        text, ocr_conf, retried = read_plate_text(p["crop"])
        validation = validate_plate_format(text)
        detections.append({
            "bbox": p["bbox"],
            "det_conf": p["det_conf"],
            "plate_text": text,
            "ocr_conf": ocr_conf,
            "format_valid": validation["valid"],
            "retried": retried,
        })

    tracks = tracker.update(detections)

    for t in tracks:
        text, agreement = t.best_text
        x1, y1, x2, y2 = t.bbox
        confirmed = t.is_confirmed and text != ""
        color = (48, 194, 242) if confirmed else (90, 90, 90)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.track_id} {text}" if text else f"#{t.track_id}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    confirmed_tracks = [t for t in tracks if t.is_confirmed and t.best_text[0]]
    return confirmed_tracks, annotated
