"""
Unit tests for pipeline.py's pure logic — the geometry, validation, and
tracking-vote math. Deliberately does NOT touch get_detector()/get_reader(),
so these run in CI in seconds with no model downloads.
"""

import numpy as np
import pytest

import pipeline


# --------------------------------------------------------------------------- #
# IOU
# --------------------------------------------------------------------------- #
def test_iou_identical_boxes():
    box = (10, 10, 50, 50)
    assert pipeline.iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = (0, 0, 10, 10)
    b = (100, 100, 110, 110)
    assert pipeline.iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    # intersection = 5x5 = 25, union = 100+100-25 = 175
    assert pipeline.iou(a, b) == pytest.approx(25 / 175)


# --------------------------------------------------------------------------- #
# Plate format validation
# --------------------------------------------------------------------------- #
def test_validate_plate_format_valid_mixed():
    result = pipeline.validate_plate_format("1234ب")
    assert result["valid"] is True
    assert result["digits"] == "1234"


def test_validate_plate_format_empty():
    result = pipeline.validate_plate_format("")
    assert result["valid"] is False
    assert result["reason"] == "empty"


def test_validate_plate_format_rejects_unexpected_chars():
    result = pipeline.validate_plate_format("12#34")
    assert result["valid"] is False
    assert "#" in result["unexpected_chars"]


def test_validate_plate_format_rejects_too_many_letters():
    # more than 3 plate letters shouldn't validate
    result = pipeline.validate_plate_format("ابتحد123")
    assert result["valid"] is False


# --------------------------------------------------------------------------- #
# Quad point ordering (perspective correction helper)
# --------------------------------------------------------------------------- #
def test_order_quad_points_returns_tl_tr_br_bl():
    # a simple axis-aligned square, given in scrambled order
    pts = np.array([[10, 10], [10, 0], [0, 0], [0, 10]], dtype="float32")
    ordered = pipeline.order_quad_points(pts)
    tl, tr, br, bl = ordered
    assert tl[0] <= tr[0]        # top-left is left of top-right
    assert bl[1] >= tl[1]        # bottom-left is below top-left


# --------------------------------------------------------------------------- #
# Track voting
# --------------------------------------------------------------------------- #
def test_track_majority_vote_prefers_higher_weighted_text():
    track = pipeline.Track(track_id=1, bbox=(0, 0, 10, 10))
    track.add_vote("1234 ABC", ocr_conf=0.4, det_conf=0.9, format_valid=False)
    track.add_vote("1234 ABC", ocr_conf=0.5, det_conf=0.9, format_valid=False)
    track.add_vote("9999 XYZ", ocr_conf=0.6, det_conf=0.9, format_valid=False)

    text, agreement = track.best_text
    assert text == "1234 ABC"   # cumulative weight (0.4+0.5=0.9) beats the single 0.6 vote


def test_track_confirmed_requires_min_hits():
    track = pipeline.Track(track_id=1, bbox=(0, 0, 10, 10), hits=1)
    assert track.is_confirmed is False
    track.hits = pipeline.TRACK_MIN_HITS_TO_REPORT
    assert track.is_confirmed is True


# --------------------------------------------------------------------------- #
# Tracker end-to-end (pure geometry, no OCR/detector calls)
# --------------------------------------------------------------------------- #
def test_tracker_assigns_same_id_across_frames():
    tracker = pipeline.PlateTracker()

    frame1_dets = [{"bbox": (10, 10, 60, 40), "det_conf": 0.9,
                    "plate_text": "1234 ABC", "ocr_conf": 0.7, "format_valid": True}]
    frame2_dets = [{"bbox": (12, 11, 62, 41), "det_conf": 0.9,   # slight motion
                    "plate_text": "1234 ABC", "ocr_conf": 0.8, "format_valid": True}]

    tracks1 = tracker.update(frame1_dets)
    tracks2 = tracker.update(frame2_dets)

    assert len(tracks1) == 1
    assert len(tracks2) == 1
    assert tracks1[0].track_id == tracks2[0].track_id
