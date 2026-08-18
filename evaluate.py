"""
evaluate.py — Real metrics, not vibes
========================================
Runs the full pipeline against a labeled test set and reports:
  - Detection:  precision / recall / F1 at IoU >= 0.5
  - OCR:        character-level accuracy (1 - normalized edit distance)
  - End-to-end: exact-match plate accuracy (detected AND read correctly)

Expected folder layout:
    test_set/
        images/          *.jpg, *.png ...
        labels.json      ground truth (see format below)

labels.json format — one entry per image, plate text + box in pixel coords:
    {
      "car001.jpg": [
        {"bbox": [120, 340, 260, 400], "text": "1234 ABC"}
      ],
      "car002.jpg": [
        {"bbox": [80, 210, 210, 265], "text": "٥٦٧٨ ط"}
      ]
    }

Run:
    python evaluate.py --test-dir test_set
"""

import argparse
import json
from pathlib import Path

import cv2

import pipeline

IOU_MATCH_THRESHOLD = 0.5


def edit_distance(a: str, b: str) -> int:
    """Plain Levenshtein distance — no extra dependency needed."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def char_accuracy(pred: str, gt: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    dist = edit_distance(pred.replace(" ", ""), gt.replace(" ", ""))
    return max(0.0, 1.0 - dist / max(len(gt), 1))


def evaluate(test_dir: Path):
    labels_path = test_dir / "labels.json"
    images_dir = test_dir / "images"
    if not labels_path.exists():
        raise SystemExit(f"labels.json not found in {test_dir}")

    ground_truth = json.loads(labels_path.read_text(encoding="utf-8"))

    total_gt_boxes = 0
    total_pred_boxes = 0
    true_positive_boxes = 0

    char_acc_scores = []
    exact_matches = 0
    matched_pairs = 0

    for filename, gt_entries in ground_truth.items():
        img_path = images_dir / filename
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] could not read {img_path}")
            continue

        results, _ = pipeline.process_frame(frame)
        total_gt_boxes += len(gt_entries)
        total_pred_boxes += len(results)

        used_preds = set()
        for gt in gt_entries:
            gt_box = gt["bbox"]
            gt_text = gt["text"]

            best_iou, best_idx = 0.0, -1
            for i, pred in enumerate(results):
                if i in used_preds:
                    continue
                score = pipeline.iou(tuple(gt_box), pred["bbox"])
                if score > best_iou:
                    best_iou, best_idx = score, i

            if best_iou >= IOU_MATCH_THRESHOLD and best_idx != -1:
                true_positive_boxes += 1
                used_preds.add(best_idx)
                matched_pairs += 1

                pred_text = results[best_idx]["plate_text"]
                acc = char_accuracy(pred_text, gt_text)
                char_acc_scores.append(acc)
                if pred_text.replace(" ", "") == gt_text.replace(" ", ""):
                    exact_matches += 1

    precision = true_positive_boxes / total_pred_boxes if total_pred_boxes else 0.0
    recall = true_positive_boxes / total_gt_boxes if total_gt_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_char_acc = sum(char_acc_scores) / len(char_acc_scores) if char_acc_scores else 0.0
    e2e_acc = exact_matches / total_gt_boxes if total_gt_boxes else 0.0

    print("\n=== Detection (IoU >= 0.5) ===")
    print(f"Ground truth boxes: {total_gt_boxes}")
    print(f"Predicted boxes:    {total_pred_boxes}")
    print(f"Precision:          {precision:.3f}")
    print(f"Recall:             {recall:.3f}")
    print(f"F1:                 {f1:.3f}")

    print("\n=== OCR (on matched boxes) ===")
    print(f"Matched pairs:          {matched_pairs}")
    print(f"Avg character accuracy: {avg_char_acc:.3f}")

    print("\n=== End-to-end ===")
    print(f"Exact-match plate accuracy: {e2e_acc:.3f}  ({exact_matches}/{total_gt_boxes})")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "avg_char_accuracy": avg_char_acc, "end_to_end_accuracy": e2e_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the plate pipeline against a labeled test set.")
    parser.add_argument("--test-dir", default="test_set", help="Folder with images/ and labels.json")
    args = parser.parse_args()
    evaluate(Path(args.test_dir))
