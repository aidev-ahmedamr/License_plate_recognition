"""
train_detector.py — Fine-tune YOLOv8 on your own license-plate dataset
==========================================================================
Generic yolov8n.pt was trained on COCO (people, cars, cats...) — it has no
concept of "license plate" as a class. This script fine-tunes it on a
plate-only dataset so detection actually works.

1. Get a dataset (pick one):
   - Roboflow Universe: search "license plate" / "Egyptian license plate",
     export in "YOLOv8" format. You'll get a folder with:
         data.yaml
         train/images, train/labels
         valid/images, valid/labels
         test/images,  test/labels  (optional)
   - Or label your own with Roboflow / CVAT / LabelImg and export the same way.

2. Point DATASET_YAML below at that data.yaml.

3. Run:
       pip install ultralytics
       python train_detector.py

4. The best weights land in runs/detect/train/weights/best.pt — copy that
   path into PLATE_MODEL_PATH when running plate_reader.py / api.py:
       set PLATE_MODEL_PATH=runs/detect/train/weights/best.pt
"""

import argparse

from ultralytics import YOLO

DEFAULT_DATASET_YAML = "dataset/data.yaml"
DEFAULT_BASE_MODEL = "yolov8n.pt"   # small + fast; use yolov8s.pt for a bit more accuracy
DEFAULT_EPOCHS = 80
DEFAULT_IMG_SIZE = 640


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for license plate detection.")
    parser.add_argument("--data", default=DEFAULT_DATASET_YAML, help="Path to dataset's data.yaml")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'cpu', '0' for first GPU, etc. Auto-detected if omitted.")
    args = parser.parse_args()

    print(f"Fine-tuning {args.base_model} on {args.data} for {args.epochs} epochs...")
    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=15,          # early stop if val metric plateaus
        project="runs/detect",
        name="train",
        val=True,
        plots=True,
    )

    metrics = model.val()
    print("\n--- Validation metrics ---")
    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print("\nBest weights saved to: runs/detect/train/weights/best.pt")
    print("Point PLATE_MODEL_PATH at that file to use it in plate_reader.py / api.py.")


if __name__ == "__main__":
    main()
