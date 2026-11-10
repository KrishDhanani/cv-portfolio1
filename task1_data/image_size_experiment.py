"""
Part 2 - Image size experiment.

Trains two short YOLOv8n models, at 416 and at 640, on the same split
and compares their validation scores. Used to justify the image size.

Lecture references: Object Detection slides 17-19 (SPPF / PANet help
small objects) and slide 22 (multi-scale grids).

Usage:
    python image_size_experiment.py --data /kaggle/working/data/data.yaml
"""

import argparse

from ultralytics import YOLO


def train_and_score(data, imgsz, epochs, seed):
    """Train a small model at one image size and return its validation scores."""
    model = YOLO("yolov8n.pt")
    model.train(data=data, imgsz=imgsz, epochs=epochs, seed=seed,
                max_det=400, name=f"imgsz_{imgsz}", exist_ok=True, plots=False)
    metrics = model.val(data=data, imgsz=imgsz, split="val", max_det=400)
    return metrics.box.map50, metrics.box.map


def main():
    parser = argparse.ArgumentParser(description="Compare image sizes for YOLO training.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--sizes", type=int, nargs="+", default=[416, 640])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {size: train_and_score(args.data, size, args.epochs, args.seed)
               for size in args.sizes}

    print("\nimgsz   mAP@0.5   mAP@0.5:0.95")
    for size, (map50, map5095) in results.items():
        print(f"{size:<7} {map50:8.3f}   {map5095:8.3f}")


if __name__ == "__main__":
    main()