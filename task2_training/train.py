"""Task 2: train and evaluate a YOLOv8 detector on the garbage dataset.

Steps
1. Train YOLOv8 (COCO-pretrained) on the train split, validate on val each epoch.
2. Keep the best checkpoint (highest val fitness) via early stopping.
3. Evaluate best.pt ONCE on the held-out test split.
4. Save overall metrics, a per-class table and the confusion matrix.

Usage
    python train.py --data data/data.yaml                    # final: imgsz 416
    python train.py --data data/data.yaml --imgsz 640 --name yolov8s_baseline   # baseline
    python train.py --data data/data.yaml --aug augmentation.yaml   # settings from Task 1b

The augmentation file is a small YAML of Ultralytics augmentation keys,
for example:  fliplr: 0.5, hsv_v: 0.3, mosaic: 1.0
"""
import argparse
import json
import os

import yaml
from ultralytics import YOLO


# Training settings. Each value is documented with the reason it was chosen.
DEFAULTS = {
    "model": "yolov8s.pt",  # small model: more capacity than 'n' for tiny objects, fits a T4/P100
    "imgsz": 416,           # chosen in Task 1b (416 vs 640 experiment); baseline run used 640
    "epochs": 100,          # upper bound, early stopping usually ends earlier
    "patience": 20,         # stop if val mAP does not improve for 20 epochs
    "batch": 16,            # fits 16 GB GPU memory at 640
    "optimizer": "SGD",     # stable, well tested for fine-tuning YOLO
    "lr0": 0.01,            # initial learning rate (Ultralytics default for SGD)
    "lrf": 0.01,            # final LR = lr0 * lrf (linear decay)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,     # slow start so pretrained weights are not destroyed
    "seed": 42,             # same seed as the data split -> reproducible
    "max_det": 400,         # some images contain up to 358 objects (default 300 would cap recall)
}


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate YOLOv8 on the garbage dataset.")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--aug", default=None, help="Optional YAML with augmentation settings")
    p.add_argument("--project", default="runs", help="Output folder")
    p.add_argument("--name", default="yolov8s_416", help="Run name")
    p.add_argument("--device", default=None, help="'0' for GPU, 'cpu', 'mps'")
    for key, value in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(value), default=value)
    return p.parse_args()


def load_augmentation(path):
    """Read augmentation settings (Task 1b). Empty dict = Ultralytics defaults."""
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def train(args):
    """Fine-tune a COCO-pretrained YOLOv8 model and return the path to best.pt."""
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        project=os.path.abspath(args.project),
        name=args.name,
        exist_ok=True,
        device=args.device,
        max_det=args.max_det,
        plots=True,
        **load_augmentation(args.aug),
    )
    return os.path.join(model.trainer.save_dir, "weights", "best.pt")


def evaluate(weights, args):
    """Evaluate on the TEST split once and save metrics + confusion matrix."""
    model = YOLO(weights)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.abspath(args.project),
        name=f"{args.name}_test",
        exist_ok=True,
        device=args.device,
        max_det=args.max_det,
        plots=True,  # writes confusion_matrix.png and PR curves
    )
    out_dir = str(metrics.save_dir)
    box = metrics.box
    names = model.names

    per_class = []
    for i, cls in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({
            "class": names[int(cls)],
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50-95": round(float(ap), 4),
        })

    summary = {
        "weights": weights,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTEST  mAP@0.5={summary['mAP50']:.3f}  mAP@0.5:0.95={summary['mAP50-95']:.3f}  "
          f"P={summary['precision']:.3f}  R={summary['recall']:.3f}")
    print(f"{'class':<14}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for row in per_class:
        print(f"{row['class']:<14}{row['precision']:>8.3f}{row['recall']:>8.3f}"
              f"{row['mAP50']:>8.3f}{row['mAP50-95']:>10.3f}")
    print(f"\nSaved to {out_dir}")
    return summary


def main():
    args = parse_args()
    best = train(args)
    evaluate(best, args)


if __name__ == "__main__":
    main()
