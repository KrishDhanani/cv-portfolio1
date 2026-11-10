"""
Part 2 - Kaggle data setup for the team's final split.

Person 1 cleaned the dataset and split it into train / val / test
(GarbageSplit). This script prepares that split for YOLO training on Kaggle:
    1. Copy the split from the read-only Kaggle input to /kaggle/working.
    2. Write data.yaml with the full path, so YOLO finds the data.
    3. Print images, boxes and box sizes per split to check the setup.

Usage:
    python setup_data.py --src /kaggle/input/datasets/talha972/garbage-split/GarbageSplit
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
SPLITS = ("train", "val", "test")


def copy_tree(src, dst, workers=64):
    """Copy all files from src to dst in parallel. Files that already exist are skipped."""
    jobs = []
    for root, _, files in os.walk(src):
        target_dir = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            target = os.path.join(target_dir, name)
            if not os.path.exists(target):
                jobs.append((os.path.join(root, name), target))

    print(f"[copy] files to copy: {len(jobs)}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: shutil.copyfile(*pair), jobs))
    print(f"[copy] done -> {dst}")


def write_data_yaml(data_dir):
    """Write data.yaml: where the images are, and what the class numbers mean."""
    lines = [f"path: {data_dir}",
             "train: images/train",
             "val: images/val",
             "test: images/test",
             "",
             "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASSES)]

    path = os.path.join(data_dir, "data.yaml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def report(data_dir):
    """Print images, boxes, the most boxes in one image, and box sizes per split."""
    print(f"\n{'split':<6} {'images':>7} {'boxes':>7} {'max/img':>8} {'small<1%':>9} {'large>50%':>10}")
    for split in SPLITS:
        label_dir = os.path.join(data_dir, "labels", split)
        boxes_per_image, areas = [], []
        for name in os.listdir(label_dir):
            with open(os.path.join(label_dir, name)) as f:
                rows = [line.split() for line in f if line.strip()]
            boxes_per_image.append(len(rows))
            areas += [float(r[3]) * float(r[4]) for r in rows]

        small = 100 * sum(a < 0.01 for a in areas) / len(areas)
        large = 100 * sum(a > 0.5 for a in areas) / len(areas)
        print(f"{split:<6} {len(boxes_per_image):>7} {len(areas):>7} {max(boxes_per_image):>8} "
              f"{small:>8.1f}% {large:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Prepare the team's split for YOLO training on Kaggle.")
    parser.add_argument("--src", required=True, help="GarbageSplit folder in /kaggle/input")
    parser.add_argument("--dst", default="/kaggle/working/GarbageSplit", help="Writable copy")
    args = parser.parse_args()

    copy_tree(args.src, args.dst)

    yaml_path = write_data_yaml(args.dst)
    print(f"\n[yaml] {yaml_path}\n" + open(yaml_path).read())

    report(args.dst)


if __name__ == "__main__":
    main()