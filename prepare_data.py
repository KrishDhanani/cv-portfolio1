import argparse
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split


# The 6 classes in the same order as the IDs in the label files (0-5)
CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]


def category(filename):
    """re is Python’s tool for finding text patterns.
    r"[a-z]+" means “one or more lowercase letters”.
    Return the waste category from a file name.
    Example: 'glass12_jpg.rf.abc.jpg' -> 'glass'
    """
    return re.match(r"[a-z]+", filename).group()


def label_name(image_name):
    """Return the label file that belongs to an image.
    Example: 'glass12.jpg' -> 'glass12.txt'
    """
    return os.path.splitext(image_name)[0] + ".txt"

def copy_dataset(src, dst, workers=64):
    """Copy images and labels to a fast, writable folder.
    Files that already exist are skipped.
    """
    jobs = []
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        for name in os.listdir(os.path.join(src, sub)):
            target = os.path.join(dst, sub, name)
            if not os.path.exists(target):
                jobs.append((os.path.join(src, sub, name), target))

    print(f"[copy] files to copy: {len(jobs)}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: shutil.copyfile(*pair), jobs))
    print(f"[copy] done -> {dst}")

def split_dataset(images, train_ratio=0.70, seed=42):
    """Split image names into train / val / test, stratified by category.
    The rest after training is divided equally into validation and test.
    """
    cats = [category(f) for f in images]

    # First split: 70% train, 30% rest
    train, rest, _, rest_cats = train_test_split(
        images, cats, train_size=train_ratio, stratify=cats, random_state=seed
    )
    # Second split: the 30% rest -> 15% val, 15% test
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest_cats, random_state=seed
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

def link_or_copy(src, dst):
    """Put a file into the new folder using a hard link (instant, no extra space).
    If that is not possible, copy it instead.
    """
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_yolo_folders(source, out_dir, splits):
    """Create out_dir/<split>/images and labels, plus a <split>.txt file list."""
    for split, files in splits.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)

        for name in files:
            link_or_copy(os.path.join(source, "images", name),
                         os.path.join(out_dir, split, "images", name))
            link_or_copy(os.path.join(source, "labels", label_name(name)),
                         os.path.join(out_dir, split, "labels", label_name(name)))

        with open(os.path.join(out_dir, f"{split}.txt"), "w") as f:
            f.write("\n".join(files))

def write_data_yaml(out_dir):
    """Write data.yaml: tells YOLO where the data is and what the classes are."""
    path = os.path.join(out_dir, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")
    return path

def report_split(source, splits):
    """Print images, boxes and the share of boxes per class in each split."""
    print(f"{'split':<6} {'images':>7} {'boxes':>7}  " + "  ".join(f"{c[:7]:>7}" for c in CLASSES))
    for split, files in splits.items():
        boxes = Counter()
        for name in files:
            with open(os.path.join(source, "labels", label_name(name))) as f:
                for line in f:
                    if line.strip():
                        boxes[int(line.split()[0])] += 1
        total = sum(boxes.values())
        shares = "  ".join(f"{100 * boxes[i] / total:6.1f}%" for i in range(len(CLASSES)))
        print(f"{split:<6} {len(files):>7} {total:>7}  {shares}")

def main():
    parser = argparse.ArgumentParser(description="Prepare the garbage dataset for YOLO training.")
    parser.add_argument("--src", required=True, help="Original dataset folder with images/ and labels/")
    parser.add_argument("--work", default="/kaggle/working", help="Writable working directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Share of images for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for a reproducible split")
    args = parser.parse_args()

    local_copy = os.path.join(args.work, "GarbageDataset")
    out_dir = os.path.join(args.work, "data")

    copy_dataset(args.src, local_copy)

    images = sorted(os.listdir(os.path.join(local_copy, "images")))
    splits = split_dataset(images, args.train_ratio, args.seed)
    build_yolo_folders(local_copy, out_dir, splits)

    yaml_path = write_data_yaml(out_dir)
    print(f"\n[yaml] {yaml_path}\n" + open(yaml_path).read())

    report_split(local_copy, splits)


if __name__ == "__main__":
    main()