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

if __name__ == "__main__":
    SRC = "/kaggle/input/datasets/talha972/garbagedataset-forcv/GarbageDataset"
    DST = "/kaggle/working/GarbageDataset"

    copy_dataset(SRC, DST)

    images = sorted(os.listdir(f"{DST}/images"))
    splits = split_dataset(images)

    for name, files in splits.items():
        counts = Counter(category(f) for f in files)
        print(f"{name:<5} {len(files):>5} images  {dict(sorted(counts.items()))}")