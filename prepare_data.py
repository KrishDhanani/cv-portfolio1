import os
import re

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


if __name__ == "__main__":
    # Quick test of the two functions
    test_files = [
        "biodegradable1000_jpg.rf.9407ff66b4ccd9f8eb5545be8f3225f4.jpg",
        "glass12_jpeg.rf.abc123.jpg",
        "plastic7_jpg.rf.xyz.jpg",
    ]
    for f in test_files:
        print(f"{f}\n  category: {category(f)}\n  label:    {label_name(f)}\n")