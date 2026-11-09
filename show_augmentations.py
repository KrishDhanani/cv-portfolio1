"""
Part 2 - Example pictures of data augmentation.

Takes real training images from the team's cleaned split and shows the augmentations we use, plus one
we deliberately do NOT use (strong hue shift), with the boxes drawn in.
The result is one figure for the presentation.

Note: this recreates the augmentations in a simple, visible way to explain
them. During training, YOLO applies them randomly with the settings
fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, mosaic=1.0.

Lecture reference: Recap 1, slides 205-210 (data augmentation,
"probably the same class" vs. "not the same class").

Usage:
    python show_augmentations.py --data-dir /kaggle/working/GarbageSplit
"""

import argparse
import os
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
COLORS = [(46, 160, 67), (180, 120, 60), (40, 120, 220), (150, 150, 150), (230, 200, 60), (220, 60, 160)]


def category(filename):
    """Waste category from a file name, e.g. 'glass12_jpg...' -> 'glass'."""
    return re.match(r"[a-z]+", filename).group()


def load_image(data_dir, name):
    """Load one training image and its boxes [(class, x, y, w, h), ...] (YOLO format, 0-1)."""
    img = cv2.imread(os.path.join(data_dir, "images", "train", name))
    label_path = os.path.join(data_dir, "labels", "train", os.path.splitext(name)[0] + ".txt")
    with open(label_path) as f:
        labels = [(int(p[0]), *map(float, p[1:5])) for p in (l.split() for l in f) if p]
    return img, labels


def draw_boxes(img, labels):
    """Draw the boxes and class names on a copy of the image."""
    out = img.copy()
    h, w = out.shape[:2]
    for c, x, y, bw, bh in labels:
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[c], 2)
        cv2.putText(out, CLASSES[c], (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[c], 1, cv2.LINE_AA)
    return out


def flip_horizontal(img, labels):
    """Mirror the image left-right; the box centres move to 1 - x."""
    return img[:, ::-1].copy(), [(c, 1 - x, y, w, h) for c, x, y, w, h in labels]


def change_colors(img, hue_shift, sat_gain, val_gain):
    """Change colour tone (hue), colour strength (saturation) and brightness (value)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def zoom_out(img, labels, factor):
    """Shrink the image into the centre of a grey canvas, so objects look smaller."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * factor), int(h * factor)))
    canvas = np.full_like(img, 114)
    top, left = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    new_labels = [(c, 0.5 + (x - 0.5) * factor, 0.5 + (y - 0.5) * factor, bw * factor, bh * factor)
                  for c, x, y, bw, bh in labels]
    return canvas, new_labels


def mosaic(samples, size=416):
    """Put four images into one 2x2 image (simplified version of YOLO's mosaic)."""
    half = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    labels = []
    for i, (img, img_labels) in enumerate(samples):
        row, col = divmod(i, 2)
        canvas[row * half:(row + 1) * half, col * half:(col + 1) * half] = cv2.resize(img, (half, half))
        labels += [(c, (x + col) / 2, (y + row) / 2, bw / 2, bh / 2) for c, x, y, bw, bh in img_labels]
    return canvas, labels


def pick_examples(data_dir, seed):
    """Pick one glass image with a few boxes, and four images of different categories for the mosaic."""
    rng = random.Random(seed)
    names = sorted(os.listdir(os.path.join(data_dir, "images", "train")))

    def has_few_boxes(name, low=2, high=6):
        return low <= len(load_image(data_dir, name)[1]) <= high

    glass = [n for n in names if category(n) == "glass"]
    rng.shuffle(glass)
    main_name = next(n for n in glass if has_few_boxes(n))

    mosaic_names = []
    for cat in ("biodegradable", "cardboard", "metal", "plastic"):
        options = [n for n in names if category(n) == cat]
        rng.shuffle(options)
        mosaic_names.append(next(n for n in options if has_few_boxes(n, 1, 8)))
    return main_name, mosaic_names


def main():
    parser = argparse.ArgumentParser(description="Show example pictures of the data augmentation.")
    parser.add_argument("--data-dir", default="/kaggle/working/GarbageSplit", help="Folder created by setup_data.py")
    parser.add_argument("--out", default="/kaggle/working/augmentation_examples.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main_name, mosaic_names = pick_examples(args.data_dir, args.seed)
    img, labels = load_image(args.data_dir, main_name)

    panels = [
        ("Original", img, labels),
        ("Horizontal flip (fliplr)", *flip_horizontal(img, labels)),
        ("Brightness + saturation (hsv_v, hsv_s)", change_colors(img, 0, 1.5, 0.6), labels),
        ("Zoom out (scale)", *zoom_out(img, labels, 0.6)),
        ("Mosaic of 4 images (mosaic)", *mosaic([load_image(args.data_dir, n) for n in mosaic_names])),
        ("NOT used: strong hue shift", change_colors(img, 0.4, 1.0, 1.0), labels),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (title, panel_img, panel_labels) in zip(axes.flat, panels):
        ax.imshow(cv2.cvtColor(draw_boxes(panel_img, panel_labels), cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    print(f"Main image: {main_name}")
    print(f"Mosaic images: {mosaic_names}")


if __name__ == "__main__":
    main()