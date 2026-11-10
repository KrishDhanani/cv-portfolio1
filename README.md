# cv-portfolio1: Waste Object Detection (Computer Vision, WS 2026/27)

Portfolio 1 for the Computer Vision course at THWS (Prof. Dominik Seuß).
We train a YOLOv8 object detector that finds waste items in images and
classifies them into six classes:

| ID | Class |
|---|---|
| 0 | biodegradable |
| 1 | cardboard |
| 2 | glass |
| 3 | metal |
| 4 | paper |
| 5 | plastic |

## Team and tasks (Assignment 1)

| Person | Task | Folder |
|---|---|---|
| Ismail | Task 1a: dataset analysis, cleaning, train/val/test split | `task1_data/` |
| Talha | Task 1b: data pipeline: Kaggle data setup, image size, augmentation | `task1_data/` |
| Areej | Task 2: model training and evaluation | `task2_training/` |
| Krish | Task 3: visualization and error analysis | `task3_error_analysis/` |

## Repository structure

```
cv-portfolio1/
├── README.md
├── requirements.txt
├── task1_data/
│   ├── setup_data.py              Copies the cleaned split to /kaggle/working and writes data.yaml
│   ├── image_size_experiment.py   Compares imgsz 416 vs 640 (short YOLOv8n runs)
│   ├── show_augmentations.py      Figure with examples of the augmentations we use
│   └── archive/
│       └── prepare_data.py        First split on the raw data (replaced by the cleaned split)
├── task2_training/                Training and evaluation (YOLOv8s)
└── task3_error_analysis/          Correct, missed and wrong detections
```

## Dataset

- Original: Kaggle dataset `talha972/garbagedataset-forcv` (10,464 images, 416x416, YOLO format, Roboflow export)
- Cleaned and split: Kaggle dataset `talha972/garbage-split`
  - 9,405 images, 36,879 boxes after removing very crowded single-class biodegradable images
  - Split about 71 / 14 / 14: 6,717 train, 1,344 val, 1,344 test
  - Stratified by main class and single/mixed images, near-duplicates kept in the same split, seed 42

## How to run on Kaggle

1. Create a notebook, turn **Internet on** and choose a **GPU (T4)**.
2. Add the dataset **garbage-split** (Add Input, Datasets).
3. Run this cell:

```python
!rm -rf /tmp/repo
!git clone --depth 1 https://github.com/MTalhaKhan326/cv-portfolio1.git /tmp/repo
!pip install -q -r /tmp/repo/requirements.txt

# Task 1: prepare the data (creates /kaggle/working/GarbageSplit/data.yaml)
!python /tmp/repo/task1_data/setup_data.py --src /kaggle/input/datasets/talha972/garbage-split/GarbageSplit
```

4. Train with `data="/kaggle/working/GarbageSplit/data.yaml"` (see `task2_training/`).

Optional, Task 1 analysis scripts:

```python
!python /tmp/repo/task1_data/image_size_experiment.py --data /kaggle/working/GarbageSplit/data.yaml
!python /tmp/repo/task1_data/show_augmentations.py --data-dir /kaggle/working/GarbageSplit
```

## Agreed training settings (from Task 1)

| Setting | Value | Reason |
|---|---|---|
| `imgsz` | 416 | Native image size; 416 matched or beat 640 in our tests and is about 1.6x faster |
| `fliplr` | 0.5 | A mirrored object keeps its class |
| `flipud` | 0.0 | The dataset already contains some upside-down images |
| `hsv_h` | 0.015 | Small, because colour helps identify the material |
| `hsv_s`, `hsv_v` | 0.7, 0.4 | Web images with very different lighting |
| `mosaic` | 1.0 | Multi-object scenes and objects at different scales |
| `scale`, `translate` | 0.5, 0.1 | Objects appear at different sizes and positions |
| `degrees`, `mixup` | 0.0 | Rotation makes boxes inaccurate; blending materials creates unrealistic objects |

Train for clearly more than 10 epochs: YOLO switches mosaic off for the last 10 epochs.
Use the test set only for the final evaluation.
