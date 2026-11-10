# Dataset location

The dataset is intentionally not included in this repository.

Suggested local layout after preparing the dataset in YOLO format:

```text
data/
├── raw/
└── processed/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Update `configs/data.yaml` to match the actual folder structure and class names. Keep large/private dataset files out of Git unless the course rules and team storage plan explicitly allow them.
