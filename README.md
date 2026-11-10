# CV Portfolio 1 — Garbage Detection

A collaborative computer-vision project for Assignment 1: object detection for automated waste sorting.

## Project status
Repository scaffold only. Replace the TODO sections and dummy files with the team's completed work.

## Team responsibilities

| Member | Task | Main location |
|---|---|---|
| Person 1 | Task 1a — Dataset analysis | `notebooks/01_dataset_analysis/`, `results/dataset_analysis/` |
| Person 2 | Task 1b — Data pipeline | `notebooks/02_data_pipeline/`, `configs/`, `src/data/` |
| Person 3 | Task 2 — Training and evaluation | `notebooks/03_training_evaluation/`, `src/training/`, `src/evaluation/`, `models/`, `results/` |
| Person 4 | Task 3 — Visualization and error analysis | `notebooks/04_visualization_error_analysis/`, `src/visualization/`, `results/error_analysis/` |

## Repository layout

- `configs/`: dataset and experiment configuration.
- `data/`: dataset location instructions. Dataset files are intentionally not included.
- `notebooks/`: task-specific exploratory and reporting notebooks.
- `src/`: reusable Python implementation.
- `models/`: trained model storage instructions. Large weights should not normally be committed directly.
- `results/`: generated metrics, plots, and error-analysis examples.
- `reports/`: final figures and presentation material.
- `docs/`: methodology, project overview, and contribution records.
- `tests/`: basic automated tests.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset setup

1. Obtain the dataset from the source provided by the course.
2. Keep the original dataset under `data/raw/` (or configure another local path).
3. Prepare the YOLO-format train/validation/test folders according to the pipeline.
4. Update `configs/data.yaml` with the actual paths and class names.

The dataset is not included in this repository.

## Running code

Reusable code is under `src/`. Example commands will be added once the team's training and evaluation scripts are finalized.

## Collaboration

Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes. Work on a task branch and open a pull request into `main`.

## Reproducibility

Record random seeds, package versions, dataset split details, model settings, and evaluation settings. Do not report placeholder values as experimental results.
