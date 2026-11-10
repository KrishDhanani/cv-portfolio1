"""Ground-truth/prediction matching and error analysis — TODO.

Implement IoU-based matching and categorize outcomes as:
- correct detections (matched class and IoU threshold),
- missed ground-truth objects (false negatives),
- incorrect predictions (false positives and/or wrong-class matches).

Agree on IoU and confidence thresholds with the team and document them.
"""


def analyze_image(ground_truth, predictions, iou_threshold: float = 0.5):
    """TODO: Match predictions to ground truth and return error categories."""
    raise NotImplementedError
