"""
huggingface_upload/score.py — Score predictions against the Haptal benchmark test set.

Takes a CSV of predictions and returns standard classification metrics.
Does NOT expose feature extraction logic or model internals.

Usage
-----
  python huggingface_upload/score.py predictions.csv
  python huggingface_upload/score.py predictions.csv --test-path data/test.parquet

Predictions CSV format
----------------------
  episode_id,predicted_class
  grasp_slip_0000,nominal
  grasp_slip_0001,grasp_slip
  ...
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)


def score_predictions(predictions_path: str,
                      test_path: str = "data/test.parquet") -> dict:
    """
    Score your model's predictions against the Haptal benchmark test set.

    Parameters
    ----------
    predictions_path : str
        Path to a CSV file with columns:
          - episode_id      : must match test set episode IDs
          - predicted_class : your model's predicted failure class

    test_path : str
        Path to the benchmark test parquet (default: data/test.parquet
        relative to this script's directory).

    Returns
    -------
    dict with accuracy, macro_f1, cohen_kappa, episodes_scored, per_class_f1
    """
    # Resolve test path relative to this script if not absolute
    if not Path(test_path).is_absolute():
        script_dir = Path(__file__).parent
        test_path  = str(script_dir / test_path)

    # Load ground truth
    test = pd.read_parquet(test_path)

    # Load predictions
    preds = pd.read_csv(predictions_path)

    # Validate format
    required_cols = ["episode_id", "predicted_class"]
    for col in required_cols:
        if col not in preds.columns:
            raise ValueError(
                f"Predictions file must have column: '{col}'\n"
                f"Found columns: {list(preds.columns)}"
            )

    # Merge on episode_id
    merged = test.merge(preds, on="episode_id", how="inner")

    if len(merged) == 0:
        raise ValueError(
            "No episode_ids matched between predictions and test set. "
            "Check that your episode_id values match the test parquet."
        )

    if len(merged) < len(test):
        print(f"Warning: {len(merged)}/{len(test)} test episodes scored "
              f"({len(test) - len(merged)} missing from predictions)")

    y_true  = merged["failure_class"]
    y_pred  = merged["predicted_class"]
    classes = sorted(y_true.unique())

    f1_per_class = f1_score(y_true, y_pred, average=None,
                            labels=classes, zero_division=0)

    results = {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":        round(float(f1_score(y_true, y_pred,
                                                average="macro",
                                                zero_division=0)), 4),
        "cohen_kappa":     round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "episodes_scored": len(merged),
        "per_class_f1":    {
            cls: round(float(f1), 4)
            for cls, f1 in zip(classes, f1_per_class)
        },
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Score predictions against the Haptal benchmark test set"
    )
    parser.add_argument("predictions",  type=str,
                        help="Path to predictions CSV (episode_id, predicted_class)")
    parser.add_argument("--test-path",  type=str, default="data/test.parquet",
                        help="Path to test parquet (default: data/test.parquet)")
    args = parser.parse_args()

    score_predictions(args.predictions, test_path=args.test_path)
