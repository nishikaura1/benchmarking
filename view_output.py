"""
View benchmark output — prints HDF5 scores and renders a plot.
Usage: python view_output.py
       python view_output.py benchmark_output/lerobot_pusht_scores.h5
"""

import sys
import json
import h5py
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def view(h5_path: Path):
    card_path = h5_path.with_name(h5_path.stem.replace("_scores", "_card") + ".json")

    print(f"\n{'='*60}")
    print(f"File: {h5_path}")

    if card_path.exists():
        card = json.loads(card_path.read_text())
        print(f"\nBENCHMARK CARD")
        print(f"  Dataset          : {card['dataset']}")
        print(f"  Model            : {card['model']}")
        print(f"  Total episodes   : {card['total_episodes']}")
        print(f"  Failure episodes : {card['failure_episodes']}")
        print(f"  ROC-AUC          : {card['roc_auc']}")
        print(f"  Detection rate   : {card['detection_rate_pct']}%")
        print(f"  False pos. rate  : {card['false_positive_rate_pct']}%")
        cm = card["confusion_matrix"]
        print(f"  Confusion matrix : TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")

    with h5py.File(h5_path, "r") as f:
        scores = f["anomaly_scores"][:]
        labels = f["true_labels"][:]
        preds  = f["predictions"][:]

    print(f"\nANOMALY SCORES")
    print(f"  Min    : {scores.min():.4f}")
    print(f"  Max    : {scores.max():.4f}")
    print(f"  Mean   : {scores.mean():.4f}")
    print(f"  Median : {np.median(scores):.4f}")

    print(f"\nPER-EPISODE BREAKDOWN (first 20)")
    print(f"  {'Ep':>4}  {'Score':>8}  {'True':>6}  {'Pred':>6}  Result")
    for i in range(min(20, len(scores))):
        result = "OK" if preds[i] == labels[i] else "WRONG"
        true_str = "FAIL" if labels[i] else "OK  "
        pred_str = "FAIL" if preds[i] else "OK  "
        print(f"  {i:>4}  {scores[i]:>8.4f}  {true_str:>6}  {pred_str:>6}  {result}")

    if HAS_PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Anomaly Detection — {h5_path.stem}", fontsize=13)

        # score distribution
        ax = axes[0]
        nom_scores = scores[labels == 0]
        fail_scores = scores[labels == 1]
        ax.hist(nom_scores, bins=30, alpha=0.6, color="steelblue", label="Nominal")
        ax.hist(fail_scores, bins=30, alpha=0.6, color="crimson", label="Failure")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend()

        # scores over episodes
        ax = axes[1]
        colors = ["crimson" if l else "steelblue" for l in labels]
        ax.scatter(range(len(scores)), scores, c=colors, s=10, alpha=0.6)
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Anomaly Score")
        ax.set_title("Scores per Episode  (red = failure)")

        plt.tight_layout()
        plot_path = h5_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=120)
        print(f"\nPlot saved: {plot_path}")
        plt.show()
    else:
        print("\nInstall matplotlib for plots: pip install matplotlib")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view(Path(sys.argv[1]))
    else:
        h5_files = sorted(Path("benchmark_output").glob("*_scores.h5"))
        if not h5_files:
            print("No output files found. Run: python main.py --source synthetic")
        for f in h5_files:
            view(f)
