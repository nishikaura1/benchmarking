"""
benchmark/upload_to_huggingface.py — Publish the benchmark to HuggingFace Datasets.

Creates haptal-ai/robotics-failure-benchmark as a public dataset.

Usage
-----
  export HUGGINGFACE_TOKEN=hf_...
  python benchmark/upload_to_huggingface.py

Or pass token directly:
  python benchmark/upload_to_huggingface.py --token hf_...
"""

import argparse
import json
import os
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR      = BENCHMARK_DIR / "data"
REPO_ID       = "haptal-ai/robotics-failure-benchmark"


def create_dataset_card(metadata: dict) -> str:
    """Generate a HuggingFace dataset card (README.md)."""
    classes = metadata.get("classes", [])
    n       = metadata.get("n_per_class", 500)
    total   = metadata.get("total_episodes", 0)

    return f"""---
license: apache-2.0
task_categories:
  - robotics
  - time-series-classification
tags:
  - robotics
  - failure-detection
  - robot-learning
  - imitation-learning
  - lerobot
pretty_name: Haptal Robotics Failure Benchmark
size_categories:
  - 1K<n<10K
---

# Haptal Robotics Failure Benchmark v1.0

A synthetic failure detection benchmark for evaluating robot training data quality.
Built on real LeRobot manipulation trajectories with physics-based failure injection.

## Overview

Most robotics teams train on datasets containing 20–40% failure episodes without knowing it.
This benchmark provides a standardized way to evaluate how well a model can detect
and classify these failures.

## Failure Classes

| Class | Description |
|---|---|
| `grasp_slip` | Grip force drops suddenly — object slips |
| `velocity_spike` | Sudden overcorrection or joint jerk |
| `trajectory_deviation` | Gradual drift from intended path |
| `stuck_joint` | Motor stall or collision |
| `overcorrect` | Post-failure panic overcorrection |
| `nominal` | Clean episode — no failure |

## Dataset Stats

- **{n} episodes per class** × {len(classes)} classes = **{total:,} total episodes**
- 80/20 train/test split, stratified by class
- Base trajectories from: {metadata.get('base_datasets', ['lerobot/pusht'])}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{REPO_ID}", split="test")
df = ds.to_pandas()

# Features column contains pre-extracted episode-level feature vectors
import numpy as np
X = np.stack(df["features"].values)
y = df["failure_class"].values
```

## Benchmark Results (Haptal model)

See `benchmark/results.json` for full evaluation results.

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{{haptal2025,
  title={{Haptal Robotics Failure Benchmark}},
  author={{Bedi, Aarav}},
  year={{2025}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/datasets/{REPO_ID}}}}}
}}
```

## License

Apache 2.0. Base trajectories are from LeRobot (MIT license).
"""


def upload_benchmark(token: str = None, dry_run: bool = False) -> str:
    """Upload benchmark data to HuggingFace."""
    from huggingface_hub import HfApi, create_repo

    # ── Resolve token ─────────────────────────────────────────────────────────
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "No HuggingFace token found.\n"
            "Set it with: export HUGGINGFACE_TOKEN=hf_...\n"
            "Or get one at: https://huggingface.co/settings/tokens"
        )

    # ── Verify data exists ────────────────────────────────────────────────────
    for f in ["train.parquet", "test.parquet", "metadata.json"]:
        if not (DATA_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing {f} — run failure_injector.py first:\n"
                f"  python benchmark/failure_injector.py"
            )

    if dry_run:
        print(f"[dry-run] Would upload {DATA_DIR} → {REPO_ID}")
        return f"https://huggingface.co/datasets/{REPO_ID}"

    api = HfApi(token=token)

    # ── Create repo ───────────────────────────────────────────────────────────
    print(f"Creating dataset repo: {REPO_ID}...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )

    # ── Write dataset card ────────────────────────────────────────────────────
    meta = json.loads((DATA_DIR / "metadata.json").read_text())
    card = create_dataset_card(meta)
    (DATA_DIR / "README.md").write_text(card)

    # Also write results if they exist
    results_src = BENCHMARK_DIR / "results.json"
    if results_src.exists():
        import shutil
        shutil.copy(results_src, DATA_DIR / "results.json")

    # ── Upload all files ──────────────────────────────────────────────────────
    print(f"Uploading files from {DATA_DIR}...")
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        commit_message="Add Haptal Robotics Failure Benchmark v1.0",
    )

    url = f"https://huggingface.co/datasets/{REPO_ID}"
    print(f"\n✅ Published at: {url}")
    print(f"   Share this link with Sergey Levine + Ken Goldberg emails.")
    return url


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload benchmark to HuggingFace")
    parser.add_argument("--token",   type=str,  default=None,
                        help="HuggingFace API token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without actually uploading")
    args = parser.parse_args()

    upload_benchmark(token=args.token, dry_run=args.dry_run)
