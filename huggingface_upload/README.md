---
license: apache-2.0
task_categories:
  - robotics
  - tabular-classification
tags:
  - robotics
  - failure-detection
  - robot-training-data
  - annotation-quality
  - manipulation
pretty_name: Haptal Robotics Failure Benchmark v1.0
size_categories:
  - 1K<n<10K
---

# Haptal Robotics Failure Benchmark v1.0

The first public benchmark for robot training data annotation quality and failure detection in manipulation episodes.

## What this is

A held-out test set of 600 robot episodes across 6 failure classes generated from real LeRobot trajectories with physics-based failure injection. The test set is fixed. Anyone can evaluate their annotation pipeline against it and get a comparable score.

## Why it exists

No standardized benchmark currently exists for robot training data annotation quality. Open X-Embodiment, BridgeData V2, DROID and LeRobot are dataset collections. None measure annotation quality or failure detection accuracy. We built this to fill that gap.

## Failure classes

| Class | Description |
|---|---|
| `grasp_slip` | Grip force drops causing object slip |
| `nominal` | Successful episode, no failure |
| `overcorrect` | Post-failure panic response |
| `stuck_joint` | Motor stall or joint lock |
| `trajectory_deviation` | Drift from intended path |
| `velocity_spike` | Sudden joint velocity anomaly |

## Dataset splits

| Split | Episodes |
|---|---|
| Train | 2,400 |
| Test (fixed) | 600 |

## Leaderboard

| Rank | Model | Accuracy | Macro F1 | Cohen's κ | OOD F1 | Gap |
|---|---|---|---|---|---|---|
| 🥇 | Haptal (multi-dataset RF) | **93.6%** | **0.937** | **0.923** | **0.907** | **0.030** |
| — | Human operator (pass/fail only)* | 83.1% | — | 0.661 | — | — |
| — | Majority baseline | 53.1% | — | 0.000 | — | — |

\* Human operators provide binary pass/fail only — no failure type, no timestep.  
Submit your model → `aarav@haptal.ai` · see `leaderboard.json` for full entry details.

## Haptal baseline results

| Metric | Value |
|---|---|
| In-distribution accuracy | 93.6% |
| OOD accuracy (unseen robot) | 90.8% |
| Generalization gap | 0.03 |
| Macro F1 | 0.937 |
| Cohen's Kappa | 0.923 |
| Weakest class F1 | 0.887 (grasp_slip) |

## How to use

```python
from datasets import load_dataset

dataset = load_dataset("haptal-ai/robotics-failure-benchmark")
train = dataset["train"]
test  = dataset["test"]
```

## How to score your model

1. Run your model on the test split
2. Save predictions as a CSV with columns `episode_id` and `predicted_class`
3. Run `python score.py your_predictions.csv`
4. Email results to [aarav@haptal.ai](mailto:aarav@haptal.ai) to be added to the leaderboard

## Base datasets

Generated from real robot trajectories across multiple platforms:

- `lerobot/pusht`
- `lerobot/xarm_lift_medium_replay`
- `lerobot/xarm_push_medium_replay`
- `lerobot/aloha_sim_transfer_cube_human`

## License

Apache 2.0. Base trajectories from LeRobot (MIT license).

## Citation

```bibtex
@dataset{haptal2026rfb,
  title   = {Haptal Robotics Failure Benchmark v1.0},
  author  = {Bedi, Aarav},
  year    = {2026},
  publisher = {HuggingFace},
  url     = {https://huggingface.co/datasets/haptal-ai/robotics-failure-benchmark}
}
```

## Contact

[aarav@haptal.ai](mailto:aarav@haptal.ai)  
[haptal.ai](https://haptal.ai)
