"""
Robotics Anomaly Detection Benchmark
Runs IsolationForest on Open X-Embodiment + LeRobot datasets.
Outputs a benchmark card showing detection rate vs. false positive rate.
"""

import numpy as np
import h5py
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)

CONTAMINATION = 0.05   # assumed fraction of anomalies in training set
CONFIDENCE_THRESHOLD = 0.75
RANDOM_SEED = 42


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_episode_features(episode: dict) -> np.ndarray:
    """
    Pull joint positions, velocities, and gripper state into a flat feature vector.
    Works with Open X-Embodiment observation schema.
    """
    obs = episode.get("observation", episode)
    parts = []

    for key in ["joint_positions", "joint_velocities", "joint_torques"]:
        if key in obs:
            arr = np.array(obs[key], dtype=np.float32).flatten()
            parts.append(arr)

    for key in ["gripper_position", "gripper_closed"]:
        if key in obs:
            arr = np.array(obs[key], dtype=np.float32).flatten()
            parts.append(arr)

    if not parts:
        raise ValueError(f"No recognised sensor keys found. Keys present: {list(obs.keys())}")

    features = np.concatenate(parts)
    # summary stats across the time dimension if multi-step
    if features.ndim > 1:
        features = np.concatenate([features.mean(0), features.std(0), features.max(0)])

    return features


def extract_step_features(step: dict) -> np.ndarray:
    """Per-step feature extraction for LeRobot-style flat steps."""
    parts = []
    for key in ["observation.state", "action"]:
        if key in step:
            parts.append(np.array(step[key], dtype=np.float32).flatten())
    if not parts:
        raise ValueError(f"No recognised keys. Keys: {list(step.keys())}")
    return np.concatenate(parts)


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_open_x_embodiment(dataset_name: str = "fractal20220817_data", max_episodes: int = 500):
    """Load episodes from Open X-Embodiment via tensorflow-datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        raise ImportError("Run: pip install tensorflow-datasets tensorflow")

    print(f"Loading {dataset_name} (up to {max_episodes} episodes)...")
    ds = tfds.load(dataset_name, split="train", shuffle_files=False)

    features_list, labels = [], []
    for i, episode in enumerate(ds.take(max_episodes)):
        if i % 50 == 0:
            print(f"  Processing episode {i}/{max_episodes}")
        try:
            steps = episode["steps"]
            step_features = []
            for step in steps:
                obs = {k: v.numpy() for k, v in step["observation"].items()
                       if hasattr(v, "numpy")}
                f = extract_episode_features(obs)
                step_features.append(f)

            episode_feat = np.stack(step_features)
            summary = np.concatenate([
                episode_feat.mean(0),
                episode_feat.std(0),
                episode_feat.max(0) - episode_feat.min(0),
            ])
            features_list.append(summary)

            # 1 = failure, 0 = success  (field name varies by dataset)
            success = episode.get("success", episode.get("is_success", None))
            labels.append(0 if (success is not None and bool(success.numpy())) else 1)

        except Exception as e:
            print(f"  Skipping episode {i}: {e}")
            continue

    return np.array(features_list), np.array(labels)


def load_lerobot(dataset_name: str = "lerobot/pusht", max_episodes: int = 500):
    """
    Load LeRobot dataset by downloading parquet files directly from HuggingFace.
    Avoids lerobot package (requires system deps) and datasets schema issues.
    """
    try:
        import pandas as pd
        from huggingface_hub import HfFileSystem
    except ImportError:
        raise ImportError("Run: pip install pandas huggingface_hub pyarrow")

    print(f"Loading {dataset_name} from HuggingFace (parquet)...")
    fs = HfFileSystem()
    repo = dataset_name.replace("lerobot/", "")

    # list parquet shards
    parquet_files = fs.glob(f"datasets/lerobot/{repo}/data/**/*.parquet")
    if not parquet_files:
        parquet_files = fs.glob(f"datasets/lerobot/{repo}/**/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {dataset_name}")

    print(f"  Found {len(parquet_files)} parquet shard(s). Downloading...")
    dfs = []
    for p in parquet_files:
        with fs.open(p, "rb") as f:
            dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total steps: {len(df):,}  |  Columns: {list(df.columns)}")

    # detect state/action columns
    state_cols = [c for c in df.columns if "observation.state" in c or c == "state"]
    action_cols = [c for c in df.columns if c == "action"]

    # LeRobot stores arrays as nested lists — explode them
    def expand_col(series):
        first = series.iloc[0]
        if hasattr(first, "__len__") and not isinstance(first, str):
            return pd.DataFrame(series.tolist(), index=series.index)
        return series.to_frame()

    parts = []
    for col in state_cols + action_cols:
        parts.append(expand_col(df[col]))
    if not parts:
        raise ValueError(f"No state/action columns found. Available: {list(df.columns)}")

    numeric = pd.concat(parts, axis=1).astype(np.float32)
    df["_feat"] = list(numeric.values)

    ep_col = "episode_index" if "episode_index" in df.columns else df.columns[0]
    episode_ids = sorted(df[ep_col].unique())[:max_episodes]

    reward_col = next((c for c in ["next.reward", "reward"] if c in df.columns), None)
    success_col = next((c for c in ["next.success", "success"] if c in df.columns), None)

    # pre-compute per-episode max reward and derive label thresholds
    ep_max_rewards = None
    if reward_col:
        ep_max_rewards = df.groupby(ep_col)[reward_col].max()
        subset = ep_max_rewards[ep_max_rewards.index.isin(episode_ids)]
        has_binary_reward = float(subset.min()) >= -0.01  # rewards are 0/1 style
        nominal_thresh = float(np.percentile(subset, 70))   # top 30% = nominal
        failure_thresh = float(np.percentile(subset, 20))   # bottom 20% = failure
        print(f"  Reward range [{subset.min():.3f}, {subset.max():.3f}] | "
              f"nominal>{nominal_thresh:.3f}, failure<{failure_thresh:.3f}")

    features_list, labels = [], []
    for i, ep_id in enumerate(episode_ids):
        if i % 50 == 0:
            print(f"  Processing episode {i}/{len(episode_ids)}")
        ep = df[df[ep_col] == ep_id]
        try:
            episode_feat = np.stack(ep["_feat"].values)
            summary = np.concatenate([
                episode_feat.mean(0),
                episode_feat.std(0),
                episode_feat.max(0) - episode_feat.min(0),
            ])

            if success_col and ep[success_col].any():
                label = 0  # success
            elif ep_max_rewards is not None:
                max_r = float(ep_max_rewards.get(ep_id, 0))
                if has_binary_reward:
                    label = 0 if max_r > 0.5 else 1
                else:
                    # continuous/negative reward — only keep clear nominal & failure
                    if max_r >= nominal_thresh:
                        label = 0
                    elif max_r <= failure_thresh:
                        label = 1
                    else:
                        continue  # skip ambiguous middle band
            else:
                label = 0

            features_list.append(summary)
            labels.append(label)

        except Exception as e:
            print(f"  Skipping episode {ep_id}: {e}")
            continue

    return np.array(features_list), np.array(labels)


def load_synthetic_demo(n_nominal: int = 400, n_failure: int = 100, n_features: int = 48):
    """
    Synthetic stand-in — lets you run the full pipeline instantly without
    downloading any dataset. Replace with a real loader once data is ready.
    """
    print("Using synthetic demo data (replace with real dataset loader).")
    rng = np.random.RandomState(RANDOM_SEED)

    nominal = rng.randn(n_nominal, n_features).astype(np.float32)
    # failures have shifted mean + higher variance
    failures = (rng.randn(n_failure, n_features) * 2 + 1.5).astype(np.float32)

    features = np.vstack([nominal, failures])
    labels = np.array([0] * n_nominal + [1] * n_failure)
    return features, labels


# ── Anomaly detection pipeline ───────────────────────────────────────────────

def run_benchmark(features: np.ndarray, labels: np.ndarray, dataset_label: str):
    print(f"\n{'='*60}")
    print(f"Benchmark: {dataset_label}")
    print(f"  Total episodes : {len(features)}")
    print(f"  Failures       : {labels.sum()} ({labels.mean()*100:.1f}%)")

    # train only on nominal episodes
    nominal_mask = labels == 0
    nominal_features = features[nominal_mask]

    scaler = StandardScaler()
    nominal_scaled = scaler.fit_transform(nominal_features)
    all_scaled = scaler.transform(features)

    clf = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(nominal_scaled)

    # higher score = more anomalous
    scores = -clf.score_samples(all_scaled)

    # metrics
    auc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # find threshold closest to CONFIDENCE_THRESHOLD quantile of scores
    thresh = np.quantile(scores, CONFIDENCE_THRESHOLD)
    predictions = (scores >= thresh).astype(int)

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    card = {
        "dataset": dataset_label,
        "model": "IsolationForest v0.1",
        "total_episodes": int(len(features)),
        "failure_episodes": int(labels.sum()),
        "roc_auc": round(float(auc), 4),
        "detection_rate_pct": round(detection_rate * 100, 1),
        "false_positive_rate_pct": round(fpr * 100, 1),
        "confidence_threshold_quantile": CONFIDENCE_THRESHOLD,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }

    print(f"\n  BENCHMARK CARD")
    print(f"  ROC-AUC              : {card['roc_auc']}")
    print(f"  Detection rate       : {card['detection_rate_pct']}%")
    print(f"  False positive rate  : {card['false_positive_rate_pct']}%")
    print(f"  Threshold (quantile) : {CONFIDENCE_THRESHOLD}")

    # save annotated scores as HDF5
    safe_name = dataset_label.replace("/", "_").replace(" ", "_")
    hdf5_path = OUTPUT_DIR / f"{safe_name}_scores.h5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("anomaly_scores", data=scores)
        f.create_dataset("true_labels", data=labels)
        f.create_dataset("predictions", data=predictions)
        f.create_dataset("features", data=features)
        f.attrs["dataset"] = dataset_label
        f.attrs["roc_auc"] = card["roc_auc"]
        f.attrs["detection_rate_pct"] = card["detection_rate_pct"]

    card_path = OUTPUT_DIR / f"{safe_name}_card.json"
    card_path.write_text(json.dumps(card, indent=2))

    print(f"\n  Saved: {hdf5_path}")
    print(f"  Saved: {card_path}")
    return card


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from models import IsolationForestModel, LSTMAEModel, compare_models, cross_dataset_validate

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["synthetic", "lerobot", "openx"],
        default="synthetic",
        help="Data source to benchmark against",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name override")
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument(
        "--model",
        choices=["isolation_forest", "lstm_ae", "compare"],
        default="isolation_forest",
        help="Model to use. 'compare' runs both side-by-side.",
    )
    parser.add_argument(
        "--cross-validate",
        nargs=2,
        metavar=("TRAIN_DATASET", "TEST_DATASET"),
        help="Cross-dataset validation: train on A, test on B. "
             "E.g. --cross-validate lerobot/xarm_lift_medium_replay lerobot/xarm_push_medium_replay",
    )
    args = parser.parse_args()

    # ── cross-dataset mode ───────────────────────────────────────────────────
    if args.cross_validate:
        train_name, test_name = args.cross_validate
        print(f"\nCross-dataset validation: {train_name} → {test_name}")
        train_feats, train_labels = load_lerobot(train_name, args.max_episodes)
        test_feats,  test_labels  = load_lerobot(test_name,  args.max_episodes)

        for ModelClass in [IsolationForestModel, LSTMAEModel]:
            card = cross_dataset_validate(
                train_feats, train_labels,
                test_feats,  test_labels,
                ModelClass,
                dataset_train=train_name,
                dataset_test=test_name,
            )
            safe = f"{train_name}_{test_name}_{card['model']}".replace("/", "_").replace(" ", "_")
            out  = OUTPUT_DIR / f"{safe}_cross_val.json"
            out.write_text(json.dumps(card, indent=2))
            print(f"  Saved: {out}")

    # ── normal benchmark mode ────────────────────────────────────────────────
    else:
        if args.source == "synthetic":
            features, labels = load_synthetic_demo()
            label = "Synthetic Demo"
        elif args.source == "lerobot":
            name = args.dataset or "lerobot/pusht"
            features, labels = load_lerobot(name, args.max_episodes)
            label = name
        elif args.source == "openx":
            name = args.dataset or "fractal20220817_data"
            features, labels = load_open_x_embodiment(name, args.max_episodes)
            label = name

        if args.model == "compare":
            compare_models(features, labels, label, OUTPUT_DIR)
        else:
            # also run the legacy run_benchmark for HDF5 output
            run_benchmark(features, labels, label)
