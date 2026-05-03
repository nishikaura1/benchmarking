"""
benchmark/failure_injector.py — Synthetic failure injection pipeline.

Takes clean robot trajectories from LeRobot datasets and injects
synthetic failures to create a labeled benchmark dataset.

Failure classes:
  grasp_slip          — grip force drops, object slips
  velocity_spike      — sudden overcorrection / joint jerk
  trajectory_deviation — gradual drift from intended path
  stuck_joint         — motor stall or collision
  overcorrect         — post-failure panic overcorrection
  nominal             — clean episode, no injection

Usage
-----
  python benchmark/failure_injector.py
  python benchmark/failure_injector.py --n-per-class 200 --dataset lerobot/pusht
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_base_episodes(dataset_name: str = "lerobot/pusht",
                       n_episodes: int = 200) -> list:
    """
    Load clean robot episodes from a LeRobot HuggingFace dataset.
    Falls back to our locally cached episodes if the dataset isn't available.

    Returns list of dicts: {'action': np.ndarray, 'state': np.ndarray}
    """
    # ── Try loading from HuggingFace ──────────────────────────────────────────
    try:
        from datasets import load_dataset
        import pandas as pd

        print(f"  Downloading {dataset_name} from HuggingFace...")
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()

        # Extract episode_index column (LeRobot standard)
        ep_col = next((c for c in df.columns if "episode" in c.lower()), None)
        if ep_col is None:
            raise ValueError("No episode_index column found")

        # Detect action + state columns
        act_cols   = [c for c in df.columns if "action" in c.lower()]
        state_cols = [c for c in df.columns if "observation.state" in c.lower()
                      or "obs" in c.lower()]

        if not act_cols:
            raise ValueError("No action columns found")

        episodes = []
        for ep_id in sorted(df[ep_col].unique())[:n_episodes]:
            ep_df = df[df[ep_col] == ep_id].reset_index(drop=True)

            # Stack action columns into (T, D_action) array
            def _stack(cols):
                first = ep_df[cols[0]].iloc[0]
                if hasattr(first, '__len__'):
                    return np.vstack(ep_df[cols[0]].values).astype(np.float32)
                return ep_df[cols].values.astype(np.float32)

            action = _stack(act_cols)
            state  = _stack(state_cols) if state_cols else action.copy()

            if len(action) < 20:
                continue
            episodes.append({"action": action, "state": state, "source": dataset_name})

        print(f"  → {len(episodes)} episodes loaded from {dataset_name}")
        return episodes[:n_episodes]

    except Exception as hf_err:
        print(f"  HuggingFace load failed ({hf_err}) — falling back to local cache")

    # ── Fallback: use locally cached LeRobot pkl files ────────────────────────
    import pickle
    cache_dir = ROOT / "benchmark_output"
    pkl_files = sorted(cache_dir.glob("lerobot_*_episodes.pkl"))

    if not pkl_files:
        raise RuntimeError(
            "No local episode cache found. Run annotation_model.py --train first, "
            "or ensure HuggingFace credentials are set."
        )

    episodes = []
    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            cached = pickle.load(f)
        for seq, ep_label, ds in cached:
            episodes.append({
                "action": seq.astype(np.float32),
                "state":  seq.astype(np.float32),
                "source": ds,
            })
        if len(episodes) >= n_episodes:
            break

    print(f"  → {len(episodes)} episodes loaded from local cache")
    return episodes[:n_episodes]


# ── Failure injectors ─────────────────────────────────────────────────────────

def _copy_ep(episode: dict) -> dict:
    """Deep copy an episode dict — action and state are kept in sync."""
    ep = {k: v.copy() for k, v in episode.items() if isinstance(v, np.ndarray)}
    return ep


def _sync(ep: dict):
    """After modifying action, sync state to match (they represent the same trajectory)."""
    ep["state"] = ep["action"].copy()


def inject_grasp_slip(episode: dict,
                      slip_start: int = None,
                      severity: float = 0.35) -> tuple:
    """Drop grip force suddenly — simulates object slipping."""
    ep  = _copy_ep(episode)
    T   = len(ep["action"])
    if slip_start is None:
        slip_start = int(T * 0.4)
    duration = min(20, T - slip_start - 1)
    for i in range(slip_start, slip_start + duration):
        decay = (i - slip_start) / max(duration, 1)
        ep["action"][i] *= (1 - severity * decay)
    _sync(ep)
    return ep, "grasp_slip", slip_start


def inject_velocity_spike(episode: dict,
                          spike_at: int = None,
                          magnitude: float = 3.2) -> tuple:
    """Sudden velocity spike — overcorrection or joint jerk."""
    ep = _copy_ep(episode)
    T  = len(ep["action"])
    if spike_at is None:
        spike_at = int(T * 0.5)
    spike_at = min(spike_at, T - 2)
    ep["action"][spike_at]     *= magnitude
    ep["action"][spike_at + 1] *= -magnitude * 0.4   # rebound
    _sync(ep)
    return ep, "velocity_spike", spike_at


def inject_trajectory_deviation(episode: dict,
                                deviation_start: int = None,
                                drift: float = 0.15) -> tuple:
    """Gradual drift from intended path."""
    ep  = _copy_ep(episode)
    T   = len(ep["action"])
    rng = np.random.RandomState(42)
    if deviation_start is None:
        deviation_start = int(T * 0.3)
    for i in range(deviation_start, T):
        progress         = (i - deviation_start) / max(T - deviation_start, 1)
        ep["action"][i] += drift * progress * rng.normal(1, 0.1, ep["action"].shape[1])
    _sync(ep)
    return ep, "trajectory_deviation", deviation_start


def inject_stuck_joint(episode: dict,
                       stuck_at: int = None,
                       duration: int = 30) -> tuple:
    """Joint stops moving — motor stall or collision."""
    ep  = _copy_ep(episode)
    T   = len(ep["action"])
    rng = np.random.RandomState(7)
    if stuck_at is None:
        stuck_at = int(T * 0.45)
    stuck_at  = min(stuck_at, T - duration - 1)
    stuck_val = ep["action"][stuck_at].copy()
    for i in range(stuck_at, min(stuck_at + duration, T)):
        ep["action"][i] = stuck_val + rng.normal(0, 0.001, stuck_val.shape)
    _sync(ep)
    return ep, "stuck_joint", stuck_at


def inject_overcorrect(episode: dict,
                       drop_at: int = None) -> tuple:
    """Post-failure overcorrection — operator panic response."""
    ep = _copy_ep(episode)
    T  = len(ep["action"])
    if drop_at is None:
        drop_at = int(T * 0.5)
    drop_at = min(drop_at, T - 26)
    for i in range(drop_at, min(drop_at + 10, T)):
        ep["action"][i] *= 0.1
    for i in range(drop_at + 10, min(drop_at + 25, T)):
        ep["action"][i] *= 2.8
    _sync(ep)
    return ep, "overcorrect", drop_at


INJECTORS = {
    "grasp_slip":           inject_grasp_slip,
    "velocity_spike":       inject_velocity_spike,
    "trajectory_deviation": inject_trajectory_deviation,
    "stuck_joint":          inject_stuck_joint,
    "overcorrect":          inject_overcorrect,
    "nominal":              None,   # clean episode, no injection
}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_episode_features(episode_dict: dict) -> np.ndarray:
    """
    Extract the same 68-dim feature vector used by the Haptal RF model,
    then reduce to episode-level summary stats for episode-level classification.

    Returns a 1D array of features for this episode.
    """
    try:
        # Use Haptal's own feature extractor when available
        from annotation_model import extract_window_features, canonicalize_dof
        seq   = episode_dict["state"]
        seq   = canonicalize_dof(seq)
        feats = extract_window_features(seq)          # (T, 68)
        # Summarise to episode level: mean + std + max across time
        return np.concatenate([
            feats.mean(axis=0),
            feats.std(axis=0),
            feats.max(axis=0),
        ])
    except Exception:
        pass

    # Fallback: hand-crafted features when annotation_model isn't importable
    actions = episode_dict["action"]
    states  = episode_dict["state"]
    feats   = []

    # Velocity features
    if len(states) > 1:
        vel = np.diff(states, axis=0)
        feats.extend([vel.mean(), vel.std(), vel.max(), vel.min(),
                      np.abs(vel).mean(), np.abs(vel).max()])
    else:
        feats.extend([0.0] * 6)

    # Action features
    feats.extend([actions.mean(), actions.std(),
                  actions.max(), actions.min(),
                  np.abs(actions).mean(), np.abs(actions).max()])

    # Jerk features
    if len(states) > 2:
        vel  = np.diff(states, axis=0)
        jerk = np.diff(vel, axis=0)
        feats.extend([float(np.abs(jerk).mean()), float(np.abs(jerk).max())])
    else:
        feats.extend([0.0, 0.0])

    return np.array(feats, dtype=np.float32)


# ── Benchmark generation ──────────────────────────────────────────────────────

def generate_benchmark(n_per_class: int = 500,
                       dataset_name: str = "lerobot/pusht",
                       output_dir: str = None,
                       seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced benchmark dataset with n_per_class episodes per failure class.

    Saves:
      benchmark/data/train.parquet
      benchmark/data/test.parquet
      benchmark/data/metadata.json
    """
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    print("Loading base episodes from LeRobot...")
    base_eps = load_base_episodes(dataset_name, n_episodes=max(200, n_per_class))
    if len(base_eps) < 10:
        raise RuntimeError("Not enough base episodes loaded. Check dataset access.")

    all_records = []

    for failure_class, injector in INJECTORS.items():
        print(f"  Generating {n_per_class} × {failure_class}...")
        generated = 0
        attempts  = 0
        while generated < n_per_class and attempts < n_per_class * 3:
            attempts += 1
            base_ep = base_eps[rng.randint(len(base_eps))]

            try:
                if injector is None:
                    ep_dict          = _copy_ep(base_ep)
                    label            = "nominal"
                    failure_timestep = None
                else:
                    ep_dict, label, failure_timestep = injector(base_ep)

                # Extract episode-level features for classification
                feat_vec = extract_episode_features(ep_dict)

                all_records.append({
                    "episode_id":        f"{failure_class}_{generated:04d}",
                    "failure_class":     label,
                    "failure_timestep":  failure_timestep,
                    "n_steps":           len(ep_dict["action"]),
                    "features":          feat_vec.tolist(),
                    # Store raw state sequence so evaluate.py can run step-level model
                    "state_seq":         ep_dict["state"].tolist(),
                    "synthetic":         True,
                    "base_dataset":      base_ep.get("source", dataset_name),
                })
                generated += 1

            except Exception as e:
                continue   # skip malformed episodes

        print(f"    → {generated} episodes generated")

    df = pd.DataFrame(all_records)

    # 80/20 train/test split — stratified by class
    train_parts, test_parts = [], []
    for cls in df["failure_class"].unique():
        cls_df     = df[df["failure_class"] == cls].sample(frac=1, random_state=seed)
        split_at   = int(len(cls_df) * 0.8)
        train_parts.append(cls_df.iloc[:split_at])
        test_parts.append(cls_df.iloc[split_at:])

    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test  = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    train.to_parquet(f"{output_dir}/train.parquet", index=False)
    test.to_parquet(f"{output_dir}/test.parquet",  index=False)

    metadata = {
        "name":           "Haptal Robotics Failure Benchmark v1.0",
        "description":    (
            "Synthetic failure detection benchmark for robot training data "
            "quality assessment. Built on real LeRobot manipulation trajectories "
            "with physics-based failure injection."
        ),
        "classes":        list(INJECTORS.keys()),
        "n_per_class":    n_per_class,
        "total_episodes": len(df),
        "train_episodes": len(train),
        "test_episodes":  len(test),
        "base_datasets":  [dataset_name],
        "feature_dim":    len(all_records[0]["features"]) if all_records else 0,
        "version":        "1.0.0",
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Benchmark generated")
    print(f"  Total : {len(df):,} episodes")
    print(f"  Train : {len(train):,}  |  Test : {len(test):,}")
    print(f"  Classes: {list(INJECTORS.keys())}")
    print(f"  Saved to: {output_dir}/")
    print(f"{'='*50}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haptal failure injection benchmark")
    parser.add_argument("--n-per-class", type=int,  default=500,
                        help="Episodes per failure class (default: 500)")
    parser.add_argument("--dataset",     type=str,  default="lerobot/pusht",
                        help="Base LeRobot dataset (default: lerobot/pusht)")
    parser.add_argument("--output-dir",  type=str,  default=None,
                        help="Output directory (default: benchmark/data/)")
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    generate_benchmark(
        n_per_class=args.n_per_class,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
    )
