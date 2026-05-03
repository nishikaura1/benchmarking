"""
Step-level 3D annotation engine.

Goes beyond episode-level pass/fail — annotates every timestep with:
  1. Anomaly score (how abnormal is this moment)
  2. Failure type label (velocity spike, position jerk, stuck joint, gripper event)
  3. 3D trajectory coordinates for visualisation

Output: per-episode annotation dict ready for the dashboard.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json, h5py

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Failure type taxonomy ─────────────────────────────────────────────────────
# These are derived from real robot failure patterns seen in Open X-Embodiment.

FAILURE_TYPES = {
    "velocity_spike":        "Sudden joint velocity spike — collision or slip",
    "position_jerk":         "Acceleration discontinuity — abrupt direction change",
    "stuck_joint":           "Joint not moving — stall or grasp failure",
    "gripper_event":         "Unexpected gripper state change",
    "workspace_boundary":    "End-effector near workspace limit",
    "nominal":               "Normal operation",
}


def compute_step_features(state_seq: np.ndarray) -> np.ndarray:
    """
    state_seq: (T, D) — raw joint states over time
    Returns   (T, D*3) — state + velocity + acceleration per timestep
    """
    T, D = state_seq.shape
    vel  = np.diff(state_seq, axis=0, prepend=state_seq[:1])   # (T, D)
    acc  = np.diff(vel,        axis=0, prepend=vel[:1])         # (T, D)
    return np.concatenate([state_seq, vel, acc], axis=1)        # (T, D*3)


def label_failure_types(state_seq: np.ndarray,
                         vel_spike_thresh: float = 3.0,
                         jerk_thresh: float = 3.0,
                         stuck_window: int = 10,
                         stuck_var_thresh: float = 1e-4) -> list[str]:
    """
    Rule-based failure type labeler — one label per timestep.
    Thresholds are in units of standard deviations from the population mean.
    """
    T, D = state_seq.shape
    vel  = np.diff(state_seq, axis=0, prepend=state_seq[:1])
    acc  = np.diff(vel,        axis=0, prepend=vel[:1])

    # normalise so thresholds are scale-invariant
    vel_norm = np.abs(vel) / (vel.std(0) + 1e-8)
    acc_norm = np.abs(acc) / (acc.std(0) + 1e-8)

    labels = []
    for t in range(T):
        if vel_norm[t].max() > vel_spike_thresh:
            labels.append("velocity_spike")
        elif acc_norm[t].max() > jerk_thresh:
            labels.append("position_jerk")
        elif (t >= stuck_window and
              state_seq[t - stuck_window:t].var(axis=0).max() < stuck_var_thresh):
            labels.append("stuck_joint")
        elif D > 6 and abs(state_seq[t, -1] - state_seq[max(0, t-1), -1]) > 0.3:
            labels.append("gripper_event")
        else:
            labels.append("nominal")

    return labels


def trajectory_to_3d(state_seq: np.ndarray) -> np.ndarray:
    """
    Project joint state sequence to 3D for visualisation.
    Uses PCA if D > 3, otherwise takes first 3 dims directly.
    Returns (T, 3) array of XYZ coordinates.
    """
    T, D = state_seq.shape
    if D >= 3:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(state_seq)
    else:
        coords = np.pad(state_seq, ((0, 0), (0, 3 - D)))
    return coords.astype(np.float32)


def annotate_episode(state_seq: np.ndarray,
                      step_model: IsolationForest,
                      step_scaler: StandardScaler) -> dict:
    """
    Fully annotate a single episode.
    Returns dict with anomaly scores, failure types, and 3D coords per timestep.
    """
    feats  = compute_step_features(state_seq)
    scaled = step_scaler.transform(feats)
    scores = -step_model.score_samples(scaled)

    failure_types = label_failure_types(state_seq)
    coords_3d     = trajectory_to_3d(state_seq)

    return {
        "n_steps":       len(state_seq),
        "anomaly_scores": scores.tolist(),
        "failure_types":  failure_types,
        "coords_3d":      coords_3d.tolist(),
        "peak_score":     float(scores.max()),
        "peak_step":      int(scores.argmax()),
        "failure_counts": {k: failure_types.count(k) for k in FAILURE_TYPES},
        "dominant_failure": max(
            [k for k in FAILURE_TYPES if k != "nominal"],
            key=lambda k: failure_types.count(k)
        ) if any(f != "nominal" for f in failure_types) else "nominal",
    }


# ── Dataset loader (step-level) ───────────────────────────────────────────────

def load_step_data(dataset_name: str = "lerobot/xarm_lift_medium_replay",
                    max_episodes: int = 100) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Returns:
        state_seqs : list of (T_i, D) arrays — one per episode
        ep_labels  : (N,) int array — 0 nominal, 1 failure
        episode_ids: (N,) array of episode indices
    """
    import pandas as pd
    from huggingface_hub import HfFileSystem

    fs   = HfFileSystem()
    repo = dataset_name.replace("lerobot/", "")
    files = fs.glob(f"datasets/lerobot/{repo}/data/**/*.parquet")

    dfs = []
    for p in files:
        with fs.open(p, "rb") as f:
            dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)

    ep_col     = "episode_index"
    reward_col = next((c for c in ["next.reward", "reward"] if c in df.columns), None)
    state_cols = [c for c in df.columns if "observation.state" in c or c == "state"]

    if not state_cols:
        raise ValueError(f"No state columns in {dataset_name}. Cols: {list(df.columns)}")

    # expand state arrays
    def expand(series):
        first = series.iloc[0]
        if hasattr(first, "__len__"):
            return pd.DataFrame(series.tolist(), index=series.index)
        return series.to_frame()

    state_df = expand(df[state_cols[0]]).astype(np.float32)
    df["_state"] = list(state_df.values)

    ep_ids = sorted(df[ep_col].unique())[:max_episodes]

    # label episodes
    if reward_col:
        ep_max_r   = df.groupby(ep_col)[reward_col].max()
        has_binary = float(ep_max_r.min()) >= -0.01
        nom_thresh = float(np.percentile(ep_max_r, 70))
        fail_thresh= float(np.percentile(ep_max_r, 20))

    state_seqs, ep_labels, ep_id_out = [], [], []
    for ep_id in ep_ids:
        ep = df[df[ep_col] == ep_id]
        seq = np.stack(ep["_state"].values)
        state_seqs.append(seq)

        if reward_col:
            max_r = float(ep_max_r.get(ep_id, 0))
            if has_binary:
                label = 0 if max_r > 0.5 else 1
            else:
                if max_r >= nom_thresh:
                    label = 0
                elif max_r <= fail_thresh:
                    label = 1
                else:
                    state_seqs.pop()
                    continue
        else:
            label = 0

        ep_labels.append(label)
        ep_id_out.append(ep_id)

    return state_seqs, np.array(ep_labels), np.array(ep_id_out)


# ── Full annotation pipeline ──────────────────────────────────────────────────

def run_annotation_pipeline(dataset_name: str = "lerobot/xarm_lift_medium_replay",
                              max_episodes: int = 100) -> dict:
    print(f"Loading step-level data from {dataset_name}...")
    state_seqs, ep_labels, ep_ids = load_step_data(dataset_name, max_episodes)
    print(f"  Loaded {len(state_seqs)} episodes")

    # fit step-level anomaly model on nominal episodes
    nominal_seqs = [state_seqs[i] for i in range(len(ep_labels)) if ep_labels[i] == 0]
    all_nominal_steps = np.vstack([compute_step_features(s) for s in nominal_seqs])

    print(f"  Fitting step model on {len(all_nominal_steps):,} nominal steps...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(all_nominal_steps)
    model  = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(scaled)

    # annotate every episode
    print("  Annotating episodes...")
    annotations = []
    for i, (seq, label, ep_id) in enumerate(zip(state_seqs, ep_labels, ep_ids)):
        ann = annotate_episode(seq, model, scaler)
        ann["episode_id"]    = int(ep_id)
        ann["episode_index"] = i
        ann["true_label"]    = int(label)
        ann["label_str"]     = "FAILURE" if label else "OK"
        annotations.append(ann)

    result = {
        "dataset":     dataset_name,
        "n_episodes":  len(annotations),
        "n_failures":  int(ep_labels.sum()),
        "feature_dim": state_seqs[0].shape[1],
        "annotations": annotations,
    }

    safe  = dataset_name.replace("/", "_")
    out   = OUTPUT_DIR / f"{safe}_annotations.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lerobot/xarm_lift_medium_replay")
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()
    run_annotation_pipeline(args.dataset, args.max_episodes)
