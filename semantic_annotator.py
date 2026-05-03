"""
SemanticAnnotator — 3D robot data labeling beyond pass/fail.

This is a separate product from the failure detector.
It answers: "what is the robot doing at each moment?"

Four label layers, each a separate classifier:
  1. task_phase     — approaching / grasping / transporting / placing / returning / idle
  2. workspace_zone — near_object / mid_transit / near_target / boundary / home
  3. contact_state  — no_contact / pre_grasp / in_grasp / releasing
  4. motion_type    — stationary / slow_move / fast_move / deceleration / rotation

Trained on:
  - lerobot/droid_100         (diverse real-world tasks, 7-DOF)
  - lerobot/aloha_static_coffee   (bimanual manipulation, 14-DOF)
  - lerobot/aloha_mobile_cabinet  (mobile + manipulation, 14-DOF)
  - lerobot/aloha_sim_transfer_cube_scripted

Usage:
  python semantic_annotator.py --train
  python semantic_annotator.py --annotate --input lerobot/droid_100
"""

import argparse, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("benchmark_output")
MODEL_PATH = OUTPUT_DIR / "semantic_annotator.pkl"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Label taxonomy ────────────────────────────────────────────────────────────

LABEL_SCHEMA = {
    "task_phase": {
        "approaching":   "Moving toward the target object",
        "grasping":      "End-effector closing around the object",
        "transporting":  "Carrying object to destination",
        "placing":       "Lowering / releasing object at target",
        "returning":     "Moving back toward home/rest pose",
        "idle":          "Minimal movement — paused or at rest",
    },
    "workspace_zone": {
        "home":          "Near the rest/start configuration",
        "near_object":   "Close to the target object",
        "mid_transit":   "In transit between object and target",
        "near_target":   "Close to the placement/goal location",
        "boundary":      "Near the edge of the reachable workspace",
    },
    "contact_state": {
        "no_contact":    "End-effector not in contact with object",
        "pre_grasp":     "Approaching, about to make contact",
        "in_grasp":      "Object securely grasped",
        "releasing":     "Opening gripper to release object",
    },
    "motion_type": {
        "stationary":    "Joint velocities near zero",
        "slow_move":     "Low-speed coordinated movement",
        "fast_move":     "High-speed transit movement",
        "decelerating":  "Velocity decreasing — approaching target",
        "rotating":      "Primarily rotational joint motion",
    },
}

WINDOW = 10


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_semantic_features(state_seq: np.ndarray) -> np.ndarray:
    """
    state_seq: (T, D) — joint states over time
    Returns  : (T, 20) — FIXED-SIZE feature vector, DOF-independent.

    All per-joint values are summarised (mean/max/std) so this works
    for any robot regardless of DOF count (7-DOF DROID, 14-DOF ALOHA, etc.)
    """
    T, D = state_seq.shape
    eps  = 1e-8

    vel  = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
    acc  = np.vstack([np.zeros((1, D)), np.diff(vel, axis=0)])

    vel_mag = np.linalg.norm(vel, axis=1)       # (T,)
    acc_mag = np.linalg.norm(acc, axis=1)

    # normalise state to [0,1] per joint
    s_min  = state_seq.min(0)
    s_max  = state_seq.max(0)
    s_norm = (state_seq - s_min) / (s_max - s_min + eps)

    ep_mean    = state_seq.mean(0)
    dist_mean  = np.linalg.norm(state_seq - ep_mean,        axis=1)
    dist_start = np.linalg.norm(state_seq - state_seq[0],   axis=1)
    dist_end   = np.linalg.norm(state_seq - state_seq[-1],  axis=1)

    # gripper: last joint if D > 5
    gripper     = state_seq[:, -1] if D > 5 else np.zeros(T)
    gripper_norm= (gripper - gripper.min()) / (gripper.max() - gripper.min() + eps)
    gripper_vel = np.abs(np.diff(gripper, prepend=gripper[0]))

    # rotation proxy: distal joints moving more than proximal
    if D >= 4:
        base_vel   = np.linalg.norm(vel[:, :D//2],  axis=1)
        distal_vel = np.linalg.norm(vel[:, D//2:],  axis=1)
        rot_ratio  = distal_vel / (base_vel + eps)
    else:
        rot_ratio  = np.zeros(T)

    rows = []
    for t in range(T):
        w_start = max(0, t - WINDOW + 1)
        w_mag   = vel_mag[w_start:t+1]
        half    = max(1, len(w_mag) // 2)
        vel_trend = float(w_mag[-half:].mean() - w_mag[:half].mean())

        w_vel = vel[w_start:t+1]
        corr  = float(np.corrcoef(w_vel.T).mean()) \
                if (w_vel.shape[0] > 2 and w_vel.shape[1] > 1) else 0.0

        row = np.array([
            float(s_norm[t].mean()),        # 1  mean normalised joint position
            float(s_norm[t].std()),         # 2  spread of joint positions
            float(s_norm[t].max()),         # 3  max normalised position
            float(s_norm[t].min()),         # 4  min normalised position
            float(vel_mag[t]),              # 5  velocity magnitude
            float(vel[t].mean()),           # 6  mean joint velocity
            float(vel[t].std()),            # 7  velocity spread across joints
            float(acc_mag[t]),              # 8  acceleration magnitude
            float(dist_mean[t]),            # 9  distance from episode centre
            float(dist_start[t]),           # 10 distance from start
            float(dist_end[t]),             # 11 distance from end
            float(dist_start[t] / (dist_start.max() + eps)),  # 12 normalised dist from start
            float(gripper_norm[t]),         # 13 normalised gripper aperture
            float(gripper_vel[t]),          # 14 gripper velocity (open/close rate)
            float(vel_trend),               # 15 velocity trend (+accel / -decel)
            float(corr),                    # 16 cross-joint correlation
            float(rot_ratio[t]),            # 17 distal vs proximal motion ratio
            float(t) / T,                   # 18 normalised episode progress (0-1)
            float(t) / T ** 0.5,            # 19 sqrt-scaled progress (emphasises early)
            float(vel_mag[t] / (vel_mag.max() + eps)),  # 20 normalised velocity
        ], dtype=np.float32)
        rows.append(row)

    return np.array(rows, dtype=np.float32)   # (T, 20)


# ── Weak supervision label generators ────────────────────────────────────────

def label_task_phase(state_seq: np.ndarray) -> list:
    """
    Segment the episode into task phases using velocity + progress heuristics.
    Works for any pick-and-place or manipulation task.
    """
    T, D = state_seq.shape
    vel      = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
    vel_mag  = np.linalg.norm(vel, axis=1)
    vel_norm = vel_mag / (vel_mag.max() + 1e-8)

    gripper  = state_seq[:, -1] if D > 5 else np.zeros(T)
    grip_vel = np.abs(np.diff(gripper, prepend=gripper[0]))

    # thresholds
    moving_thresh = 0.1
    fast_thresh   = 0.5
    grip_thresh   = np.percentile(grip_vel[grip_vel > 0], 50) if (grip_vel > 0).any() else 0.05

    # identify gripper close and open events
    grip_close_t = np.where((grip_vel > grip_thresh) & (np.diff(gripper, prepend=gripper[0]) < 0))[0]
    grip_open_t  = np.where((grip_vel > grip_thresh) & (np.diff(gripper, prepend=gripper[0]) > 0))[0]
    grasp_start  = int(grip_close_t[0])  if len(grip_close_t) else T // 3
    place_start  = int(grip_open_t[-1])  if len(grip_open_t)  else 2 * T // 3

    labels = []
    for t in range(T):
        prog = t / T
        if vel_norm[t] < 0.05:
            labels.append("idle")
        elif t < grasp_start and vel_norm[t] > moving_thresh:
            labels.append("approaching")
        elif t < grasp_start + T // 10 and grip_vel[t] > grip_thresh * 0.5:
            labels.append("grasping")
        elif grasp_start <= t < place_start:
            labels.append("transporting")
        elif t >= place_start and grip_vel[t] > grip_thresh * 0.5:
            labels.append("placing")
        elif t > place_start and vel_norm[t] > moving_thresh:
            labels.append("returning")
        else:
            labels.append("idle")

    return labels


def label_workspace_zone(state_seq: np.ndarray) -> list:
    """
    Classify where in the workspace the robot is at each step.
    Uses the episode's own trajectory geometry as reference.
    """
    T, D    = state_seq.shape
    start   = state_seq[0]
    end     = state_seq[-1]
    ep_min  = state_seq.min(0)
    ep_max  = state_seq.max(0)
    ep_range= ep_max - ep_min + 1e-8

    # boundary = within 10% of joint range extremes
    near_bound = ((state_seq - ep_min) / ep_range < 0.10) | \
                 ((ep_max - state_seq) / ep_range < 0.10)

    dist_start = np.linalg.norm(state_seq - start, axis=1)
    dist_end   = np.linalg.norm(state_seq - end,   axis=1)
    ds_norm    = dist_start / (dist_start.max() + 1e-8)
    de_norm    = dist_end   / (dist_end.max()   + 1e-8)

    # find the "object" region: max displacement from start in first half
    first_half   = state_seq[:T//2]
    object_step  = int(np.linalg.norm(first_half - start, axis=1).argmax())
    object_pos   = state_seq[object_step]
    dist_object  = np.linalg.norm(state_seq - object_pos, axis=1)
    do_norm      = dist_object / (dist_object.max() + 1e-8)

    labels = []
    for t in range(T):
        if near_bound[t].any():
            labels.append("boundary")
        elif ds_norm[t] < 0.15:
            labels.append("home")
        elif de_norm[t] < 0.15:
            labels.append("near_target")
        elif do_norm[t] < 0.20:
            labels.append("near_object")
        else:
            labels.append("mid_transit")

    return labels


def label_contact_state(state_seq: np.ndarray) -> list:
    """
    Infer contact/grasp state from gripper dimension.
    Falls back to trajectory-based heuristic if no gripper channel.
    """
    T, D    = state_seq.shape
    gripper = state_seq[:, -1] if D > 5 else None

    if gripper is None:
        # no gripper data — use velocity proxy
        vel     = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
        vel_mag = np.linalg.norm(vel, axis=1)
        thresh  = np.percentile(vel_mag, 30)
        labels  = []
        for t in range(T):
            prog = t / T
            if prog < 0.2:
                labels.append("no_contact")
            elif prog < 0.35:
                labels.append("pre_grasp")
            elif prog < 0.75:
                labels.append("in_grasp")
            else:
                labels.append("releasing")
        return labels

    # normalise gripper to [0,1]
    g_min, g_max = gripper.min(), gripper.max()
    g_norm = (gripper - g_min) / (g_max - g_min + 1e-8)
    g_vel  = np.abs(np.diff(g_norm, prepend=g_norm[0]))

    close_thresh = 0.3   # gripper < 30% open = in grasp
    open_thresh  = 0.7   # gripper > 70% open = no contact

    labels = []
    for t in range(T):
        if g_norm[t] > open_thresh:
            labels.append("no_contact")
        elif g_vel[t] > 0.05 and g_norm[t] < 0.5:
            labels.append("pre_grasp")
        elif g_norm[t] < close_thresh:
            labels.append("in_grasp")
        elif g_vel[t] > 0.05 and g_norm[t] > 0.5:
            labels.append("releasing")
        else:
            labels.append("no_contact")

    return labels


def label_motion_type(state_seq: np.ndarray) -> list:
    """Classify the type of motion at each timestep."""
    T, D    = state_seq.shape
    vel     = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
    acc     = np.vstack([np.zeros((1, D)), np.diff(vel, axis=0)])
    vel_mag = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    # rotation = wrist/end joints moving more than shoulder joints
    if D >= 4:
        base_vel  = np.linalg.norm(vel[:, :2], axis=1)
        distal_vel= np.linalg.norm(vel[:, -2:], axis=1)
        rot_ratio = distal_vel / (base_vel + 1e-8)
    else:
        rot_ratio = np.zeros(T)

    slow_thresh = np.percentile(vel_mag, 33)
    fast_thresh = np.percentile(vel_mag, 66)

    labels = []
    for t in range(T):
        if vel_mag[t] < slow_thresh * 0.3:
            labels.append("stationary")
        elif rot_ratio[t] > 2.0 and vel_mag[t] > slow_thresh:
            labels.append("rotating")
        elif vel_mag[t] > fast_thresh:
            labels.append("fast_move")
        elif acc_mag[t] < 0 or (t > 0 and vel_mag[t] < vel_mag[t-1] * 0.8):
            labels.append("decelerating")
        else:
            labels.append("slow_move")

    return labels


# ── Dataset loader ────────────────────────────────────────────────────────────

SEMANTIC_DATASETS = [
    "lerobot/droid_100",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_sim_transfer_cube_scripted",
]

def load_semantic_data(dataset_names: list, max_eps: int = 50) -> list:
    """
    Returns list of (state_seq, dataset_name) tuples.
    No failure labels needed — purely semantic.
    """
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    episodes = []
    for name in dataset_names:
        repo  = name.replace("lerobot/", "")
        try:
            files = fs.glob(f"datasets/lerobot/{repo}/data/**/*.parquet")
        except Exception as e:
            print(f"  {name}: skipping ({e})")
            continue
        if not files:
            print(f"  {name}: not found, skipping")
            continue

        print(f"  Loading {name}...")
        dfs = []
        for p in files:
            try:
                dfs.append(pd.read_parquet(fs.open(p, "rb")))
            except Exception as e:
                print(f"    shard failed: {e}")
                continue
        if not dfs:
            continue
        df  = pd.concat(dfs, ignore_index=True)

        state_cols = [c for c in df.columns if "observation.state" in c]
        if not state_cols:
            print(f"  {name}: no state columns, skipping")
            continue

        def expand(s):
            first = s.iloc[0]
            return pd.DataFrame(s.tolist(), index=s.index) \
                   if hasattr(first, "__len__") else s.to_frame()

        state_df   = expand(df[state_cols[0]]).astype(np.float32)
        df["_state"] = list(state_df.values)

        ep_col = "episode_index"
        ep_ids = sorted(df[ep_col].unique())[:max_eps]
        for ep_id in ep_ids:
            ep  = df[df[ep_col] == ep_id]
            seq = np.stack(ep["_state"].values)
            episodes.append((seq, name))

        print(f"    → {len(ep_ids)} episodes loaded")

    return episodes


# ── Training pipeline ─────────────────────────────────────────────────────────

class SemanticAnnotator:
    """
    Semantic labeling model for 3D robot trajectory data.
    Produces four label layers per timestep: task_phase, workspace_zone,
    contact_state, motion_type.
    """

    def __init__(self):
        self.models   = {}    # one RF per label layer
        self.scalers  = {}
        self.encoders = {}
        self.trained_on = []

    def train(self, dataset_names: list = SEMANTIC_DATASETS, max_eps: int = 50):
        print(f"\nLoading data from {len(dataset_names)} datasets...")
        episodes = load_semantic_data(dataset_names, max_eps)
        print(f"Total episodes: {len(episodes)}")

        label_fns = {
            "task_phase":     label_task_phase,
            "workspace_zone": label_workspace_zone,
            "contact_state":  label_contact_state,
            "motion_type":    label_motion_type,
        }

        for layer, fn in label_fns.items():
            print(f"\n── Training [{layer}] classifier ──")
            X_all, y_all = [], []
            for seq, ds_name in episodes:
                feats  = extract_semantic_features(seq)
                labels = fn(seq)
                X_all.extend(feats)
                y_all.extend(labels)

            X = np.array(X_all)
            y = np.array(y_all)

            print(f"  Steps: {len(X):,}  |  Features: {X.shape[1]}")
            dist = {k: (y==k).sum() for k in np.unique(y)}
            for k, v in sorted(dist.items(), key=lambda x: -x[1]):
                print(f"    {k:20s}: {v:5,}  ({v/len(y)*100:.1f}%)")

            scaler  = StandardScaler()
            le      = LabelEncoder()
            X_sc    = scaler.fit_transform(X)
            y_enc   = le.fit_transform(y)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_sc, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

            clf = RandomForestClassifier(
                n_estimators=120, max_depth=10,
                min_samples_leaf=5, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )
            clf.fit(X_tr, y_tr)

            y_pred  = clf.predict(X_val)
            report  = classification_report(y_val, y_pred,
                                             target_names=le.classes_,
                                             zero_division=0)
            acc     = (y_pred == y_val).mean()
            print(f"  Accuracy: {acc:.3f}")
            print(report)

            self.models[layer]   = clf
            self.scalers[layer]  = scaler
            self.encoders[layer] = le

        self.trained_on = dataset_names
        self.save()

    def annotate(self, state_seq: np.ndarray) -> dict:
        """
        Annotate one episode with all four label layers.
        Returns dict with per-step labels + confidences for each layer.
        """
        T     = len(state_seq)
        feats = extract_semantic_features(state_seq)

        result = {"n_steps": T, "layers": {}}

        for layer in ["task_phase", "workspace_zone", "contact_state", "motion_type"]:
            scaler  = self.scalers[layer]
            clf     = self.models[layer]
            le      = self.encoders[layer]

            X_sc     = scaler.transform(feats)
            pred_enc = clf.predict(X_sc)
            pred_prob= clf.predict_proba(X_sc)
            labels   = le.inverse_transform(pred_enc).tolist()
            confs    = pred_prob.max(axis=1).tolist()

            counts   = {k: labels.count(k) for k in le.classes_}
            dominant = max(counts, key=counts.get)

            result["layers"][layer] = {
                "labels":    labels,
                "confs":     [round(c, 3) for c in confs],
                "counts":    counts,
                "dominant":  dominant,
            }

        # 3D trajectory
        pca    = PCA(n_components=3)
        coords = pca.fit_transform(state_seq).astype(float).tolist()
        result["coords_3d"] = coords

        return result

    def annotate_dataset(self, dataset_name: str, max_eps: int = 100) -> dict:
        print(f"\nAnnotating {dataset_name}...")
        episodes = load_semantic_data([dataset_name], max_eps)
        annotations = []

        for i, (seq, _) in enumerate(episodes):
            ann = self.annotate(seq)
            ann["episode_index"] = i
            annotations.append(ann)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(episodes)} episodes done")

        result = {
            "dataset":           dataset_name,
            "model":             "SemanticAnnotator v1.0",
            "trained_on":        self.trained_on,
            "n_episodes":        len(annotations),
            "label_schema":      LABEL_SCHEMA,
            "annotations":       annotations,
        }
        safe = dataset_name.replace("/", "_")
        out  = OUTPUT_DIR / f"{safe}_semantic.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"Saved: {out}")
        return result

    def save(self, path: Path = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"models":     self.models,
                          "scalers":    self.scalers,
                          "encoders":   self.encoders,
                          "trained_on": self.trained_on}, f)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SemanticAnnotator":
        with open(path, "rb") as f:
            s = pickle.load(f)
        obj = cls()
        obj.models     = s["models"]
        obj.scalers    = s["scalers"]
        obj.encoders   = s["encoders"]
        obj.trained_on = s["trained_on"]
        print(f"SemanticAnnotator loaded — trained on: {obj.trained_on}")
        return obj


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--input",    type=str, default="lerobot/droid_100")
    parser.add_argument("--datasets", nargs="+", default=SEMANTIC_DATASETS)
    parser.add_argument("--max-episodes", type=int, default=50)
    args = parser.parse_args()

    if args.train:
        m = SemanticAnnotator()
        m.train(args.datasets, args.max_episodes)

    elif args.annotate:
        m = SemanticAnnotator.load()
        result = m.annotate_dataset(args.input, args.max_episodes)

        # print sample
        ep = result["annotations"][0]
        print(f"\nSample — Episode 0  ({ep['n_steps']} steps)")
        print(f"{'Step':>4}  {'Phase':>14}  {'Zone':>12}  {'Contact':>12}  {'Motion':>12}")
        print("-" * 62)
        for t in range(min(25, ep["n_steps"])):
            row = {layer: ep["layers"][layer]["labels"][t]
                   for layer in ["task_phase","workspace_zone","contact_state","motion_type"]}
            print(f"{t:>4}  {row['task_phase']:>14}  {row['workspace_zone']:>12}  "
                  f"{row['contact_state']:>12}  {row['motion_type']:>12}")
    else:
        parser.print_help()
