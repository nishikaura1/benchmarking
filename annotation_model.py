"""
RobotAnnotator — Step-level 3D trajectory annotation model.

This is the second product offering alongside benchmarking.

Architecture:
  - Sliding window feature extraction (position + velocity + acceleration)
  - RandomForestClassifier trained with weak supervision from rule-based labels
  - Multi-dataset training for generalisation across robot platforms
  - Outputs per-step failure type labels + confidence scores

Failure taxonomy (6 classes):
  0  nominal             — normal operation
  1  velocity_spike      — sudden joint velocity spike (collision, slip)
  2  position_jerk       — acceleration discontinuity (abrupt direction change)
  3  stuck_joint         — joint not moving when it should (stall, grasp fail)
  4  gripper_event       — unexpected gripper state change
  5  high_anomaly        — high anomaly score, unclassified (catch-all)

Usage:
  python annotation_model.py --train    # train + save model
  python annotation_model.py --annotate --input path/to/episode.h5
"""

import argparse, json, pickle, warnings, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss
from sklearn.decomposition import PCA
import h5py

warnings.filterwarnings("ignore")

OUTPUT_DIR        = Path("benchmark_output")
MODEL_PATH        = OUTPUT_DIR / "robot_annotator.pkl"
CORRECTIONS_PATH  = OUTPUT_DIR / "corrections.json"
HUMAN_LABEL_WEIGHT = 10.0   # human-verified labels count 10× vs weak supervision
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Failure taxonomy ──────────────────────────────────────────────────────────

FAILURE_CLASSES = [
    "nominal",
    "velocity_spike",
    "position_jerk",
    "stuck_joint",
    "gripper_event",
    "high_anomaly",
    "self_collision",
    "overshoot",
    "trajectory_deviation",
    "perception_failure",
]

FAILURE_DESCRIPTIONS = {
    "nominal":              "Normal operation",
    "velocity_spike":       "Sudden joint velocity spike — collision or slip",
    "position_jerk":        "Acceleration discontinuity — abrupt direction change",
    "stuck_joint":          "Joint not moving — stall or grasp failure",
    "gripper_event":        "Unexpected gripper state change",
    "high_anomaly":         "High anomaly score — unclassified failure",
    "self_collision":       "Adjacent joints moving in opposing directions — kinematic conflict",
    "overshoot":            "Velocity reversal after large motion — control overshoot / instability",
    "trajectory_deviation": "Position drifts from nominal path — accumulated tracking error",
    "perception_failure":   "Smooth motion but sudden unexpected displacement — pose estimation drift",
}

# Retraining strategies per failure class — tells clients what to fix
FAILURE_RETRAINING_STRATEGY = {
    "nominal":              "No action needed",
    "velocity_spike":       "Reduce max joint velocity limits; add collision avoidance layer",
    "position_jerk":        "Smooth trajectory with spline interpolation; tune PD gains",
    "stuck_joint":          "Add grasp detection feedback; increase torque limits",
    "gripper_event":        "Retrain grasp policy with more contact-rich demonstrations",
    "high_anomaly":         "Collect more nominal demonstrations for this task region",
    "self_collision":       "Add self-collision constraint to policy optimizer",
    "overshoot":            "Tune controller damping; reduce learning rate near target",
    "trajectory_deviation": "Add trajectory tracking reward term; increase waypoint density",
    "perception_failure":   "Improve state estimation; add IMU/force fusion for pose correction",
}

# ── Feature engineering ───────────────────────────────────────────────────────

WINDOW      = 10   # timesteps of history per sample
D_CANONICAL = 8    # all trajectories padded/truncated to this DOF before feature extraction


def canonicalize_dof(state_seq: np.ndarray, d: int = D_CANONICAL) -> np.ndarray:
    """
    Pad (with zeros) or truncate (drop last joints) a (T, D) trajectory so
    that D == d_canonical. This makes feature vectors from different robots
    (xarm D=4, ALOHA D=14, Franka D=7, etc.) all the same size.
    """
    T, D = state_seq.shape
    if D == d:
        return state_seq
    if D < d:
        pad = np.zeros((T, d - D), dtype=state_seq.dtype)
        return np.concatenate([state_seq, pad], axis=1)
    return state_seq[:, :d]   # truncate extra joints


def extract_window_features(state_seq: np.ndarray) -> np.ndarray:
    """
    state_seq : (T, D) joint state trajectory — any DOF, canonicalized to D_CANONICAL
    Returns   : (T, F) feature matrix — one row per timestep

    Features per timestep:
      - Raw state (D)
      - Velocity  (D)   — finite difference
      - Acceleration (D) — second derivative
      - Jerk (D)         — third derivative
      - Rolling mean over window (D)
      - Rolling std  over window (D)
      - Rolling max  over window (D)
      - L2 norm of velocity, acceleration
      - Cross-joint correlation proxy (std of velocity across joints)
    """
    state_seq = canonicalize_dof(state_seq)   # always D_CANONICAL columns
    T, D = state_seq.shape
    eps  = 1e-8

    vel  = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
    acc  = np.vstack([np.zeros((1, D)), np.diff(vel, axis=0)])
    jerk = np.vstack([np.zeros((1, D)), np.diff(acc, axis=0)])

    features = []
    for t in range(T):
        w_start = max(0, t - WINDOW + 1)
        window  = state_seq[w_start:t+1]          # (w, D)
        w_vel   = vel[w_start:t+1]

        row = np.concatenate([
            state_seq[t],                          # D  — current position
            vel[t],                                # D  — current velocity
            acc[t],                                # D  — current acceleration
            jerk[t],                               # D  — current jerk
            window.mean(0),                        # D  — rolling mean position
            window.std(0) + eps,                   # D  — rolling position variance
            window.max(0) - window.min(0),         # D  — rolling range
            w_vel.std(0) + eps,                    # D  — rolling velocity variance
            [np.linalg.norm(vel[t])],              # 1  — velocity magnitude
            [np.linalg.norm(acc[t])],              # 1  — acceleration magnitude
            [vel[t].std()],                        # 1  — cross-joint velocity spread
            [float(t) / T],                        # 1  — normalised timestep position
        ])
        features.append(row)

    return np.array(features, dtype=np.float32)


# ── Weak supervision labeler (rule-based) ─────────────────────────────────────

def generate_weak_labels(state_seq: np.ndarray,
                          anomaly_scores: np.ndarray = None) -> list:
    """
    Assign a failure-type label to each timestep using physics-based rules.
    Covers all 10 failure classes in FAILURE_CLASSES.

    Thresholds are adaptive (percentile-based per episode) so they fire on
    the most anomalous steps in every episode regardless of absolute scale.

    Priority order (first matching rule wins):
      velocity_spike → position_jerk → self_collision → overshoot →
      trajectory_deviation → stuck_joint → gripper_event →
      perception_failure → high_anomaly → nominal
    """
    state_seq = canonicalize_dof(state_seq)   # normalize DOF before rule thresholds
    T, D  = state_seq.shape
    eps   = 1e-8
    vel   = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
    acc   = np.vstack([np.zeros((1, D)), np.diff(vel, axis=0)])

    vel_mag = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    # ── adaptive thresholds ────────────────────────────────────────────────
    vel_thresh  = np.percentile(vel_mag, 85)
    acc_thresh  = np.percentile(acc_mag, 85)
    stuck_var   = np.percentile(state_seq.var(axis=0), 20)
    mean_vel    = vel_mag.mean() + eps
    # For stuck_joint detection: rolling window mean velocity (cached for speed)
    _windowed_vel = np.array([
        vel_mag[max(0, t - WINDOW):t].mean() if t > 0 else 0.0
        for t in range(T)
    ])

    # episode-level statistics used by trajectory_deviation and perception_failure
    ep_mean_pos   = state_seq.mean(axis=0)           # (D,) global mean position
    ep_range      = state_seq.max(axis=0) - state_seq.min(axis=0) + eps  # (D,)
    mean_acc      = acc_mag.mean() + eps

    # gripper = last dim if D > 5
    has_gripper  = D > 5
    if has_gripper:
        gripper      = state_seq[:, -1]
        gripper_diff = np.abs(np.diff(gripper, prepend=gripper[0]))
        grip_thresh  = np.percentile(gripper_diff[gripper_diff > 0], 75) \
                       if (gripper_diff > 0).any() else 1e9

    if anomaly_scores is not None:
        anom_thresh = np.percentile(anomaly_scores, 90)

    labels = []
    for t in range(T):
        vm, am = vel_mag[t], acc_mag[t]

        # ── 1. velocity_spike ─────────────────────────────────────────────
        if vm > vel_thresh and vm > mean_vel * 2:
            labels.append("velocity_spike")

        # ── 2. position_jerk ──────────────────────────────────────────────
        elif am > acc_thresh and am > mean_acc * 2:
            labels.append("position_jerk")

        # ── 3. self_collision — adjacent joints opposing with large velocities
        elif (D >= 2 and t > 0 and
              sum(1 for i in range(D - 1)
                  if (vel[t, i] * vel[t, i + 1] < 0
                      and abs(vel[t, i])     > vel_thresh * 0.6
                      and abs(vel[t, i + 1]) > vel_thresh * 0.6)
                  ) >= max(1, D // 3)):
            labels.append("self_collision")

        # ── 4. overshoot — velocity sign reversal after a fast previous step
        elif (t > 0 and
              np.any((np.sign(vel[t]) != np.sign(vel[t - 1])) &
                     (np.abs(vel[t - 1]) > vel_thresh * 1.5))):
            labels.append("overshoot")

        # ── 5. trajectory_deviation — drifted far from episode mean position
        elif np.any(np.abs(state_seq[t] - ep_mean_pos) / ep_range > 1.8):
            labels.append("trajectory_deviation")

        # ── 6. stuck_joint — no motion over the last WINDOW steps
        # Three conditions must ALL be true:
        #   (a) window position variance is low                  — window is flat
        #   (b) window velocity is << episode mean velocity      — this window much slower than normal
        #   (c) episode has meaningful overall motion            — not a globally slow/still robot
        # Without (b)+(c), slow robots (UR5 pipetting, fine manipulation) get
        # stuck_joint labels on every normal step, flooding training with false positives.
        elif (t >= WINDOW and
              state_seq[t - WINDOW:t].var(axis=0).max() < stuck_var * 0.5 and
              _windowed_vel[t] < mean_vel * 0.25 and
              mean_vel > 0.004):
            labels.append("stuck_joint")

        # ── 7. gripper_event ──────────────────────────────────────────────
        elif has_gripper and gripper_diff[t] > grip_thresh:
            labels.append("gripper_event")

        # ── 8. perception_failure — near-stillness but high acceleration
        #       proxy for pose-estimation jump: sudden state discontinuity
        #       while the robot "wasn't moving"
        elif (vm < 0.15 * mean_vel and am > acc_thresh * 1.5):
            labels.append("perception_failure")

        # ── 9. high_anomaly — catch-all ───────────────────────────────────
        elif anomaly_scores is not None and anomaly_scores[t] > anom_thresh:
            labels.append("high_anomaly")

        # ── 10. nominal ───────────────────────────────────────────────────
        else:
            labels.append("nominal")

    return labels


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_training_data(dataset_names: list, max_episodes_each: int = 80):
    """
    Load step-level state sequences + episode labels from multiple LeRobot datasets.
    Caches parsed episodes to benchmark_output/<dataset>_episodes.pkl so subsequent
    calls skip the HuggingFace download entirely (major speedup).
    Returns list of (state_seq, ep_label, dataset_name) tuples.
    """
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    all_episodes = []
    for dataset_name in dataset_names:
        # ── disk cache check ─────────────────────────────────────────────────
        safe_name  = dataset_name.replace("/", "_")
        cache_path = OUTPUT_DIR / f"{safe_name}_episodes.pkl"
        if cache_path.exists():
            print(f"  Loading {dataset_name} from cache...")
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            eps_this = cached[:max_episodes_each]
            all_episodes.extend(eps_this)
            print(f"    → {len(eps_this)} episodes (cached)")
            continue
        repo  = dataset_name.replace("lerobot/", "")
        files = fs.glob(f"datasets/lerobot/{repo}/data/**/*.parquet")
        if not files:
            print(f"  Skipping {dataset_name} — no parquet files found")
            continue

        print(f"  Loading {dataset_name}...")
        dfs = []
        for p in files:
            with fs.open(p, "rb") as f:
                dfs.append(pd.read_parquet(f))
        df = pd.concat(dfs, ignore_index=True)

        ep_col     = "episode_index"
        reward_col = next((c for c in ["next.reward", "reward"] if c in df.columns), None)
        state_cols = [c for c in df.columns if "observation.state" in c]
        if not state_cols:
            print(f"  Skipping {dataset_name} — no state columns")
            continue

        def expand(series):
            first = series.iloc[0]
            if hasattr(first, "__len__"):
                return pd.DataFrame(series.tolist(), index=series.index)
            return series.to_frame()

        state_df   = expand(df[state_cols[0]]).astype(np.float32)
        df["_state"] = list(state_df.values)

        ep_ids = sorted(df[ep_col].unique())[:max_episodes_each]

        if reward_col:
            ep_max_r   = df.groupby(ep_col)[reward_col].max()
            has_binary = float(ep_max_r.min()) >= -0.01
            nom_thresh = float(np.percentile(ep_max_r, 70))
            fail_thresh= float(np.percentile(ep_max_r, 20))

        for ep_id in ep_ids:
            ep  = df[df[ep_col] == ep_id]
            seq = np.stack(ep["_state"].values)

            if reward_col:
                max_r = float(ep_max_r.get(ep_id, 0))
                if has_binary:
                    ep_label = 0 if max_r > 0.5 else 1
                else:
                    if   max_r >= nom_thresh:  ep_label = 0
                    elif max_r <= fail_thresh: ep_label = 1
                    else:                      continue
            else:
                ep_label = 0

            all_episodes.append((seq, ep_label, dataset_name))

        eps_this = [e for e in all_episodes if e[2] == dataset_name]
        print(f"    → {len(eps_this)} episodes")
        # write cache so next run is instant
        with open(cache_path, "wb") as fh:
            pickle.dump(eps_this, fh)
        print(f"    → cached to {cache_path.name}")

    return all_episodes


# ── Training pipeline ─────────────────────────────────────────────────────────

class RobotAnnotator:
    """
    Trained step-level failure annotation model.

    Exposes:
        .train(dataset_names)               — train on open source datasets
        .annotate(state_seq)                — label every step in an episode
        .annotate_file(path)                — annotate an HDF5 or parquet file
        .save() / .load()
    """

    # confidence threshold below which a step is routed to human review
    REVIEW_THRESHOLD = 0.60

    def __init__(self):
        self.model            = None   # raw RandomForest (kept for feature importances)
        self.calibrated_model = None   # Platt-scaled wrapper — used for all inference
        self.scaler           = StandardScaler()
        self.le               = LabelEncoder().fit(FAILURE_CLASSES)
        self.feature_dim      = None
        self.train_report     = None
        self.train_accuracy   = None   # stored explicitly — never rely on report dict key
        self.calibration_report = None
        self.datasets_used    = []
        # training distribution — used for outlier detection at inference
        self._train_feat_mean = None   # (F,) per-feature mean of training set
        self._train_feat_std  = None   # (F,) per-feature std  of training set

    # ── Human corrections loader ──────────────────────────────────────────────

    def _load_human_corrections(self) -> tuple:
        """
        Load human-verified step labels from corrections.json and cross-reference
        feature vectors stored in the review queue JSON files.

        Returns (X_human, y_human) arrays — empty if no corrections exist.
        Feature vectors were stored in the review queue at pipeline annotation time,
        so we can re-inject them into the training pool at retrain time.
        """
        if not CORRECTIONS_PATH.exists():
            return np.empty((0,)), np.array([])

        with open(CORRECTIONS_PATH) as fh:
            corrections = json.load(fh)

        if not corrections:
            return np.empty((0,)), np.array([])

        # Build lookup: (episode_id, step_index) → feature_vector (unscaled)
        feat_lookup: dict = {}
        for rq_file in OUTPUT_DIR.glob("*_review_queue.json"):
            with open(rq_file) as fh:
                rq_data = json.load(fh)
            for ep in rq_data:
                ep_id   = ep["episode_id"]
                sf_dict = ep.get("step_features", {})   # str(step) → [floats]
                for step_str, feat_vec in sf_dict.items():
                    if feat_vec:
                        feat_lookup[(ep_id, int(step_str))] = feat_vec

        X_human, y_human = [], []
        for corr in corrections:
            key   = (corr["episode_id"], int(corr["step"]))
            label = corr.get("corrected_label") or corr.get("original_label", "nominal")
            if label not in FAILURE_CLASSES:
                continue
            if key in feat_lookup:
                X_human.append(feat_lookup[key])
                y_human.append(label)

        if not X_human:
            return np.empty((0,)), np.array([])

        X_arr = np.array(X_human, dtype=float)

        # Stored features may come from a different DOF era — pad/truncate columns
        # to match the canonical feature size: 8 * D_CANONICAL + 4
        expected_F = 8 * D_CANONICAL + 4
        if X_arr.shape[1] < expected_F:
            pad = np.zeros((len(X_arr), expected_F - X_arr.shape[1]), dtype=float)
            X_arr = np.concatenate([X_arr, pad], axis=1)
        elif X_arr.shape[1] > expected_F:
            X_arr = X_arr[:, :expected_F]

        return X_arr, np.array(y_human)

    # ── Training ──────────────────────────────────────────────────────────────

    # Rare classes: bring each up to at least this many training steps via augmentation
    AUGMENT_TARGET = 2_000

    def train(self, dataset_names: list, max_episodes_each: int = 80):
        from augmentation import augment_rare_classes, compute_class_counts

        print(f"\nLoading training data from {len(dataset_names)} datasets...")
        episodes = load_training_data(dataset_names, max_episodes_each)
        print(f"Total episodes loaded: {len(episodes)}")

        # build step-level training set (weak supervision, weight = 1.0)
        X_rows, y_rows, w_rows = [], [], []
        for seq, ep_label, ds_name in episodes:
            feats  = extract_window_features(seq)      # (T, F)
            labels = generate_weak_labels(seq)         # (T,) strings
            for feat, label in zip(feats, labels):
                X_rows.append(feat)
                y_rows.append(label)
                w_rows.append(1.0)

        # ── Balance: cap stuck_joint to 2× nominal to prevent class domination ──
        # stuck_joint fires heavily on slow-moving robots; without a cap it
        # accounts for >60% of training data and biases the model against nominal.
        nominal_n  = sum(1 for y in y_rows if y == "nominal")
        stuck_cap  = nominal_n * 2
        stuck_idxs = [i for i, y in enumerate(y_rows) if y == "stuck_joint"]
        if len(stuck_idxs) > stuck_cap:
            rng_bal  = np.random.RandomState(99)
            keep_set = set(rng_bal.choice(stuck_idxs, stuck_cap, replace=False).tolist())
            drop_set = set(stuck_idxs) - keep_set
            mask     = [i not in drop_set for i in range(len(y_rows))]
            X_rows   = [x for x, k in zip(X_rows, mask) if k]
            y_rows   = [y for y, k in zip(y_rows, mask) if k]
            w_rows   = [w for w, k in zip(w_rows, mask) if k]
            print(f"  [balance] stuck_joint capped: {len(stuck_idxs):,} → {stuck_cap:,} steps "
                  f"(2× nominal={nominal_n:,})")

        # ── Synthetic augmentation for rare failure classes ────────────────────
        current_counts = compute_class_counts(X_rows, y_rows)
        target_counts  = {
            cls: self.AUGMENT_TARGET
            for cls in FAILURE_CLASSES
            if cls != "nominal" and current_counts.get(cls, 0) < self.AUGMENT_TARGET
        }
        if target_counts:
            print(f"\n  ★ Augmenting {len(target_counts)} rare classes "
                  f"to {self.AUGMENT_TARGET:,} steps each...")
            aug_episodes = augment_rare_classes(
                episodes, target_counts, current_counts, seed=42
            )
            for aug_seq, aug_labels in aug_episodes:
                feats = extract_window_features(aug_seq)
                for feat, label in zip(feats, aug_labels):
                    # Only inject the failure-labeled steps — skip nominal padding
                    # to avoid flooding the training set with synthetic nominals
                    if label != "nominal":
                        X_rows.append(feat)
                        y_rows.append(label)
                        w_rows.append(1.5)   # slightly upweighted vs real weak labels
            print(f"  Augmented training set: {len(X_rows):,} steps total")

        # inject human-verified corrections at HUMAN_LABEL_WEIGHT× importance
        X_human, y_human = self._load_human_corrections()
        n_human = len(y_human)
        if n_human > 0:
            print(f"\n  ★ Injecting {n_human} human-corrected steps "
                  f"(weight = {HUMAN_LABEL_WEIGHT}× weak labels)")
            X_rows.extend(X_human.tolist())
            y_rows.extend(y_human.tolist())
            w_rows.extend([HUMAN_LABEL_WEIGHT] * n_human)
        else:
            print("  (No human corrections found — training on weak labels only)")

        X = np.array(X_rows)
        y = np.array(y_rows)
        w = np.array(w_rows, dtype=float)
        self.feature_dim = X.shape[1]

        # Store unscaled training distribution for outlier detection at inference
        self._train_feat_mean = X.mean(axis=0)
        self._train_feat_std  = X.std(axis=0) + 1e-8

        print(f"\nTraining set: {X.shape[0]:,} steps × {X.shape[1]} features")
        print("Label distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  {cls:20s}: {cnt:6,}  ({cnt/len(y)*100:.1f}%)")

        # scale features
        X_scaled = self.scaler.fit_transform(X)
        y_enc    = self.le.transform(y)

        # train/val split — carry sample weights through the split
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X_scaled, y_enc, w, test_size=0.15, random_state=42, stratify=y_enc)

        n_human_tr = int((w_tr > 1).sum())
        if n_human_tr > 0:
            print(f"  → {n_human_tr} human-verified steps in training split "
                  f"(weighted {HUMAN_LABEL_WEIGHT}×)")

        # ── 1. Train raw RandomForest ─────────────────────────────────────────
        # Tuned settings vs defaults:
        #   n_estimators  150 → 300  : more trees = lower variance, better rare-class recall
        #   max_depth      12 → 20   : deeper trees capture complex interaction features
        #   min_samples_leaf 5 → 2   : finer splits help rare classes (self_collision, overshoot)
        #   max_features  'sqrt'→0.4 : more features per split for 68-dim feature space
        print(f"\nTraining RandomForest on {len(X_tr):,} steps...")
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_tr, y_tr, sample_weight=w_tr)

        # ── 2. Platt scaling — calibrate probabilities on held-out val set ────
        # RF confidence scores are known to be overconfident.
        # CalibratedClassifierCV(cv='prefit') fits a logistic (sigmoid) layer
        # on top of the already-trained RF using the val set, so calibration
        # never sees the training data. This makes the confidence threshold
        # for human review routing actually meaningful.
        print("  Fitting Platt scaling on validation set...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method="sigmoid", cv="prefit"
        )
        self.calibrated_model.fit(X_val, y_val)

        # measure calibration quality: Brier score (lower = better)
        y_prob_raw  = self.model.predict_proba(X_val)
        y_prob_cal  = self.calibrated_model.predict_proba(X_val)
        present_classes = sorted(set(y_tr) | set(y_val))
        # macro-average Brier score across classes
        brier_raw = np.mean([
            brier_score_loss((y_val == c).astype(int), y_prob_raw[:, i])
            for i, c in enumerate(self.model.classes_)
            if c in present_classes
        ])
        brier_cal = np.mean([
            brier_score_loss((y_val == c).astype(int), y_prob_cal[:, j])
            for j, c in enumerate(self.calibrated_model.classes_)
            if c in present_classes
        ])
        self.calibration_report = {
            "brier_score_raw":       round(float(brier_raw), 5),
            "brier_score_calibrated": round(float(brier_cal), 5),
            "improvement_pct":        round((brier_raw - brier_cal) / brier_raw * 100, 1),
            "method": "Platt scaling (sigmoid)",
        }
        print(f"  Brier score — raw RF: {brier_raw:.5f}  →  calibrated: {brier_cal:.5f}"
              f"  ({self.calibration_report['improvement_pct']:+.1f}%)")

        # ── 3. Evaluate calibrated model ──────────────────────────────────────
        y_pred = self.calibrated_model.predict(X_val)
        present_names   = self.le.inverse_transform(present_classes).tolist()
        report = classification_report(
            y_val, y_pred,
            labels=present_classes,
            target_names=present_names,
            output_dict=True, zero_division=0,
        )
        self.train_report   = report
        self.train_accuracy = float((y_pred == y_val).mean())   # always reliable
        self.datasets_used  = dataset_names

        print(f"\nValidation Results (calibrated model):")
        acc = self.train_accuracy
        print(f"  Overall accuracy : {acc:.3f}")
        for cls in FAILURE_CLASSES:
            if cls in report:
                r = report[cls]
                print(f"  {cls:20s}  prec={r['precision']:.2f}  "
                      f"rec={r['recall']:.2f}  f1={r['f1-score']:.2f}")
        print(f"\n  Confidence routing threshold : {self.REVIEW_THRESHOLD}")
        n_low = (y_prob_cal.max(axis=1) < self.REVIEW_THRESHOLD).sum()
        print(f"  Steps routed to human review : {n_low}/{len(y_val)} "
              f"({n_low/len(y_val)*100:.1f}%)")

        self.save()
        return report

    # ── Inference ─────────────────────────────────────────────────────────────

    def annotate(self, state_seq: np.ndarray) -> dict:
        """
        Annotate a single episode trajectory.

        state_seq : (T, D) joint state array
        Returns dict with per-step labels, confidences, failure counts, 3D coords.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        T = len(state_seq)
        feats  = extract_window_features(state_seq)
        scaled = self.scaler.transform(feats)

        # Use calibrated model for inference if available, else fall back to raw RF
        infer_model = self.calibrated_model if self.calibrated_model is not None else self.model

        pred_enc  = infer_model.predict(scaled)
        pred_prob = infer_model.predict_proba(scaled)  # (T, n_classes_seen) — calibrated
        labels    = self.le.inverse_transform(pred_enc).tolist()
        confs     = pred_prob.max(axis=1).tolist()

        # ── Mahalanobis outlier guard ─────────────────────────────────────────
        # If a step's feature vector is far from the training distribution
        # (measured by mean z-score across features), the model is operating
        # outside its experience. Cap confidence at 0.65 and route to review.
        # This prevents conf=1.0 wrong predictions on high-sensor-noise inputs.
        OUTLIER_Z_THRESHOLD = 4.0   # mean z-score above this = likely OOD
        OUTLIER_CONF_CAP    = 0.65
        if self._train_feat_mean is not None:
            z_scores  = np.abs((feats - self._train_feat_mean) / self._train_feat_std)
            mean_z    = z_scores.mean(axis=1)      # (T,) — one score per step
            for i in range(T):
                if mean_z[i] > OUTLIER_Z_THRESHOLD and confs[i] > OUTLIER_CONF_CAP:
                    confs[i] = OUTLIER_CONF_CAP    # cap confidence for OOD inputs

        # ── Unknown failure type detection ────────────────────────────────────
        # If no class reaches 0.75 confidence the model is genuinely uncertain.
        # Instead of forcing the step into the nearest class (which produces a
        # confident wrong label), flag it as "unknown_failure_type" and route
        # to human review. Repeated unknowns (>10 from the same client) signal
        # a new failure class that should be added to the taxonomy.
        UNKNOWN_THRESHOLD = 0.75
        unknown_steps = []
        for i in range(T):
            if pred_prob[i].max() < UNKNOWN_THRESHOLD:
                labels[i] = "unknown_failure_type"
                unknown_steps.append(i)

        # steps below the confidence threshold are flagged for human review
        needs_review = [
            c < self.REVIEW_THRESHOLD or labels[i] == "unknown_failure_type"
            for i, c in enumerate(confs)
        ]

        # store raw (unscaled) feature vectors for uncertain steps
        # so human corrections can be re-injected with proper features at retrain time
        lowconf_features: dict = {
            i: feats[i].tolist()
            for i, flag in enumerate(needs_review) if flag
        }

        # anomaly score = 1 - P(nominal) using calibrated probabilities
        nom_encoded = int(self.le.transform(["nominal"])[0])
        inf_classes = list(infer_model.classes_)
        if nom_encoded in inf_classes:
            nom_idx     = inf_classes.index(nom_encoded)
            anom_scores = (1 - pred_prob[:, nom_idx]).tolist()
        else:
            anom_scores = [0.5] * T   # fallback if nominal never seen in training

        # 3D trajectory via PCA
        pca    = PCA(n_components=3)
        coords = pca.fit_transform(state_seq).astype(float).tolist()

        all_label_keys = FAILURE_CLASSES + ["unknown_failure_type"]
        failure_counts = {k: labels.count(k) for k in all_label_keys}
        dominant = max(
            [k for k in all_label_keys if k != "nominal"],
            key=lambda k: failure_counts[k]
        ) if any(l != "nominal" for l in labels) else "nominal"

        # active learning: rank uncertain steps by information gain
        # so human reviewers tackle the most valuable steps first
        al_ranked: list = []
        if lowconf_features:
            try:
                from active_learning import ActiveLearningSelector
                lc_indices = sorted(lowconf_features.keys())
                lc_feats   = np.array([lowconf_features[i] for i in lc_indices])
                lc_probs   = pred_prob[lc_indices]
                selector   = ActiveLearningSelector(budget=min(20, len(lc_indices)),
                                                    diversity_frac=0.4)
                al_ranked  = selector.rank(lc_feats, lc_probs,
                                           step_indices=lc_indices)
            except Exception:
                pass  # graceful degradation — active learning is optional

        n_review = sum(needs_review)
        return {
            "n_steps":          T,
            "labels":           labels,
            "confidences":      confs,
            "needs_review":     needs_review,      # bool per step — route to human
            "n_needs_review":   n_review,          # count for quick filtering
            "review_rate":      round(n_review / T, 4),
            "anomaly_scores":   anom_scores,
            "coords_3d":        coords,
            "failure_counts":   failure_counts,
            "dominant_failure": dominant,
            "peak_score":       float(max(anom_scores)),
            "peak_step":        int(np.argmax(anom_scores)),
            "calibrated":       self.calibrated_model is not None,
            # human-in-the-loop fields
            "lowconf_features": lowconf_features,  # step_idx → unscaled feat vec
            "al_ranked":        al_ranked,          # steps sorted by info gain
            # unknown failure detection
            "unknown_steps":    unknown_steps,      # step indices where no class hit 0.75
            "n_unknown":        len(unknown_steps),
        }

    # ── Unknown failure pattern tracking ──────────────────────────────────────

    def track_unknowns(self, result: dict, client_id: str = "base",
                       promote_threshold: int = 10) -> list:
        """
        Log unknown_failure_type steps for a client. When the same feature
        cluster is flagged >promote_threshold times it gets surfaced as a
        candidate new failure class.

        Returns list of promoted candidate class names (usually empty).
        """
        if not result.get("unknown_steps"):
            return []

        pattern_path = OUTPUT_DIR / f"unknown_patterns_{client_id}.json"
        patterns     = json.loads(pattern_path.read_text()) if pattern_path.exists() else []

        # Each unknown step stored as a compact fingerprint
        for step_idx in result["unknown_steps"]:
            conf    = result["confidences"][step_idx]
            feat    = result.get("lowconf_features", {}).get(step_idx, [])
            patterns.append({
                "step":      step_idx,
                "conf":      round(conf, 3),
                "timestamp": datetime.utcnow().isoformat() if "datetime" in dir() else "",
                "feat_hash": hashlib.md5(str(feat[:8]).encode()).hexdigest()[:6],
            })
        pattern_path.write_text(json.dumps(patterns, indent=2))

        # Check if any feature cluster has been seen >promote_threshold times
        from collections import Counter
        hash_counts = Counter(p["feat_hash"] for p in patterns)
        promoted    = [h for h, cnt in hash_counts.items() if cnt >= promote_threshold]
        if promoted:
            print(f"  [unknown] {len(promoted)} cluster(s) seen >{promote_threshold}× for "
                  f"client={client_id} — candidate new failure class(es): {promoted}")
        return promoted

    def annotate_file(self, path: str) -> dict:
        """
        Annotate all episodes in an HDF5 or Parquet file.
        Returns annotation dict + writes annotated HDF5 to benchmark_output/.
        """
        path = Path(path)
        print(f"Annotating {path.name}...")

        if path.suffix == ".h5":
            with h5py.File(path) as f:
                features = f["features"][:]
                labels   = f["true_labels"][:] if "true_labels" in f else None
            # features here are episode summaries — expand to pseudo-sequences
            state_seqs = [feat.reshape(-1, max(1, len(feat) // 10))
                          for feat in features]

        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            state_cols = [c for c in df.columns if "observation.state" in c]
            if not state_cols:
                raise ValueError(f"No state columns in {path}")
            def expand(series):
                first = series.iloc[0]
                return pd.DataFrame(series.tolist(), index=series.index) \
                       if hasattr(first, "__len__") else series.to_frame()
            state_df = expand(df[state_cols[0]]).astype(np.float32)
            df["_state"] = list(state_df.values)
            ep_col = "episode_index"
            state_seqs = [np.stack(df[df[ep_col]==eid]["_state"].values)
                          for eid in sorted(df[ep_col].unique())[:200]]
            labels = None
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        annotations = []
        for i, seq in enumerate(state_seqs):
            ann = self.annotate(seq)
            ann["episode_index"] = i
            if labels is not None:
                ann["true_label"] = int(labels[i])
                ann["label_str"]  = "FAILURE" if labels[i] else "OK"
            annotations.append(ann)
            if (i + 1) % 20 == 0:
                print(f"  Annotated {i+1}/{len(state_seqs)} episodes")

        result = {
            "source_file":  str(path),
            "n_episodes":   len(annotations),
            "model_version": "RobotAnnotator v1.0",
            "datasets_trained_on": self.datasets_used,
            "annotations":  annotations,
        }

        # write annotated HDF5
        safe    = path.stem
        out_h5  = OUTPUT_DIR / f"{safe}_annotated.h5"
        with h5py.File(out_h5, "w") as f:
            for i, ann in enumerate(annotations):
                grp = f.create_group(f"episode_{i:04d}")
                grp.create_dataset("anomaly_scores", data=ann["anomaly_scores"])
                grp.create_dataset("coords_3d",      data=ann["coords_3d"])
                grp.attrs["dominant_failure"] = ann["dominant_failure"]
                grp.attrs["peak_score"]       = ann["peak_score"]
                # store labels as byte strings
                grp.create_dataset("labels", data=np.array(ann["labels"], dtype="S32"))
                if "true_label" in ann:
                    grp.attrs["true_label"] = ann["true_label"]

        out_json = OUTPUT_DIR / f"{safe}_annotations.json"
        out_json.write_text(json.dumps(result, indent=2))

        print(f"\nSaved annotated HDF5 : {out_h5}")
        print(f"Saved annotations    : {out_json}")
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({
                "model":               self.model,
                "calibrated_model":    self.calibrated_model,
                "scaler":              self.scaler,
                "le":                  self.le,
                "feature_dim":         self.feature_dim,
                "train_report":        self.train_report,
                "train_accuracy":      self.train_accuracy,
                "calibration_report":  self.calibration_report,
                "datasets_used":       self.datasets_used,
                "train_feat_mean":     self._train_feat_mean,
                "train_feat_std":      self._train_feat_std,
            }, f)
        print(f"\nModel saved: {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RobotAnnotator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        ann = cls()
        ann.model              = state["model"]
        ann.calibrated_model   = state.get("calibrated_model")   # None in old pkls
        ann.scaler             = state["scaler"]
        ann.le                 = state["le"]
        ann.feature_dim        = state["feature_dim"]
        ann.train_report       = state["train_report"]
        ann.train_accuracy     = state.get("train_accuracy") or (
            ann.train_report.get("accuracy")
            or ann.train_report.get("weighted avg", {}).get("f1-score", 0.0)
            if ann.train_report else 0.0
        )
        ann.calibration_report = state.get("calibration_report")
        ann.datasets_used      = state["datasets_used"]
        ann._train_feat_mean   = state.get("train_feat_mean")
        ann._train_feat_std    = state.get("train_feat_std")
        cal_str = " (calibrated)" if ann.calibrated_model else " (uncalibrated)"
        print(f"Model loaded from {path}{cal_str}")
        print(f"  Trained on: {ann.datasets_used}")
        print(f"  Validation accuracy: {ann.train_accuracy:.3f}")
        if ann.calibration_report:
            print(f"  Brier score: {ann.calibration_report['brier_score_calibrated']:.5f} "
                  f"(vs {ann.calibration_report['brier_score_raw']:.5f} uncalibrated, "
                  f"{ann.calibration_report['improvement_pct']:+.1f}%)")
        return ann

    def model_card(self) -> dict:
        """Return a benchmark-style model card for the annotation model."""
        if not self.train_report:
            return {}
        card = {
            "model":               "RobotAnnotator v1.1",
            "base_model":          "RandomForestClassifier (n_estimators=300, max_depth=20)",
            "calibration":         "Platt scaling (sigmoid) on held-out validation set",
            "confidence_threshold": self.REVIEW_THRESHOLD,
            "task":                "Step-level failure type classification",
            "failure_classes":     FAILURE_CLASSES,
            "datasets_trained_on": self.datasets_used,
            "validation_accuracy": round(self.train_accuracy or 0.0, 4),
            "train_samples":       sum(
                len(v) for v in getattr(self, "_class_counts", {}).values()
            ) if hasattr(self, "_class_counts") else None,
            "calibration_report":  self.calibration_report or {},
            "per_class": {
                cls: {
                    "precision": round(self.train_report[cls]["precision"], 3),
                    "recall":    round(self.train_report[cls]["recall"],    3),
                    "f1":        round(self.train_report[cls]["f1-score"],  3),
                }
                for cls in FAILURE_CLASSES if cls in self.train_report
            },
        }
        return card


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true", help="Train model on open source datasets")
    parser.add_argument("--annotate", action="store_true", help="Annotate a file with trained model")
    parser.add_argument("--input",    type=str, help="HDF5 or parquet file to annotate")
    parser.add_argument("--datasets", nargs="+",
                        default=[
                            "lerobot/xarm_lift_medium_replay",
                            "lerobot/xarm_push_medium_replay",
                            "lerobot/aloha_sim_transfer_cube_human",
                            "lerobot/aloha_sim_insertion_human",
                        ],
                        help="Datasets to train on")
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.train:
        ann = RobotAnnotator()
        report = ann.train(args.datasets, args.max_episodes)

        card = ann.model_card()
        card_path = OUTPUT_DIR / "annotation_model_card.json"
        card_path.write_text(json.dumps(card, indent=2))
        print(f"\nModel card saved: {card_path}")

    elif args.annotate:
        if not args.input:
            parser.error("--annotate requires --input")
        ann = RobotAnnotator.load()

        if Path(args.input).exists():
            ann.annotate_file(args.input)
        else:
            # treat as a LeRobot dataset name
            dataset_name = args.input
            print(f"Treating input as dataset name: {dataset_name}")
            episodes = load_training_data([dataset_name], max_episodes_each=100)
            annotations = []
            for i, (seq, ep_label, _) in enumerate(episodes):
                a = ann.annotate(seq)
                a["episode_index"] = i
                a["true_label"]    = ep_label
                a["label_str"]     = "FAILURE" if ep_label else "OK"
                annotations.append(a)
                if (i + 1) % 20 == 0:
                    print(f"  Annotated {i+1}/{len(episodes)} episodes")
            result = {
                "dataset":            dataset_name,
                "n_episodes":         len(annotations),
                "n_failures":         int(sum(e["true_label"] for e in annotations)),
                "feature_dim":        episodes[0][0].shape[1] if episodes else 0,
                "model_version":      "RobotAnnotator v1.0",
                "datasets_trained_on": ann.datasets_used,
                "annotations":        annotations,
            }
            safe = dataset_name.replace("/", "_")
            out  = OUTPUT_DIR / f"{safe}_annotations.json"
            out.write_text(json.dumps(result, indent=2))
            print(f"\nSaved: {out}")

    else:
        parser.print_help()
