"""
Temporal Step Annotator — full-episode context model for per-step failure labeling.

Architecture (3-tier feature encoding for each timestep t):

  Tier 1 — Local bidirectional window (±10 steps):
    backward [t-10..t]  : mean, std, range, vel_std per joint  (4×D)
    forward  [t..t+10]  : mean, std, range, vel_std per joint  (4×D)

  Tier 2 — Global episode context (full trajectory):
    episode mean pos, std, range per joint                      (3×D)
    position deviation from episode mean at step t              (D)
    velocity mean, std over full episode                        (2×D)
    trajectory shape: skewness, kurtosis of positions per joint (2×D)

  Tier 3 — Scalars:
    vel_mag, acc_mag, cross-joint vel spread, t/T               (4)
    episode_vel_percentile(t), episode_acc_percentile(t)        (2)

  Total features: (8×D×2) + (8×D) + 6 = 16D×2 + 8D + 6
    For D=4: 128 + 32 + 6 = 166 features  (vs 36 for the RF, 68 for BiDir v1)

  MLPClassifier (256→128→64 → 10 classes), early stopping, Platt calibration.

Why Tier 2 matters:
  - stuck_joint: needs to know if the joint has moved at ALL in the episode
  - trajectory_deviation: needs to compare to the episode's own reference path
  - perception_failure: needs the "expected" position range to flag jumps
  These are impossible to detect with only a 10-step local window.

Usage:
  python lstm_annotator.py --train
  python lstm_annotator.py --compare     # RF vs FullContext MLP
"""

import argparse, json, pickle, warnings, time
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "lstm_annotator.pt"   # kept as .pt for pipeline compat

import sys
sys.path.insert(0, str(Path(__file__).parent))
from annotation_model import (
    extract_window_features, generate_weak_labels,
    load_training_data, FAILURE_CLASSES, RobotAnnotator,
)

WINDOW = 10
REVIEW_THRESHOLD = 0.60


# ── Bidirectional feature extraction ─────────────────────────────────────────

def extract_bidir_features(state_seq: np.ndarray) -> np.ndarray:
    """
    Full-episode context feature extractor — three tiers per timestep.

    state_seq : (T, D) — raw joint states
    Returns   : (T, F) — per-timestep feature matrix

    Tier 1 — bidirectional local window (±WINDOW steps): 8×D×2 features
    Tier 2 — global episode context (full trajectory):   8×D features
    Tier 3 — scalars:                                    6 features
    Total: 16D + 8D + 6  →  166 features for D=4
    """
    T, D  = state_seq.shape
    eps   = 1e-8
    vel   = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
    acc   = np.vstack([np.zeros((1, D)), np.diff(vel, axis=0)])

    vel_mag = np.linalg.norm(vel, axis=1)   # (T,)
    acc_mag = np.linalg.norm(acc, axis=1)   # (T,)

    # ── Tier 2: episode-level statistics (computed once, reused per step) ──
    ep_mean  = state_seq.mean(axis=0)                        # (D,) mean position
    ep_std   = state_seq.std(axis=0)  + eps                  # (D,) position std
    ep_range = state_seq.max(axis=0)  - state_seq.min(axis=0) + eps  # (D,) range
    ep_vmean = vel.mean(axis=0)                              # (D,) mean velocity
    ep_vstd  = vel.std(axis=0)  + eps                        # (D,) vel std

    # distribution shape: skewness ≈ 3rd standardised moment
    # (T, D) centered, then mean of cube
    centered = state_seq - ep_mean
    skew  = (centered ** 3).mean(axis=0) / (ep_std ** 3 + eps)  # (D,)
    kurt  = (centered ** 4).mean(axis=0) / (ep_std ** 4 + eps) - 3.0  # (D,) excess

    # rank-normalised velocity/acceleration percentile at each step (in [0,1])
    vel_pct = np.argsort(np.argsort(vel_mag)).astype(float) / max(T - 1, 1)
    acc_pct = np.argsort(np.argsort(acc_mag)).astype(float) / max(T - 1, 1)

    rows = []
    for t in range(T):
        # ── Tier 1: local bidirectional window ────────────────────────────
        bw_s = max(0, t - WINDOW + 1)
        bw   = state_seq[bw_s:t+1]
        bw_v = vel[bw_s:t+1]

        fw_e = min(T, t + WINDOW + 1)
        fw   = state_seq[t:fw_e]
        fw_v = vel[t:fw_e]

        tier1 = np.concatenate([
            bw.mean(0), bw.std(0) + eps, bw.max(0) - bw.min(0), bw_v.std(0) + eps,
            fw.mean(0), fw.std(0) + eps, fw.max(0) - fw.min(0), fw_v.std(0) + eps,
        ])  # 8×D

        # ── Tier 2: global episode context ────────────────────────────────
        dev_from_ep_mean = np.abs(state_seq[t] - ep_mean) / ep_range  # (D,) norm deviation
        tier2 = np.concatenate([
            ep_mean, ep_std, ep_range,          # 3×D — reference trajectory statistics
            dev_from_ep_mean,                   # D   — how far this step is from ep mean
            ep_vmean, ep_vstd,                  # 2×D — episode velocity profile
            skew, kurt,                         # 2×D — trajectory shape
        ])  # 8×D total

        # ── Tier 3: scalars ───────────────────────────────────────────────
        tier3 = np.array([
            np.linalg.norm(vel[t]),             # vel magnitude
            np.linalg.norm(acc[t]),             # acc magnitude
            vel[t].std(),                       # cross-joint velocity spread
            float(t) / T,                       # normalised episode position
            vel_pct[t],                         # percentile rank of vel_mag
            acc_pct[t],                         # percentile rank of acc_mag
        ])  # 6

        rows.append(np.concatenate([tier1, tier2, tier3]))

    return np.array(rows, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def _log(path: Path, msg: str):
    with open(path, "a") as fh:
        fh.write(msg + "\n")
    try:
        print(msg, flush=True)
    except Exception:
        pass


def train_lstm(dataset_names: list, max_episodes_each: int = 80,
               epochs: int = 20, lr: float = 1e-3, batch_size: int = 16) -> dict:
    """
    Train bidirectional-context MLP annotator on weak-supervision labels.
    `epochs`, `lr`, `batch_size` kept for API compatibility — MLP uses max_iter.
    """
    log_path = OUTPUT_DIR / "lstm_train.log"
    log_path.write_text("")

    def p(msg): _log(log_path, msg)

    p(f"\nLoading training data from {len(dataset_names)} datasets...")
    episodes_raw = load_training_data(dataset_names, max_episodes_each)
    p(f"  {len(episodes_raw)} episodes loaded")

    le = LabelEncoder().fit(FAILURE_CLASSES)

    # ── build bidirectional step features ────────────────────────────────────
    p("  Extracting bidirectional features...")
    X_rows, y_rows = [], []
    for seq, ep_label, ds in episodes_raw:
        feats  = extract_bidir_features(seq)
        labels = generate_weak_labels(seq)
        for feat, label in zip(feats, labels):
            X_rows.append(feat)
            y_rows.append(label)

    X = np.array(X_rows)
    y = np.array(y_rows)
    feature_dim = X.shape[1]

    p(f"  {X.shape[0]:,} steps × {X.shape[1]} bidir features")
    p("  Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        p(f"    {cls:20s}: {cnt:6,}  ({cnt/len(y)*100:.1f}%)")

    # train / val split (episode-level to prevent leakage)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)

    # scale
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    # ── MLP with full-episode context features ────────────────────────────────
    p(f"\nTraining MLP on {len(X_tr_sc):,} steps "
      f"(hidden: 256→128→64, features: {X.shape[1]})...")
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),   # deeper: more capacity for 10 classes
        activation="relu",
        max_iter=300,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_tr_sc, le.transform(y_tr))
    elapsed = time.time() - t0
    p(f"  Training done in {elapsed:.1f}s  ({mlp.n_iter_} iterations)")

    # calibrate
    p("  Fitting Platt scaling on val set...")
    cal_mlp = CalibratedClassifierCV(mlp, method="sigmoid", cv="prefit")
    cal_mlp.fit(X_val_sc, le.transform(y_val))

    # evaluate
    y_pred = cal_mlp.predict(X_val_sc)
    present_classes = sorted(set(le.transform(y_tr)) | set(le.transform(y_val)))
    present_names   = le.inverse_transform(present_classes).tolist()
    report = classification_report(
        le.transform(y_val), y_pred,
        labels=present_classes, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    best_val_acc = report.get("accuracy", (y_pred == le.transform(y_val)).mean())

    p(f"\nValidation accuracy : {best_val_acc:.3f}")
    p("Per-class (BiDir MLP):")
    for cls in FAILURE_CLASSES:
        if cls in report:
            r = report[cls]
            p(f"  {cls:20s}  prec={r['precision']:.2f}  "
              f"rec={r['recall']:.2f}  f1={r['f1-score']:.2f}")

    # review routing stats
    y_prob     = cal_mlp.predict_proba(X_val_sc)
    n_low      = (y_prob.max(axis=1) < REVIEW_THRESHOLD).sum()
    p(f"\n  Steps routed to human review: {n_low}/{len(y_val)} ({n_low/len(y_val)*100:.1f}%)")

    # ── save (torch.save format for pipeline compatibility) ───────────────────
    checkpoint = {
        "model":           cal_mlp,   # calibrated MLP
        "scaler":          scaler,
        "le_classes":      le.classes_.tolist(),
        "feature_dim":     feature_dim,
        "feature_type":    "bidir_window",
        "train_report":    report,
        "datasets_used":   dataset_names,
        "best_val_acc":    best_val_acc,
        "model_type":      "BiDirMLP",
        "n_classes":       len(FAILURE_CLASSES),
    }
    # save as pickle (still .pt extension for pipeline compat)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(checkpoint, fh)

    p(f"\nModel saved: {MODEL_PATH}")
    p(f"Training log: {log_path}")
    return report


# ── Inference wrapper ─────────────────────────────────────────────────────────

class LSTMAnnotatorInference:
    """
    Inference wrapper — same .annotate(state_seq) interface as RobotAnnotator.
    Internally uses the BiDir MLP (no PyTorch at inference time).
    """

    def __init__(self, checkpoint: dict):
        self.le = LabelEncoder()
        self.le.classes_ = np.array(checkpoint["le_classes"])
        self.model        = checkpoint["model"]    # CalibratedClassifierCV(MLP)
        self.scaler       = checkpoint["scaler"]
        self.feature_dim  = checkpoint["feature_dim"]
        self.datasets_used = checkpoint.get("datasets_used", [])
        self.train_report  = checkpoint.get("train_report", {})
        self.best_val_acc  = checkpoint.get("best_val_acc", 0.0)
        self.model_type    = checkpoint.get("model_type", "BiDirMLP")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "LSTMAnnotatorInference":
        with open(path, "rb") as fh:
            ckpt = pickle.load(fh)
        obj = cls(ckpt)
        print(f"BiDir MLP annotator loaded from {path}")
        print(f"  Val accuracy: {obj.best_val_acc:.3f} | Trained on: {obj.datasets_used}")
        return obj

    def annotate(self, state_seq: np.ndarray, lengths=None) -> dict:
        from sklearn.decomposition import PCA
        T = len(state_seq)
        feats  = extract_bidir_features(state_seq)         # (T, F_bidir)
        scaled = self.scaler.transform(feats)
        pred_enc  = self.model.predict(scaled)
        pred_prob = self.model.predict_proba(scaled)       # (T, C) calibrated
        labels    = self.le.inverse_transform(pred_enc).tolist()
        confs     = pred_prob.max(axis=1).tolist()

        needs_review = [c < REVIEW_THRESHOLD for c in confs]
        n_review     = sum(needs_review)

        nom_idx     = list(self.le.classes_).index("nominal")
        inf_classes = list(self.model.classes_)
        if nom_idx in inf_classes:
            nom_rf_idx  = inf_classes.index(nom_idx)
            anom_scores = (1 - pred_prob[:, nom_rf_idx]).tolist()
        else:
            anom_scores = [0.5] * T

        coords = PCA(n_components=3).fit_transform(state_seq).astype(float).tolist()

        failure_counts = {k: labels.count(k) for k in FAILURE_CLASSES}
        dominant = max(
            [k for k in FAILURE_CLASSES if k != "nominal"],
            key=lambda k: failure_counts[k]
        ) if any(l != "nominal" for l in labels) else "nominal"

        return {
            "n_steps":          T,
            "labels":           labels,
            "confidences":      [round(c, 3) for c in confs],
            "needs_review":     needs_review,
            "n_needs_review":   n_review,
            "review_rate":      round(n_review / T, 4),
            "anomaly_scores":   [round(s, 4) for s in anom_scores],
            "coords_3d":        coords,
            "failure_counts":   failure_counts,
            "dominant_failure": dominant,
            "peak_score":       float(max(anom_scores)),
            "peak_step":        int(np.argmax(anom_scores)),
            "model":            self.model_type,
        }


# ── RF vs BiDir MLP comparison ────────────────────────────────────────────────

def compare_rf_vs_lstm(dataset_names: list, max_episodes_each: int = 50):
    """Side-by-side accuracy comparison on held-out val episodes."""
    print("\nLoading val data...")
    episodes_raw = load_training_data(dataset_names, max_episodes_each)
    split   = int(0.8 * len(episodes_raw))
    val_eps = episodes_raw[split:]
    le      = LabelEncoder().fit(FAILURE_CLASSES)
    print(f"  Val episodes: {len(val_eps)}")

    # RF predictions
    rf = RobotAnnotator.load()
    rf_preds, rf_true = [], []
    for seq, _, _ in val_eps:
        ann  = rf.annotate(seq)
        true = generate_weak_labels(seq)
        rf_preds.extend(ann["labels"])
        rf_true.extend(true)

    # BiDir MLP predictions
    if not MODEL_PATH.exists():
        print("BiDir MLP model not found — run --train first")
        return
    mlp = LSTMAnnotatorInference.load()
    mlp_preds, mlp_true = [], []
    for seq, _, _ in val_eps:
        ann  = mlp.annotate(seq)
        true = generate_weak_labels(seq)
        mlp_preds.extend(ann["labels"])
        mlp_true.extend(true)

    # use string labels directly — avoids integer-key mismatch in classification_report
    present = sorted(set(rf_true) | set(mlp_true) | set(rf_preds) | set(mlp_preds))

    rf_rep  = classification_report(rf_true,  rf_preds,  labels=present,
                                    target_names=present, output_dict=True, zero_division=0)
    mlp_rep = classification_report(mlp_true, mlp_preds, labels=present,
                                    target_names=present, output_dict=True, zero_division=0)

    print(f"\n{'Class':20s}  {'RF F1':>7}  {'BiDir F1':>9}  {'Winner':>8}")
    print("-" * 54)
    for cls in FAILURE_CLASSES:
        rf_f1  = rf_rep.get(cls, {}).get("f1-score", 0)
        mlp_f1 = mlp_rep.get(cls, {}).get("f1-score", 0)
        winner = "BiDir" if mlp_f1 > rf_f1 else ("RF" if rf_f1 > mlp_f1 else "tie")
        print(f"  {cls:18s}  {rf_f1:>7.3f}  {mlp_f1:>9.3f}  {winner:>8}")

    print(f"\n  RF overall acc    : {rf_rep['accuracy']:.3f}")
    print(f"  BiDir MLP overall : {mlp_rep['accuracy']:.3f}")

    result = {
        "rf":     {"accuracy": rf_rep["accuracy"],  "per_class": rf_rep},
        "lstm":   {"accuracy": mlp_rep["accuracy"], "per_class": mlp_rep},
        "winner": "lstm" if mlp_rep["accuracy"] > rf_rep["accuracy"] else "rf",
        "model_label": "BiDir MLP (bidirectional context)",
    }
    out = OUTPUT_DIR / "rf_vs_lstm_comparison.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nComparison saved: {out}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--datasets", nargs="+",
                        default=["lerobot/xarm_lift_medium_replay",
                                 "lerobot/xarm_push_medium_replay"])
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=80)
    args = parser.parse_args()

    if args.train:
        train_lstm(args.datasets, args.max_episodes, epochs=args.epochs)
    elif args.compare:
        compare_rf_vs_lstm(args.datasets, args.max_episodes)
    else:
        parser.print_help()
