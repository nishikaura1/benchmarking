"""
public_eval/large_scale_train.py
==================================
Large-scale training pipeline on real + augmented client data.

Strategy
--------
1. Load real labeled client data (biotech: 188 eps, acme: 85 eps)
2. Stratified 80/20 train/test split — test set is NEVER augmented
3. Augment training split: inject physics failures into nominal episodes
   until each class reaches TARGET_PER_CLASS (default 500)
4. Train RF + HGB + IsolationForest on three configurations:
     A. real_aug    — original real train + augmented synthetic
     B. real_only   — original real train data (no augmentation)
     C. aug_only    — augmented data only (no original real train)
5. Evaluate all three on fixed real held-out test set
6. Report: accuracy, per-class F1, macro-F1, ROC-AUC, confusion matrix
7. Save best model + normalizer to benchmark_output/client_eval/model/

Usage
-----
    python public_eval/large_scale_train.py
    python public_eval/large_scale_train.py --target 300
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from augmentation import INJECTORS  # numpy-array-level injectors
from public_eval.benchmark_runner import extract_episode_features, build_feature_matrix
from public_eval.physics_normalizer import RobotDataNormalizer, PhysicsPreFilter, extract_physics_features

OUT_DIR = Path("benchmark_output/client_eval")
MODEL_DIR = OUT_DIR / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_PHYSICS_FILTER = PhysicsPreFilter()

# ── Failure classes supported by augmentation.py injectors ───────────────────
AUG_CLASS_MAP = {
    "velocity_spike":       "velocity_spike",
    "stuck_joint":          "stuck_joint",
    "gripper_event":        "gripper_event",
    "overshoot":            "overshoot",
    "position_jerk":        "position_jerk",
    "perception_failure":   "perception_failure",
    "trajectory_deviation": "trajectory_deviation",
    # weld_stutter and pipette_clog don't have a dedicated injector —
    # we'll proxy them with position_jerk and perception_failure respectively
    "weld_stutter":         "position_jerk",
    "pipette_clog":         "perception_failure",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_client_json(path: Path, dataset_name: str) -> list[dict]:
    """Load client JSON → common episode schema."""
    with open(path) as f:
        raw = json.load(f)
    episodes = []
    for item in raw:
        seq = np.array(item["seq"], dtype=np.float32)
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        ep = {
            "dataset_name":     dataset_name,
            "episode_id":       item["id"],
            "timesteps":        len(seq),
            "state_seq":        seq,
            "action_seq":       None,
            "video_frames":     None,
            "image_paths":      None,
            "language_task":    None,
            "episode_label":    item["true_label"],
            "step_labels":      None,
            "semantic_labels":  None,
            "failure_category": None if item["true_label"] == "nominal" else item["true_label"],
            "source_label_type": "human",
            "metadata":         {"note": item.get("note", ""), "augmented": False},
        }
        episodes.append(ep)
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_to_target(train_episodes: list[dict], target_per_class: int,
                      seed: int = 42) -> list[dict]:
    """
    For each failure class below target_per_class, generate synthetic episodes
    by injecting the failure pattern into randomly sampled nominal episodes.

    Returns only the NEW synthetic episodes (caller concatenates with originals).
    """
    rng = np.random.RandomState(seed)

    # Pool of nominal episodes to inject into (training only)
    nominal_pool = [ep for ep in train_episodes if ep["episode_label"] == "nominal"]
    if not nominal_pool:
        print("  ⚠ No nominal episodes in training split — cannot augment")
        return []

    # Current counts
    label_counts = Counter(ep["episode_label"] for ep in train_episodes)
    all_classes = [lbl for lbl in label_counts if lbl != "nominal"]

    synthetic = []
    for cls in all_classes:
        current = label_counts.get(cls, 0)
        needed  = max(0, target_per_class - current)
        if needed == 0:
            print(f"  {cls:25s}: already at {current} ≥ {target_per_class}")
            continue

        injector_key = AUG_CLASS_MAP.get(cls)
        injector_fn  = INJECTORS.get(injector_key) if injector_key else None
        if injector_fn is None:
            print(f"  {cls:25s}: no injector available, skipping")
            continue

        print(f"  {cls:25s}: {current} real → generating {needed} synthetic", end="", flush=True)
        generated = 0
        attempts  = 0
        while generated < needed and attempts < needed * 5:
            attempts += 1
            base_ep = nominal_pool[rng.randint(len(nominal_pool))]
            seq = base_ep["state_seq"].copy()   # (T, D) numpy array

            if seq.shape[0] < 10:  # too short
                continue

            ep_rng = np.random.RandomState(int(rng.randint(1_000_000)))
            try:
                new_seq, step_labels = injector_fn(seq, ep_rng)
            except Exception:
                continue

            new_ep = {
                "dataset_name":     base_ep["dataset_name"] + "_aug",
                "episode_id":       f"{cls}_aug_{generated:04d}_from_{base_ep['episode_id']}",
                "timesteps":        len(new_seq),
                "state_seq":        new_seq.astype(np.float32),
                "action_seq":       None,
                "video_frames":     None,
                "image_paths":      None,
                "language_task":    None,
                "episode_label":    cls,
                "step_labels":      step_labels,
                "semantic_labels":  None,
                "failure_category": cls,
                "source_label_type": "synthetic",
                "metadata":         {
                    "augmented": True,
                    "base_episode": base_ep["episode_id"],
                    "injector_used": injector_key,
                },
            }
            synthetic.append(new_ep)
            generated += 1

        print(f"  → {generated} generated")

    return synthetic


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix helpers
# ─────────────────────────────────────────────────────────────────────────────

def episodes_to_Xy(episodes: list[dict]):
    """Build (X, y_labels, y_bin) from episode list."""
    X, labels, _ = build_feature_matrix(episodes)
    y_bin = np.array([0 if lbl == "nominal" else 1 for lbl in labels], dtype=int)
    return X, labels, y_bin


def pad_or_trim(X_a, X_b):
    """Make two feature matrices the same width by zero-padding the narrower one."""
    da, db = X_a.shape[1], X_b.shape[1]
    if da == db:
        return X_a, X_b
    d = max(da, db)
    def _pad(X, d):
        if X.shape[1] == d:
            return X
        pad = np.zeros((X.shape[0], d - X.shape[1]), dtype=X.dtype)
        return np.hstack([X, pad])
    return _pad(X_a, d), _pad(X_b, d)


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(X_train, y_train_labels, X_test, y_test_labels,
                   config_name: str) -> dict:
    """
    Train RF + HGB + IsolationForest on (X_train, y_train_labels).
    Evaluate on (X_test, y_test_labels).
    Returns metrics dict + trained models.
    """
    le = LabelEncoder()
    le.fit(list(set(y_train_labels) | set(y_test_labels)))
    y_tr  = le.transform(y_train_labels)
    y_te  = le.transform(y_test_labels)

    nom_idx = list(le.classes_).index("nominal") if "nominal" in le.classes_ else -1
    y_tr_bin = (y_tr != nom_idx).astype(int)
    y_te_bin = (y_te != nom_idx).astype(int)

    classes     = list(le.classes_)
    n_classes   = len(classes)

    norm = RobotDataNormalizer()
    X_tr_n = norm.fit_transform(X_train)
    X_te_n = norm.transform(X_test)

    results = {
        "config": config_name,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "n_classes": n_classes,
        "class_list": classes,
        "train_label_dist": dict(Counter(y_train_labels)),
        "test_label_dist":  dict(Counter(y_test_labels)),
    }

    # ── IsolationForest (binary: nominal vs failure) ───────────────────────
    nom_mask = (y_tr_bin == 0)
    if nom_mask.sum() >= 5:
        iso = IsolationForest(n_estimators=300, contamination=0.25,
                              random_state=42, n_jobs=-1)
        iso.fit(X_tr_n[nom_mask])
        scores = -iso.score_samples(X_te_n)
        thr = np.percentile(-iso.score_samples(X_tr_n[nom_mask]), 75)
        y_iso = (scores > thr).astype(int)
        iso_f1 = f1_score(y_te_bin, y_iso, average="macro", zero_division=0)
        try:
            iso_auc = roc_auc_score(y_te_bin, scores)
        except Exception:
            iso_auc = None
        results["IsolationForest"] = {
            "binary_macro_f1": round(float(iso_f1), 4),
            "roc_auc": round(float(iso_auc), 4) if iso_auc else None,
            "precision": round(float(precision_score(y_te_bin, y_iso, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_te_bin, y_iso, zero_division=0)), 4),
        }
        results["_iso_model"]   = iso
        results["_iso_norm"]    = norm
        results["_iso_thr"]     = float(thr)

    # ── RandomForest (multiclass + binary) ───────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=20,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr_n, y_tr)
    y_pred_rf    = rf.predict(X_te_n)
    y_pred_rf_bin = (y_pred_rf != nom_idx).astype(int)

    rf_multi_f1  = f1_score(y_te, y_pred_rf, average="macro", zero_division=0)
    rf_bin_f1    = f1_score(y_te_bin, y_pred_rf_bin, average="macro", zero_division=0)
    rf_acc       = accuracy_score(y_te, y_pred_rf)

    # Per-class F1
    per_cls = {}
    cr = classification_report(y_te, y_pred_rf, labels=list(range(n_classes)),
                                target_names=classes, zero_division=0, output_dict=True)
    for cls in classes:
        per_cls[cls] = round(cr.get(cls, {}).get("f1-score", 0.0), 4)

    try:
        proba = rf.predict_proba(X_te_n)
        if proba.shape[1] == 2:
            rf_auc = roc_auc_score(y_te_bin, proba[:, 1])
        else:
            rf_auc = roc_auc_score(y_te_bin, 1 - proba[:, nom_idx] if nom_idx >= 0 else proba.max(1))
    except Exception:
        rf_auc = None

    results["RandomForest"] = {
        "multiclass_macro_f1": round(float(rf_multi_f1), 4),
        "binary_macro_f1":     round(float(rf_bin_f1), 4),
        "accuracy":            round(float(rf_acc), 4),
        "roc_auc":             round(float(rf_auc), 4) if rf_auc else None,
        "per_class_f1":        per_cls,
        "confusion_matrix":    confusion_matrix(y_te, y_pred_rf).tolist(),
    }
    results["_rf_model"]  = rf
    results["_rf_norm"]   = norm
    results["_rf_le"]     = le

    # ── HistGradientBoosting (multiclass) ────────────────────────────────────
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05,
        max_depth=8, min_samples_leaf=5,
        early_stopping=True, class_weight="balanced",
        random_state=42,
    )
    hgb.fit(X_tr_n, y_tr)
    y_pred_hgb    = hgb.predict(X_te_n)
    y_pred_hgb_bin = (y_pred_hgb != nom_idx).astype(int)

    hgb_multi_f1 = f1_score(y_te, y_pred_hgb, average="macro", zero_division=0)
    hgb_bin_f1   = f1_score(y_te_bin, y_pred_hgb_bin, average="macro", zero_division=0)
    hgb_acc      = accuracy_score(y_te, y_pred_hgb)

    per_cls_hgb = {}
    cr_hgb = classification_report(y_te, y_pred_hgb, labels=list(range(n_classes)),
                                    target_names=classes, zero_division=0, output_dict=True)
    for cls in classes:
        per_cls_hgb[cls] = round(cr_hgb.get(cls, {}).get("f1-score", 0.0), 4)

    try:
        proba_hgb = hgb.predict_proba(X_te_n)
        if proba_hgb.shape[1] == 2:
            hgb_auc = roc_auc_score(y_te_bin, proba_hgb[:, 1])
        else:
            hgb_auc = roc_auc_score(y_te_bin, 1 - proba_hgb[:, nom_idx] if nom_idx >= 0 else proba_hgb.max(1))
    except Exception:
        hgb_auc = None

    results["HistGradientBoosting"] = {
        "multiclass_macro_f1": round(float(hgb_multi_f1), 4),
        "binary_macro_f1":     round(float(hgb_bin_f1), 4),
        "accuracy":            round(float(hgb_acc), 4),
        "roc_auc":             round(float(hgb_auc), 4) if hgb_auc else None,
        "per_class_f1":        per_cls_hgb,
    }
    results["_hgb_model"] = hgb
    results["_hgb_norm"]  = norm
    results["_hgb_le"]    = le

    _print_config_results(results)
    return results


def _print_config_results(r: dict):
    rf  = r.get("RandomForest", {})
    hgb = r.get("HistGradientBoosting", {})
    iso = r.get("IsolationForest", {})
    print(f"\n  ┌─ {r['config']} (train={r['n_train']}, test={r['n_test']}) ─────────────────")
    print(f"  │  IsolationForest  binary macro-F1={iso.get('binary_macro_f1','–')}  ROC-AUC={iso.get('roc_auc','–')}")
    print(f"  │  RandomForest     multi  macro-F1={rf.get('multiclass_macro_f1','–')}  binary={rf.get('binary_macro_f1','–')}  acc={rf.get('accuracy','–')}  ROC-AUC={rf.get('roc_auc','–')}")
    print(f"  │  HistGradBoost    multi  macro-F1={hgb.get('multiclass_macro_f1','–')}  binary={hgb.get('binary_macro_f1','–')}  acc={hgb.get('accuracy','–')}")
    print(f"  │  Per-class F1 (RF):")
    for cls, f1 in rf.get("per_class_f1", {}).items():
        bar = "█" * int(f1 * 20)
        print(f"  │    {cls:25s} {f1:.4f}  {bar}")
    print(f"  └──────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(target_per_class: int = 500):
    print("=" * 70)
    print("LARGE-SCALE TRAINING PIPELINE — REAL + AUGMENTED CLIENT DATA")
    print(f"Target per class: {target_per_class}")
    print("=" * 70)

    # ── 1. Load real labeled data ─────────────────────────────────────────────
    bio_path  = Path("benchmark_output/biotech_client_episodes.json")
    acme_path = Path("benchmark_output/acme_client_episodes.json")

    print("\n[1/6] Loading real labeled client data...")
    biotech = load_client_json(bio_path,  "biotech")
    acme    = load_client_json(acme_path, "acme")

    all_real = biotech + acme
    print(f"  Biotech : {len(biotech)} eps  | {dict(Counter(e['episode_label'] for e in biotech))}")
    print(f"  ACME    : {len(acme)} eps  | {dict(Counter(e['episode_label'] for e in acme))}")
    print(f"  Combined: {len(all_real)} eps")

    # ── 2. Stratified train/test split (80/20) on real data ──────────────────
    print("\n[2/6] Stratified 80/20 split on real data (test set is never augmented)...")
    labels_all = [ep["episode_label"] for ep in all_real]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(all_real, labels_all))

    real_train = [all_real[i] for i in train_idx]
    real_test  = [all_real[i] for i in test_idx]

    print(f"  Train : {len(real_train)} eps | {dict(Counter(e['episode_label'] for e in real_train))}")
    print(f"  Test  : {len(real_test)} eps  | {dict(Counter(e['episode_label'] for e in real_test))}")

    # ── 3. Augment training data ──────────────────────────────────────────────
    print(f"\n[3/6] Augmenting training data to {target_per_class} per failure class...")
    aug_episodes = augment_to_target(real_train, target_per_class, seed=42)
    aug_train = real_train + aug_episodes

    aug_label_dist = Counter(ep["episode_label"] for ep in aug_train)
    print(f"\n  Augmented training set: {len(aug_train)} total episodes")
    print(f"  Label distribution:")
    for cls, cnt in sorted(aug_label_dist.items()):
        real_cnt = sum(1 for ep in real_train if ep["episode_label"] == cls)
        syn_cnt  = cnt - real_cnt
        print(f"    {cls:25s}: {cnt:4d}  (real={real_cnt}, synthetic={syn_cnt})")

    # ── 4. Build feature matrices ─────────────────────────────────────────────
    print("\n[4/6] Extracting features...")

    X_aug_train,  y_aug_train_lbl,  _ = episodes_to_Xy(aug_train)
    X_real_train, y_real_train_lbl, _ = episodes_to_Xy(real_train)
    X_aug_only,   y_aug_only_lbl,   _ = episodes_to_Xy(aug_episodes) if aug_episodes else (np.zeros((0,1)), [], [])
    X_test,       y_test_lbl,       _ = episodes_to_Xy(real_test)

    # Ensure consistent feature dimensionality across splits
    # (biotech=6DOF, acme=7DOF → different feat dims; pad to max)
    max_d = max(X_aug_train.shape[1], X_test.shape[1])

    def _pad_to(X, d):
        if X.shape[1] < d:
            X = np.hstack([X, np.zeros((X.shape[0], d - X.shape[1]), dtype=X.dtype)])
        return X

    X_aug_train  = _pad_to(X_aug_train, max_d)
    X_real_train = _pad_to(X_real_train, max_d)
    X_test       = _pad_to(X_test, max_d)
    if len(X_aug_only) > 0:
        X_aug_only = _pad_to(X_aug_only, max_d)

    print(f"  Feature dim: {max_d}")
    print(f"  Aug train  : X{X_aug_train.shape}  classes={len(set(y_aug_train_lbl))}")
    print(f"  Real train : X{X_real_train.shape}  classes={len(set(y_real_train_lbl))}")
    print(f"  Test       : X{X_test.shape}  classes={len(set(y_test_lbl))}")

    # ── 5. Train + evaluate 3 configurations ─────────────────────────────────
    print("\n[5/6] Training & evaluating three configurations on fixed test set...")

    all_results = {}

    # Config A: real + augmented (primary)
    print("\n  CONFIG A: real_aug (real train + augmented synthetic)")
    res_a = train_and_eval(X_aug_train, y_aug_train_lbl,
                            X_test, y_test_lbl, "real_aug")
    all_results["real_aug"] = _strip_models(res_a)

    # Config B: real only (baseline comparison)
    print("\n  CONFIG B: real_only (no augmentation)")
    res_b = train_and_eval(X_real_train, y_real_train_lbl,
                            X_test, y_test_lbl, "real_only")
    all_results["real_only"] = _strip_models(res_b)

    # Config C: augmented only (synthetic-only ablation)
    if len(X_aug_only) > 0 and len(set(y_aug_only_lbl)) > 1:
        print("\n  CONFIG C: aug_only (synthetic data only, no real train)")
        res_c = train_and_eval(X_aug_only, y_aug_only_lbl,
                                X_test, y_test_lbl, "aug_only")
        all_results["aug_only"] = _strip_models(res_c)

    # ── 6. Save best model ────────────────────────────────────────────────────
    print("\n[6/6] Saving best model (real_aug config, RandomForest)...")
    best_rf   = res_a["_rf_model"]
    best_norm = res_a["_rf_norm"]
    best_le   = res_a["_rf_le"]

    joblib.dump(best_rf,   MODEL_DIR / "best_rf_model.joblib")
    joblib.dump(best_norm, MODEL_DIR / "best_normalizer.joblib")
    joblib.dump(best_le,   MODEL_DIR / "label_encoder.joblib")
    print(f"  Saved: {MODEL_DIR}/best_rf_model.joblib")
    print(f"  Saved: {MODEL_DIR}/best_normalizer.joblib")
    print(f"  Saved: {MODEL_DIR}/label_encoder.joblib")

    # Also save HGB
    joblib.dump(res_a["_hgb_model"], MODEL_DIR / "best_hgb_model.joblib")
    print(f"  Saved: {MODEL_DIR}/best_hgb_model.joblib")

    # ── Write summary JSON ────────────────────────────────────────────────────
    summary = {
        "target_per_class": target_per_class,
        "n_real_episodes": len(all_real),
        "n_augmented_episodes": len(aug_episodes),
        "n_total_train_aug": len(aug_train),
        "n_test": len(real_test),
        "feature_dim": max_d,
        "augmented_label_dist": dict(aug_label_dist),
        "configurations": all_results,
    }

    out_path = OUT_DIR / "large_scale_train_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {out_path}")

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON — evaluated on real held-out test set")
    print("=" * 70)
    print(f"{'Config':<15} {'Model':<22} {'Multi-F1':>9} {'Bin-F1':>8} {'Acc':>7} {'ROC-AUC':>9}")
    print("-" * 70)
    for cfg, res in all_results.items():
        for model_name in ["RandomForest", "HistGradientBoosting", "IsolationForest"]:
            m = res.get(model_name, {})
            if not m:
                continue
            mf1 = m.get("multiclass_macro_f1", m.get("binary_macro_f1", "–"))
            bf1 = m.get("binary_macro_f1", "–")
            acc = m.get("accuracy", "–")
            auc = m.get("roc_auc", "–")
            print(f"  {cfg:<13} {model_name:<22} {str(mf1):>9} {str(bf1):>8} {str(acc):>7} {str(auc):>9}")
    print("=" * 70)

    return summary


def _strip_models(r: dict) -> dict:
    """Remove non-serialisable model objects before JSON dump."""
    return {k: v for k, v in r.items() if not k.startswith("_")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500,
                        help="Target episodes per failure class after augmentation")
    args = parser.parse_args()
    main(target_per_class=args.target)
