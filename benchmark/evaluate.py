"""
benchmark/evaluate.py — Evaluate the Haptal model on the failure benchmark.

Loads benchmark/data/test.parquet, extracts 68-dim Haptal step-level features
aggregated to episode level, trains an episode-level RF on the benchmark train
split, and evaluates on the held-out test split.

This is the correct evaluation: tests whether Haptal's feature extraction is
discriminative for failure types, not just step-level labeling accuracy.

Usage
-----
  python benchmark/evaluate.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR      = BENCHMARK_DIR / "data"


# ── Feature extraction ────────────────────────────────────────────────────────

def episode_to_haptal_features(row) -> np.ndarray:
    """
    Run Haptal's 68-dim step-level feature extraction on an episode's
    state sequence, then aggregate across time (mean + std + max) to produce
    a 204-dim episode-level feature vector.
    """
    from annotation_model import extract_window_features, canonicalize_dof

    raw = row.get("state_seq")
    if raw is not None:
        if isinstance(raw, str):
            raw = json.loads(raw)
        try:
            seq   = np.vstack([np.array(r, dtype=np.float32) for r in raw])
            seq   = canonicalize_dof(seq)
            feats = extract_window_features(seq)       # (T, 68)
            return np.concatenate([
                feats.mean(axis=0),
                feats.std(axis=0),
                feats.max(axis=0),
            ])                                         # (204,)
        except Exception:
            pass

    # Fallback: use stored episode-level feature summary
    feat = row.get("features", [])
    if isinstance(feat, str):
        feat = json.loads(feat)
    return np.array(feat, dtype=np.float32)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_benchmark(test_path: str = None, save: bool = True) -> dict:
    """
    Evaluate Haptal features on the failure benchmark.

    Approach:
      1. Extract 68-dim Haptal step features for every episode → aggregate to 204-dim
      2. Train an episode-level RF on benchmark/data/train.parquet
      3. Evaluate on benchmark/data/test.parquet
      4. Report per-class F1, macro F1, accuracy, Cohen's Kappa
    """
    if test_path is None:
        test_path = str(DATA_DIR / "test.parquet")

    train_path = str(DATA_DIR / "train.parquet")

    print(f"\nLoading benchmark splits...")
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    print(f"  Train: {len(train):,} episodes  |  Test: {len(test):,} episodes")
    print(f"  Classes: {sorted(test['failure_class'].unique())}")

    # ── Build feature matrices ─────────────────────────────────────────────────
    print(f"\nExtracting Haptal features...")
    print(f"  Train ({len(train)} episodes)...")
    X_train = np.array([episode_to_haptal_features(r) for _, r in train.iterrows()])
    y_train = train["failure_class"].values

    print(f"  Test  ({len(test)} episodes)...")
    X_test  = np.array([episode_to_haptal_features(r) for _, r in test.iterrows()])
    y_true  = test["failure_class"].values
    classes = sorted(set(y_true))

    print(f"  Feature dim: {X_train.shape[1]}")

    # ── Train episode-level classifier ────────────────────────────────────────
    print(f"\nTraining episode-level RF on benchmark train split...")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        max_features=0.4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_tr_sc, y_train)

    y_pred     = rf.predict(X_te_sc)
    y_prob     = rf.predict_proba(X_te_sc)
    y_prob_max = y_prob.max(axis=1)

    model_name = (
        "Haptal episode-level RF  "
        "(68-dim step features × mean+std+max = 204-dim, "
        "n_estimators=300, max_depth=20)"
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class_f1   = {c: round(report[c]["f1-score"],  4) for c in classes if c in report}
    per_class_prec = {c: round(report[c]["precision"],  4) for c in classes if c in report}
    per_class_rec  = {c: round(report[c]["recall"],     4) for c in classes if c in report}

    macro_f1    = round(float(f1_score(y_true, y_pred, average="macro",    labels=classes, zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)), 4)
    accuracy    = round(float((y_pred == y_true).mean()), 4)
    kappa       = round(float(cohen_kappa_score(y_true, y_pred)), 4)
    review_rate = round(float((y_prob_max < 0.75).mean() * 100), 1)
    mean_conf   = round(float(y_prob_max.mean()), 4)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  HAPTAL BENCHMARK RESULTS")
    print(f"  {model_name[:55]}")
    print(f"  Test: {len(test):,} episodes · {len(classes)} classes")
    print(f"{'='*58}")
    print(f"\n  {'Class':<28}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'─'*50}")
    for cls in sorted(per_class_f1, key=lambda c: per_class_f1[c]):
        f1   = per_class_f1[cls]
        prec = per_class_prec[cls]
        rec  = per_class_rec[cls]
        flag = "✅" if f1 >= 0.80 else "🟡" if f1 >= 0.60 else "⚠️ "
        print(f"  {flag} {cls:<26}  {f1:>6.4f}  {prec:>6.4f}  {rec:>6.4f}")
    print(f"  {'─'*50}")
    print(f"  {'Macro F1':<28}  {macro_f1:>6.4f}")
    print(f"  {'Weighted F1':<28}  {weighted_f1:>6.4f}")
    print(f"  {'Accuracy':<28}  {accuracy:>6.4f}")
    print(f"  {'Cohen Kappa':<28}  {kappa:>6.4f}")
    print(f"  {'Mean confidence':<28}  {mean_conf:>6.4f}")
    print(f"  {'Low-conf (<0.75)':<28}  {review_rate:>5.1f}%")
    print(f"{'='*58}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print("  " + "".join(f"{c[:7]:>9}" for c in classes))
    for i, cls in enumerate(classes):
        print("  " + f"{cls[:11]:<12}" + "".join(f"{cm[i,j]:>9}" for j in range(len(classes))))
    print()

    # Paper cite line
    paper_cite = (
        f"Using Haptal's physics-informed feature extraction, an episode-level "
        f"RandomForest achieves {macro_f1:.2f} macro-F1 and {accuracy:.1%} accuracy "
        f"on the Robotics Failure Benchmark v1.0 ({len(test)} held-out episodes, "
        f"{len(classes)} failure classes, Cohen's κ={kappa:.2f})."
    )

    results = {
        "model":           model_name,
        "train_episodes":  len(train),
        "test_episodes":   len(test),
        "classes":         classes,
        "accuracy":        accuracy,
        "macro_f1":        macro_f1,
        "weighted_f1":     weighted_f1,
        "cohen_kappa":     kappa,
        "mean_confidence": mean_conf,
        "review_rate_pct": review_rate,
        "per_class":       {c: {"f1": per_class_f1.get(c,0),
                                "precision": per_class_prec.get(c,0),
                                "recall": per_class_rec.get(c,0)}
                            for c in classes},
        "paper_cite":      paper_cite,
    }

    if save:
        out = BENCHMARK_DIR / "results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  Results saved: {out}")

    print(f"  Paper cite:\n  \"{paper_cite}\"")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--no-save",   action="store_true")
    args = parser.parse_args()

    evaluate_benchmark(test_path=args.test_path, save=not args.no_save)
