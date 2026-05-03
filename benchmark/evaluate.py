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


# ── Cross-dataset evaluation ──────────────────────────────────────────────────

def evaluate_cross_dataset(trained_rf,
                           trained_scaler,
                           trained_classes: list,
                           train_datasets: list,
                           held_out_dataset: str = "lerobot/aloha_sim_insertion_human",
                           n_per_class: int = 100,
                           seed: int = 42) -> dict:
    """
    Cross-dataset validation.

    Generates synthetic failures from a dataset that was NOT in the training
    pool, then runs the trained RF on it to measure true generalisation.

    held_out_dataset must be different from every dataset in train_datasets.
    Returns a dict with cross_macro_f1, cross_accuracy, and the gap.
    """
    from benchmark.failure_injector import generate_benchmark
    import tempfile

    train_str = " + ".join(d.split("/")[-1] for d in train_datasets)
    ood_short = held_out_dataset.split("/")[-1]

    print(f"\n{'='*58}")
    print(f"  CROSS-DATASET EVALUATION")
    print(f"  Train: {train_str}")
    print(f"  OOD  : {ood_short}")
    print(f"{'='*58}")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            generate_benchmark(
                n_per_class=n_per_class,
                dataset_name=held_out_dataset,
                output_dir=tmp,
                seed=seed,
            )
            ood_test = pd.read_parquet(f"{tmp}/test.parquet")
        except Exception as e:
            print(f"  ⚠️  Could not load {held_out_dataset}: {e}")
            print(f"  Skipping cross-dataset evaluation.")
            return {}

    print(f"  OOD test set: {len(ood_test):,} episodes from {ood_short}")

    X_ood      = np.array([episode_to_haptal_features(r) for _, r in ood_test.iterrows()])
    y_ood      = ood_test["failure_class"].values
    X_ood_sc   = trained_scaler.transform(X_ood)
    y_ood_pred = trained_rf.predict(X_ood_sc)

    cross_macro_f1 = round(float(
        f1_score(y_ood, y_ood_pred, average="macro",
                 labels=trained_classes, zero_division=0)
    ), 4)
    cross_accuracy = round(float((y_ood_pred == y_ood).mean()), 4)

    print(f"\n  Cross-dataset macro F1 : {cross_macro_f1:.4f}")
    print(f"  Cross-dataset accuracy : {cross_accuracy:.4f}")

    return {
        "train_datasets":   train_datasets,
        "held_out_dataset": held_out_dataset,
        "n_ood_episodes":   len(ood_test),
        "cross_macro_f1":   cross_macro_f1,
        "cross_accuracy":   cross_accuracy,
    }


# Datasets used for multi-dataset training (held-out OOD = aloha_sim_insertion)
TRAIN_DATASETS = [
    "lerobot/pusht",
    "lerobot/xarm_lift_medium_replay",
    "lerobot/xarm_push_medium_replay",
    "lerobot/aloha_sim_transfer_cube_human",
]
OOD_DATASET = "lerobot/aloha_sim_insertion_human"


def evaluate_benchmark(test_path: str = None,
                       save: bool = True,
                       cross_dataset: bool = True,
                       multi_dataset: bool = True,
                       n_per_class: int = 500) -> dict:
    """
    Evaluate Haptal features on the failure benchmark.

    Approach:
      1. If multi_dataset=True, regenerate train/test from TRAIN_DATASETS
         (pusht + xArm lift + xArm push + ALOHA transfer) so the RF sees
         diverse robot kinematics and generalises cross-platform.
      2. Extract 68-dim Haptal step features → aggregate to 204-dim per episode
      3. Train an episode-level RF on the multi-dataset train split
      4. Evaluate on the in-distribution test split
      5. Evaluate on aloha_sim_insertion (fully held-out OOD dataset)
      6. Report per-class F1, macro F1, accuracy, Cohen's Kappa, and OOD gap
    """
    from benchmark.failure_injector import generate_benchmark

    if multi_dataset:
        print(f"\nRegenerating multi-dataset benchmark ({len(TRAIN_DATASETS)} datasets)...")
        generate_benchmark(
            n_per_class=n_per_class,
            dataset_names=TRAIN_DATASETS,
            output_dir=str(DATA_DIR),
            seed=42,
        )

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
    print(f"  HAPTAL BENCHMARK RESULTS  (in-distribution)")
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

    # ── Cross-dataset evaluation ──────────────────────────────────────────────
    ood_results = {}
    if cross_dataset:
        ood_results = evaluate_cross_dataset(
            trained_rf=rf,
            trained_scaler=scaler,
            trained_classes=classes,
            train_datasets=TRAIN_DATASETS if multi_dataset else ["lerobot/pusht"],
            held_out_dataset=OOD_DATASET,
        )
        if ood_results:
            ood_f1  = ood_results["cross_macro_f1"]
            ood_acc = ood_results["cross_accuracy"]
            gap_f1  = round(macro_f1 - ood_f1, 4)
            gap_acc = round(accuracy - ood_acc, 4)
            ood_results["generalisation_gap_macro_f1"] = gap_f1
            ood_results["generalisation_gap_accuracy"]  = gap_acc

            print(f"\n  {'─'*58}")
            print(f"  GENERALISATION GAP  (in-dist vs OOD)")
            print(f"  {'─'*58}")
            print(f"  In-dist  macro F1 : {macro_f1:.4f}   accuracy: {accuracy:.4f}")
            print(f"  OOD      macro F1 : {ood_f1:.4f}   accuracy: {ood_acc:.4f}")
            flag = "✅" if gap_f1 <= 0.15 else "⚠️ "
            print(f"  {flag} Gap (↓ better) : {gap_f1:.4f}  "
                  f"({'≤15% — good generalisation' if gap_f1 <= 0.15 else '>15% — overfitting to train distribution'})")
            print(f"  {'─'*58}\n")

    # Paper cite line
    paper_cite = (
        f"Using Haptal's physics-informed feature extraction, an episode-level "
        f"RandomForest achieves {macro_f1:.2f} macro-F1 and {accuracy:.1%} accuracy "
        f"on the Robotics Failure Benchmark v1.1 ({len(test)} held-out episodes, "
        f"{len(classes)} failure classes, Cohen's κ={kappa:.2f})."
    )
    if ood_results:
        paper_cite += (
            f" Cross-dataset macro-F1: {ood_results['cross_macro_f1']:.2f} "
            f"(generalisation gap: {ood_results.get('generalisation_gap_macro_f1', 'N/A'):.2f})."
        )

    results = {
        "model":              model_name,
        "train_episodes":     len(train),
        "test_episodes":      len(test),
        "classes":            classes,
        "accuracy":           accuracy,
        "macro_f1":           macro_f1,
        "weighted_f1":        weighted_f1,
        "cohen_kappa":        kappa,
        "mean_confidence":    mean_conf,
        "review_rate_pct":    review_rate,
        "per_class":          {c: {"f1": per_class_f1.get(c,0),
                                   "precision": per_class_prec.get(c,0),
                                   "recall": per_class_rec.get(c,0)}
                               for c in classes},
        "cross_dataset":      ood_results,
        "paper_cite":         paper_cite,
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
    parser.add_argument("--test-path",         type=str,  default=None)
    parser.add_argument("--no-save",           action="store_true")
    parser.add_argument("--no-cross-dataset",  action="store_true",
                        help="Skip cross-dataset OOD evaluation")
    parser.add_argument("--no-multi-dataset",  action="store_true",
                        help="Use single-dataset train split (faster, less accurate)")
    parser.add_argument("--n-per-class",       type=int,  default=500,
                        help="Episodes per class when regenerating (default: 500)")
    args = parser.parse_args()

    evaluate_benchmark(
        test_path=args.test_path,
        save=not args.no_save,
        cross_dataset=not args.no_cross_dataset,
        multi_dataset=not args.no_multi_dataset,
        n_per_class=args.n_per_class,
    )
