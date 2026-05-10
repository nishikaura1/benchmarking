"""
public_eval/benchmark_runner.py
=================================
Benchmark runner for public robotics datasets.

Trains and evaluates 3 models per dataset:
  M1: IsolationForest  (episode-level anomaly baseline)
  M2: RandomForest     (classifier, episode-level)
  M3: HistGradientBoosting (classifier, episode-level)

Computes all metrics defined in PRODUCT_TRAINING_PLAN.md §7.
Saves results as JSON under benchmark_output/public_dataset_eval/{dataset}/.

Usage:
    python public_eval/benchmark_runner.py
    python public_eval/benchmark_runner.py --max-episodes 200
"""

import sys
import json
import time
import warnings
import argparse
import traceback
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    cohen_kappa_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from public_eval.dataset_loaders import load_all_datasets, dataset_eda

OUT_ROOT = Path("benchmark_output/public_dataset_eval")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_episode_features(ep: dict) -> np.ndarray | None:
    """
    Extract a fixed-size episode-level feature vector from state_seq.
    Strategy: sliding-window step features → mean + std + max → episode vector.
    Falls back to raw mean+std+max of states if window too short.
    Returns None if state_seq is unusable.
    """
    states = ep["state_seq"]  # (T, D)
    if states is None or states.size == 0:
        return None

    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    T, D = states.shape

    # Pad single-step episodes
    if T < 2:
        pad = np.zeros((4, D), dtype=np.float32)
        states = np.vstack([states, pad])
        T = states.shape[0]

    # Step-level features: position deltas, velocities, accelerations
    deltas = np.diff(states, axis=0)           # (T-1, D) velocity proxy
    accels = np.diff(deltas, axis=0) if T > 2 else np.zeros((1, D))  # (T-2, D)

    # Also include raw states, squared states (energy proxy)
    sq = states ** 2

    # Assemble step features across the sequence
    all_step_feats = []
    for arr in [states, deltas if len(deltas) > 0 else np.zeros((1, D)),
                accels if len(accels) > 0 else np.zeros((1, D)), sq]:
        mn   = arr.mean(axis=0)
        sd   = arr.std(axis=0) + 1e-8
        mx   = np.abs(arr).max(axis=0)
        all_step_feats.extend([mn, sd, mx])

    feat = np.concatenate(all_step_feats, axis=0).astype(np.float32)

    # If actions available, append action features
    if ep["action_seq"] is not None:
        acts = np.asarray(ep["action_seq"], dtype=np.float32)
        if acts.ndim == 1:
            acts = acts.reshape(-1, 1)
        if len(acts) > 0:
            act_feats = np.concatenate([acts.mean(0), acts.std(0) + 1e-8, np.abs(acts).max(0)])
            feat = np.concatenate([feat, act_feats])

    return feat


def build_feature_matrix(episodes: list[dict]) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Build feature matrix X, list of labels, and valid episode indices.
    Returns (X, labels, valid_indices).
    Skips episodes with unusable features.
    """
    X_rows, labels, valid_idx = [], [], []
    for i, ep in enumerate(episodes):
        feat = extract_episode_features(ep)
        if feat is None or not np.isfinite(feat).all():
            continue
        X_rows.append(feat)
        labels.append(ep["episode_label"] or "unknown")
        valid_idx.append(i)

    if not X_rows:
        return np.zeros((0, 1)), [], []

    # Pad to same length
    max_len = max(len(r) for r in X_rows)
    X_padded = np.zeros((len(X_rows), max_len), dtype=np.float32)
    for i, row in enumerate(X_rows):
        X_padded[i, :len(row)] = row

    return X_padded, labels, valid_idx


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_score: np.ndarray | None,
                           label_type: str, pos_label=1) -> dict:
    """Compute full binary classification metrics."""
    metrics = {
        "label_type": label_type,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "class_balance": round(float(y_true.mean()), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        metrics["false_positive_rate"] = round(float(fp / (fp + tn)) if (fp + tn) else 0, 4)
        metrics["detection_rate"] = round(float(tp / (tp + fn)) if (tp + fn) else 0, 4)
    else:
        metrics["confusion_matrix"] = cm.tolist()

    if y_score is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
        except Exception:
            metrics["pr_auc"] = None
        try:
            if len(y_score) > 10:
                brier = brier_score_loss(y_true, np.clip(y_score, 0, 1))
                metrics["brier_score"] = round(float(brier), 4)
        except Exception:
            pass

    return metrics


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray | None,
                               classes: list[str], label_type: str) -> dict:
    """Compute full multiclass classification metrics."""
    metrics = {
        "label_type": label_type,
        "n_samples": int(len(y_true)),
        "n_classes": len(classes),
        "class_distribution": {c: int((y_true == i).sum()) for i, c in enumerate(classes)},
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class_f1": {},
    }

    try:
        metrics["cohen_kappa"] = round(float(cohen_kappa_score(y_true, y_pred)), 4)
    except Exception:
        pass

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, c in enumerate(classes):
        if i < len(per_class_f1):
            metrics["per_class_f1"][c] = {
                "f1":        round(float(per_class_f1[i]), 4),
                "precision": round(float(per_class_prec[i]), 4),
                "recall":    round(float(per_class_rec[i]), 4),
                "support":   int((y_true == i).sum()),
            }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    if y_proba is not None and y_proba.shape[1] == len(classes):
        try:
            metrics["roc_auc_ovr"] = round(float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 4)
        except Exception:
            pass

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: IsolationForest (episode-level anomaly)
# ─────────────────────────────────────────────────────────────────────────────

def run_isolation_forest(
    X: np.ndarray,
    labels: list[str],
    label_type: str,
    review_rate_targets: list[float] = [0.10, 0.20, 0.30],
) -> dict:
    """
    Train IsolationForest on nominal episodes, score all episodes.
    If no nominal/failure split: train on all, use quantile threshold.
    """
    result = {"model": "IsolationForest", "label_type": label_type}

    binary_labels = np.array([0 if l in ("nominal", "normal", "success") else 1 for l in labels])
    n_nominal = int((binary_labels == 0).sum())
    n_failure = int((binary_labels == 1).sum())
    result["n_nominal"] = n_nominal
    result["n_failure"] = n_failure

    # Fit on nominal only if we have enough
    if n_nominal >= 5:
        X_train = X[binary_labels == 0]
        fit_desc = "nominal_only"
    else:
        X_train = X
        fit_desc = "all_episodes"
    result["fit_on"] = fit_desc

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_all_sc   = scaler.transform(X)

    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(X_train_sc)

    scores = -clf.score_samples(X_all_sc)  # higher = more anomalous

    # Normalise to [0,1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = scores * 0.0
    result["anomaly_score_mean"] = round(float(scores_norm.mean()), 4)
    result["anomaly_score_std"]  = round(float(scores_norm.std()), 4)

    # Evaluate if we have binary labels
    if n_nominal >= 2 and n_failure >= 2:
        # Use 75th percentile of training scores as threshold
        train_scores = scores_norm[binary_labels == 0] if n_nominal >= 5 else scores_norm
        tau = float(np.quantile(train_scores, 0.75))
        preds = (scores_norm >= tau).astype(int)
        result["threshold_75pct"] = round(tau, 4)
        result["metrics_75pct"] = compute_binary_metrics(binary_labels, preds, scores_norm, label_type)

        # Sweep review rate targets
        review_metrics = {}
        for rrt in review_rate_targets:
            tau_rrt = float(np.quantile(scores_norm, 1 - rrt))
            preds_rrt = (scores_norm >= tau_rrt).astype(int)
            review_metrics[f"review_rate_{int(rrt*100)}pct"] = {
                "threshold": round(tau_rrt, 4),
                "review_rate_actual": round(float(preds_rrt.mean()), 3),
                "precision": round(float(precision_score(binary_labels, preds_rrt, zero_division=0)), 4),
                "recall": round(float(recall_score(binary_labels, preds_rrt, zero_division=0)), 4),
            }
        result["review_rate_sweep"] = review_metrics
    else:
        result["metrics_75pct"] = None
        result["note"] = (
            f"Insufficient class balance for metric computation "
            f"(n_nominal={n_nominal}, n_failure={n_failure}). "
            "Anomaly scores computed but no ROC-AUC available."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 & 3: RandomForest and HistGradientBoosting classifiers
# ─────────────────────────────────────────────────────────────────────────────

def run_classifier(
    X: np.ndarray,
    labels: list[str],
    label_type: str,
    model_name: str,
    n_cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train and evaluate a classifier with stratified cross-validation.
    Falls back to train/test split if CV isn't viable.
    """
    result = {"model": model_name, "label_type": label_type}

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)
    result["classes"] = classes
    result["class_distribution"] = {c: int((y == i).sum()) for i, c in enumerate(classes)}

    n_samples = len(y)
    n_classes = len(classes)
    min_class = min(result["class_distribution"].values())

    result["n_samples"] = n_samples
    result["n_classes"] = n_classes

    # Normalise
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)

    # Build model
    if model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        )
    elif model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, learning_rate=0.05,
            l2_regularization=0.1, early_stopping=True,
            n_iter_no_change=10, validation_fraction=0.15,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Decide evaluation strategy
    use_cv = (n_samples >= 20 and min_class >= n_cv and n_classes >= 2)
    is_binary = (n_classes == 2)

    if use_cv:
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state)
        fold_metrics = []
        all_y_true, all_y_pred, all_y_proba = [], [], []

        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_sc, y)):
            X_tr, X_te = X_sc[tr_idx], X_sc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            clf_fold = (RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced",
                n_jobs=-1, random_state=random_state + fold_i,
            ) if model_name == "RandomForest" else HistGradientBoostingClassifier(
                max_iter=200, max_depth=8, learning_rate=0.05,
                l2_regularization=0.1, early_stopping=True,
                n_iter_no_change=10, validation_fraction=min(0.15, max(0.1, 2/len(y_tr))),
                random_state=random_state + fold_i,
            ))
            clf_fold.fit(X_tr, y_tr)
            preds = clf_fold.predict(X_te)
            proba = clf_fold.predict_proba(X_te)

            all_y_true.extend(y_te)
            all_y_pred.extend(preds)
            all_y_proba.append(proba)

        y_true_all = np.array(all_y_true)
        y_pred_all = np.array(all_y_pred)
        y_proba_all = np.vstack(all_y_proba)

        result["evaluation"] = "stratified_cv"
        result["n_folds"] = n_cv

        if is_binary:
            # Use failure class probability for ROC/PR
            failure_idx = list(classes).index("failure") if "failure" in classes else 1
            y_score = y_proba_all[:, failure_idx]
            binary_true = (y_true_all == failure_idx).astype(int)
            result["metrics"] = compute_binary_metrics(
                binary_true, (y_pred_all == failure_idx).astype(int), y_score, label_type)
        else:
            result["metrics"] = compute_multiclass_metrics(
                y_true_all, y_pred_all, y_proba_all, classes, label_type)

    elif n_samples >= 10 and n_classes >= 2 and min_class >= 2:
        # Simple train/test split
        test_size = min(0.3, max(0.1, 10 / n_samples))
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sc, y, test_size=test_size, stratify=y, random_state=random_state)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sc, y, test_size=test_size, random_state=random_state)

        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)

        result["evaluation"] = "train_test_split"
        result["train_size"] = len(y_tr)
        result["test_size"] = len(y_te)

        if is_binary:
            failure_idx = list(classes).index("failure") if "failure" in classes else 1
            y_score = proba[:, failure_idx]
            binary_true = (y_te == failure_idx).astype(int)
            result["metrics"] = compute_binary_metrics(
                binary_true, (preds == failure_idx).astype(int), y_score, label_type)
        else:
            result["metrics"] = compute_multiclass_metrics(
                y_te, preds, proba, classes, label_type)
    else:
        result["metrics"] = None
        result["note"] = (
            f"Insufficient data for evaluation: n_samples={n_samples}, "
            f"n_classes={n_classes}, min_class_count={min_class}. "
            "Model not evaluated — data/label constraints prevent split."
        )
        result["evaluation"] = "skipped"

    # Feature importance (if available and evaluation ran)
    if result.get("evaluation") not in ("skipped", None) and hasattr(clf, "feature_importances_"):
        try:
            clf.fit(X_sc, y)  # fit on all for importance
            top_k = 10
            imp = clf.feature_importances_
            top_idx = np.argsort(imp)[::-1][:top_k]
            result["top_feature_importances"] = {
                f"feature_{i}": round(float(imp[i]), 5) for i in top_idx
            }
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_benchmark(
    name: str,
    episodes: list[dict],
    access_report: dict,
) -> dict:
    """Run all 3 models on one dataset. Return combined result dict."""
    print(f"\n  [{name}] Building feature matrix…")
    t0 = time.time()

    X, labels, valid_idx = build_feature_matrix(episodes)
    label_type = episodes[0]["source_label_type"] if episodes else "unknown"

    result = {
        "dataset": name,
        "label_type": label_type,
        "n_episodes_total": len(episodes),
        "n_episodes_usable": len(valid_idx),
        "feature_dim": X.shape[1] if len(X) > 0 else 0,
        "label_distribution": {},
        "eda": {},
        "model_results": {},
        "access_report": access_report,
    }

    if not episodes:
        result["skip_reason"] = "No episodes loaded"
        return result

    # EDA
    result["eda"] = dataset_eda(episodes, name)
    for lbl in labels:
        result["label_distribution"][lbl] = result["label_distribution"].get(lbl, 0) + 1

    if len(X) == 0:
        result["skip_reason"] = "Feature extraction failed for all episodes"
        return result

    n_unique_labels = len(set(labels))
    print(f"  [{name}] X={X.shape}, labels={result['label_distribution']}")

    # M1: IsolationForest
    print(f"  [{name}] Running IsolationForest…")
    try:
        result["model_results"]["IsolationForest"] = run_isolation_forest(
            X, labels, label_type)
    except Exception as e:
        result["model_results"]["IsolationForest"] = {"error": str(e), "traceback": traceback.format_exc()}

    # M2: RandomForest
    if n_unique_labels >= 2 and len(X) >= 6:
        print(f"  [{name}] Running RandomForest…")
        try:
            result["model_results"]["RandomForest"] = run_classifier(
                X, labels, label_type, "RandomForest")
        except Exception as e:
            result["model_results"]["RandomForest"] = {"error": str(e), "traceback": traceback.format_exc()}
    else:
        result["model_results"]["RandomForest"] = {
            "skipped": True,
            "reason": f"Insufficient data or labels: n_episodes={len(X)}, n_label_classes={n_unique_labels}",
        }

    # M3: HistGradientBoosting
    if n_unique_labels >= 2 and len(X) >= 6:
        print(f"  [{name}] Running HistGradientBoosting…")
        try:
            result["model_results"]["HistGradientBoosting"] = run_classifier(
                X, labels, label_type, "HistGradientBoosting")
        except Exception as e:
            result["model_results"]["HistGradientBoosting"] = {"error": str(e), "traceback": traceback.format_exc()}
    else:
        result["model_results"]["HistGradientBoosting"] = {
            "skipped": True,
            "reason": f"Insufficient data: n_episodes={len(X)}, n_label_classes={n_unique_labels}",
        }

    elapsed = time.time() - t0
    result["elapsed_seconds"] = round(elapsed, 2)
    print(f"  [{name}] Done in {elapsed:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset transfer
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_dataset_transfer(all_X: dict, all_labels: dict, all_label_types: dict) -> dict:
    """
    For pairs of datasets with binary labels, train on source and test on target.
    Reports Δmacro-F1 vs in-distribution performance.
    """
    BINARY_CLASSES = {"nominal", "normal", "success", "failure"}
    report = {"cross_dataset_pairs": []}

    # Collect datasets with enough binary-labeled episodes
    eligible = {}
    for name, (X, labels) in {n: (all_X[n], all_labels[n]) for n in all_X}.items():
        if X is None or len(X) < 10:
            continue
        uniq = set(labels)
        if len(uniq) < 2 or not (uniq & BINARY_CLASSES):
            continue
        # Binarize
        y_bin = np.array([0 if l in ("nominal", "normal", "success") else 1 for l in labels])
        if y_bin.sum() < 3 or (len(y_bin) - y_bin.sum()) < 3:
            continue
        eligible[name] = (X, y_bin)

    names = list(eligible.keys())
    for i, src in enumerate(names):
        for tgt in names[i+1:]:
            X_src, y_src = eligible[src]
            X_tgt, y_tgt = eligible[tgt]

            # Align feature dims by padding/truncating to min
            min_dim = min(X_src.shape[1], X_tgt.shape[1])
            X_src_t = X_src[:, :min_dim]
            X_tgt_t = X_tgt[:, :min_dim]

            scaler = StandardScaler()
            X_src_sc = scaler.fit_transform(X_src_t)
            X_tgt_sc = scaler.transform(X_tgt_t)

            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )
            try:
                clf.fit(X_src_sc, y_src)
                preds = clf.predict(X_tgt_sc)
                f1_cross = float(f1_score(y_tgt, preds, average="macro", zero_division=0))

                # In-distribution baseline for target (10-fold or split)
                if len(X_tgt_sc) >= 20 and y_tgt.sum() >= 2 and (len(y_tgt) - y_tgt.sum()) >= 2:
                    try:
                        Xtr, Xte, ytr, yte = train_test_split(
                            X_tgt_sc, y_tgt, test_size=0.3, stratify=y_tgt, random_state=42)
                        clf2 = RandomForestClassifier(
                            n_estimators=100, max_depth=10, class_weight="balanced",
                            n_jobs=-1, random_state=42)
                        clf2.fit(Xtr, ytr)
                        f1_indist = float(f1_score(yte, clf2.predict(Xte), average="macro", zero_division=0))
                        delta = round(f1_cross - f1_indist, 4)
                    except Exception:
                        f1_indist = None
                        delta = None
                else:
                    f1_indist = None
                    delta = None

                report["cross_dataset_pairs"].append({
                    "source": src,
                    "target": tgt,
                    "source_label_type": all_label_types.get(src, "?"),
                    "target_label_type": all_label_types.get(tgt, "?"),
                    "cross_macro_f1": round(f1_cross, 4),
                    "target_indist_macro_f1": round(f1_indist, 4) if f1_indist is not None else None,
                    "delta_f1": delta,
                    "n_source": len(X_src),
                    "n_target": len(X_tgt),
                    "feature_dim_used": min_dim,
                })
                print(f"  [Cross] {src} → {tgt}: macro-F1={f1_cross:.3f} (in-dist={f1_indist})")
            except Exception as e:
                report["cross_dataset_pairs"].append({
                    "source": src, "target": tgt, "error": str(e)})

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(max_episodes: int = 300):
    print("\n" + "="*70)
    print("  HAPTAL PUBLIC DATASET BENCHMARK RUNNER")
    print("="*70)

    # 1. Load datasets
    all_episodes, all_access_reports = load_all_datasets(
        max_episodes_per_dataset=max_episodes)

    # 2. Run per-dataset benchmarks
    all_results = {}
    all_X = {}
    all_labels_dict = {}
    all_label_types = {}

    for ds_name, episodes in all_episodes.items():
        ds_dir = OUT_ROOT / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        if not episodes:
            print(f"\n[{ds_name}] SKIPPED — no episodes loaded")
            all_results[ds_name] = {
                "dataset": ds_name,
                "skip_reason": "No episodes loaded",
                "access_report": all_access_reports.get(ds_name, {}),
            }
            # Save anyway
            with open(ds_dir / "benchmark_result.json", "w") as f:
                json.dump(all_results[ds_name], f, indent=2, default=str)
            continue

        result = run_dataset_benchmark(
            ds_name, episodes, all_access_reports.get(ds_name, {}))
        all_results[ds_name] = result

        # Save per-dataset result
        out_path = ds_dir / "benchmark_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  [{ds_name}] Saved → {out_path}")

        # Collect for cross-dataset
        X, labels, _ = build_feature_matrix(episodes)
        if len(X) > 0:
            all_X[ds_name] = X
            all_labels_dict[ds_name] = labels
            all_label_types[ds_name] = episodes[0]["source_label_type"]

    # 3. Cross-dataset transfer
    print("\n" + "="*70)
    print("  CROSS-DATASET TRANSFER")
    print("="*70)
    cross_result = run_cross_dataset_transfer(all_X, all_labels_dict, all_label_types)
    cross_path = OUT_ROOT / "cross_dataset_transfer.json"
    with open(cross_path, "w") as f:
        json.dump(cross_result, f, indent=2, default=str)
    print(f"  Saved → {cross_path}")

    # 4. Aggregate summary
    summary = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_episodes_per_dataset": max_episodes,
        "datasets": {},
        "cross_dataset_transfer": cross_result,
    }
    for ds_name, result in all_results.items():
        ar = all_access_reports.get(ds_name, {})
        ds_summary = {
            "loaded": ar.get("success", False),
            "n_episodes": ar.get("n_episodes", 0),
            "label_type": ar.get("label_type", "unknown"),
            "synthetic_fallback": ar.get("synthetic_fallback", False),
        }
        mr = result.get("model_results", {})
        for model_name, mres in mr.items():
            if mres and not mres.get("skipped") and not mres.get("error"):
                mets = mres.get("metrics") or mres.get("metrics_75pct")
                if mets:
                    ds_summary[f"{model_name}_macro_f1"] = mets.get("macro_f1")
                    ds_summary[f"{model_name}_roc_auc"]  = mets.get("roc_auc")
        summary["datasets"][ds_name] = ds_summary

    summary_path = OUT_ROOT / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved → {summary_path}")
    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE")
    print("="*70)

    return all_results, cross_result, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=300,
                        help="Max episodes per dataset (default 300)")
    args = parser.parse_args()
    main(max_episodes=args.max_episodes)
