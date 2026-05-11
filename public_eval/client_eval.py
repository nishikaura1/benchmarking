"""
public_eval/client_eval.py
===========================
Evaluate the trained pipeline on real previously-labeled client episodes.

Datasets:
  - benchmark_output/biotech_client_episodes.json  (188 eps, 6-DOF, 8 classes)
  - benchmark_output/acme_client_episodes.json     (85 eps, 7-DOF, 7 classes)

Evaluations:
  1. In-distribution CV  — 5-fold stratified, each dataset separately
  2. Cross-client transfer — train biotech → test acme, and vice versa
  3. Physics audit — PhysicsPreFilter precision/recall on both datasets
  4. Per-class F1 breakdown (multi-class)
  5. Binary failure-detection metrics (nominal vs any-failure)

Output: benchmark_output/client_eval/client_eval_summary.json
"""

import sys
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from public_eval.physics_normalizer import (
    RobotDataNormalizer,
    PhysicsPreFilter,
    extract_physics_features,
    episode_to_physics_features,
)
from public_eval.benchmark_runner import extract_episode_features, build_feature_matrix

OUT_DIR = Path("benchmark_output/client_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_PHYSICS_FILTER = PhysicsPreFilter()


# ─────────────────────────────────────────────────────────────────────────────
# Load client JSON files → common episode schema
# ─────────────────────────────────────────────────────────────────────────────

def load_client_json(path: Path, dataset_name: str) -> list[dict]:
    """Convert client episode JSON → common episode schema dicts."""
    with open(path) as f:
        raw = json.load(f)

    episodes = []
    for item in raw:
        seq = np.array(item["seq"], dtype=np.float32)   # (T, D)
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)

        ep = {
            "dataset_name": dataset_name,
            "episode_id": item["id"],
            "timesteps": len(seq),
            "state_seq": seq,
            "action_seq": None,
            "video_frames": None,
            "image_paths": None,
            "language_task": None,
            "episode_label": item["true_label"],
            "step_labels": None,
            "semantic_labels": None,
            "failure_category": None if item["true_label"] == "nominal" else item["true_label"],
            "source_label_type": "human",
            "metadata": {"note": item.get("note", "")},
        }
        episodes.append(ep)
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def binary_labels(labels: list[str]) -> np.ndarray:
    """nominal → 0, any failure → 1."""
    return np.array([0 if lbl == "nominal" else 1 for lbl in labels], dtype=int)


def per_class_f1(y_true, y_pred, classes):
    """Return dict of {class: f1}."""
    report = classification_report(y_true, y_pred, labels=classes,
                                   zero_division=0, output_dict=True)
    # classification_report keys numeric labels as strings
    return {cls: round(report.get(str(cls), report.get(cls, {})).get("f1-score", 0.0), 4)
            for cls in classes}


def safe_roc_auc(y_true, y_score):
    try:
        return round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception:
        return None


def align_features(X_src, X_tgt):
    """Truncate to shared feature dim for cross-dataset evaluation."""
    d = min(X_src.shape[1], X_tgt.shape[1])
    return X_src[:, :d], X_tgt[:, :d]


# ─────────────────────────────────────────────────────────────────────────────
# In-distribution CV evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_indist(episodes: list[dict], dataset_name: str) -> dict:
    """5-fold stratified CV, binary + multiclass metrics."""
    X, labels, _ = build_feature_matrix(episodes)
    if len(X) == 0:
        return {"error": "no usable features"}

    y_bin = binary_labels(labels)
    classes = sorted(set(labels))
    le = LabelEncoder().fit(classes)
    y_multi = le.transform(labels)

    n_splits = min(5, int(min(Counter(labels).values())))
    n_splits = max(2, n_splits)

    # ── IsolationForest (binary, fit on nominal only) ────────────────────────
    norm_if = RobotDataNormalizer()
    X_norm = norm_if.fit_transform(X)
    nominal_mask = (y_bin == 0)
    if nominal_mask.sum() >= 5:
        iso = IsolationForest(n_estimators=200, contamination=0.25,
                              random_state=42, n_jobs=-1)
        iso.fit(X_norm[nominal_mask])
        scores = -iso.score_samples(X_norm)
        thr = np.percentile(scores[nominal_mask], 75)
        y_iso_pred = (scores > thr).astype(int)
        iso_metrics = {
            "macro_f1": round(float(f1_score(y_bin, y_iso_pred, average="macro", zero_division=0)), 4),
            "binary_f1": round(float(f1_score(y_bin, y_iso_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_bin, y_iso_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_bin, y_iso_pred, zero_division=0)), 4),
            "roc_auc": safe_roc_auc(y_bin, scores),
            "accuracy": round(float(accuracy_score(y_bin, y_iso_pred)), 4),
        }
    else:
        iso_metrics = {"error": "insufficient nominal samples"}

    # ── 5-fold CV: RandomForest + HistGB (binary and multiclass) ────────────
    def cv_eval(model_factory, X, y_bin, y_multi, le, classes, n_splits):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        bin_f1s, multi_f1s, roc_aucs, accs = [], [], [], []
        all_y_true_mc, all_y_pred_mc = [], []

        for train_idx, test_idx in skf.split(X, y_bin):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr_bin, y_te_bin = y_bin[train_idx], y_bin[test_idx]
            y_tr_mc, y_te_mc = y_multi[train_idx], y_multi[test_idx]

            norm = RobotDataNormalizer()
            X_tr_n = norm.fit_transform(X_tr)
            X_te_n = norm.transform(X_te)

            clf = model_factory()
            # Use binary if multiclass is skewed
            clf.fit(X_tr_n, y_tr_mc)
            y_pred = clf.predict(X_te_n)

            bin_pred = (y_pred != le.transform(["nominal"])[0]).astype(int)
            bin_pred_true = (y_te_mc != le.transform(["nominal"])[0]).astype(int)

            bin_f1s.append(f1_score(bin_pred_true, bin_pred, average="macro", zero_division=0))
            multi_f1s.append(f1_score(y_te_mc, y_pred, average="macro", zero_division=0))
            accs.append(accuracy_score(y_te_mc, y_pred))
            all_y_true_mc.extend(y_te_mc.tolist())
            all_y_pred_mc.extend(y_pred.tolist())

            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_te_n)
                if prob.shape[1] == 2:
                    roc_aucs.append(safe_roc_auc(bin_pred_true, prob[:, 1]) or 0)

        # Per-class F1 on pooled predictions
        per_cls = per_class_f1(all_y_true_mc, all_y_pred_mc,
                               list(range(len(classes))))
        per_cls_named = {classes[int(k)]: v for k, v in per_cls.items()}

        return {
            "binary_macro_f1": round(float(np.mean(bin_f1s)), 4),
            "multiclass_macro_f1": round(float(np.mean(multi_f1s)), 4),
            "accuracy": round(float(np.mean(accs)), 4),
            "roc_auc_binary": round(float(np.mean(roc_aucs)), 4) if roc_aucs else None,
            "cv_folds": n_splits,
            "per_class_f1": per_cls_named,
        }

    rf_results = cv_eval(
        lambda: RandomForestClassifier(n_estimators=200, max_depth=15,
                                       class_weight="balanced", random_state=42, n_jobs=-1),
        X, y_bin, y_multi, le, classes, n_splits
    )
    hgb_results = cv_eval(
        lambda: HistGradientBoostingClassifier(max_iter=200, early_stopping=True,
                                               class_weight="balanced", random_state=42),
        X, y_bin, y_multi, le, classes, n_splits
    )

    return {
        "dataset": dataset_name,
        "n_episodes": len(episodes),
        "n_usable": len(X),
        "feature_dim": X.shape[1],
        "label_distribution": dict(Counter(labels)),
        "class_list": classes,
        "n_cv_folds": n_splits,
        "IsolationForest": iso_metrics,
        "RandomForest": rf_results,
        "HistGradientBoosting": hgb_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-client transfer
# ─────────────────────────────────────────────────────────────────────────────

def eval_cross_client(src_eps, src_name, tgt_eps, tgt_name) -> dict:
    """Train RF on source, predict on entire target."""
    X_src, labels_src, _ = build_feature_matrix(src_eps)
    X_tgt, labels_tgt, _ = build_feature_matrix(tgt_eps)
    if len(X_src) == 0 or len(X_tgt) == 0:
        return {"error": "insufficient data"}

    y_src = binary_labels(labels_src)
    y_tgt = binary_labels(labels_tgt)

    X_src_a, X_tgt_a = align_features(X_src, X_tgt)
    feature_dim = X_src_a.shape[1]

    norm = RobotDataNormalizer()
    X_src_n = norm.fit_transform(X_src_a)
    X_tgt_n = norm.transform(X_tgt_a)

    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    # Train on full source in binary mode
    le_src = LabelEncoder().fit(labels_src)
    y_src_mc = le_src.transform(labels_src)
    clf.fit(X_src_n, y_src_mc)

    # Predict on target — map to binary: nominal vs failure
    le_tgt = LabelEncoder().fit(labels_tgt)
    y_pred_mc = clf.predict(X_tgt_n)

    # Convert source predictions to binary using source label encoder
    nominal_class_src = le_src.transform(["nominal"])[0] if "nominal" in le_src.classes_ else -1
    y_pred_bin = (y_pred_mc != nominal_class_src).astype(int)

    cross_macro_f1 = round(float(f1_score(y_tgt, y_pred_bin, average="macro", zero_division=0)), 4)
    cross_precision = round(float(precision_score(y_tgt, y_pred_bin, zero_division=0)), 4)
    cross_recall = round(float(recall_score(y_tgt, y_pred_bin, zero_division=0)), 4)

    # In-dist reference (on source via train/test split)
    X_tr, X_te, y_tr, y_te = train_test_split(X_src_n, y_src, test_size=0.3,
                                               random_state=42, stratify=y_src)
    clf_ref = RandomForestClassifier(n_estimators=200, max_depth=15,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    clf_ref.fit(X_tr, y_tr)
    y_te_pred = clf_ref.predict(X_te)
    indist_f1 = round(float(f1_score(y_te, y_te_pred, average="macro", zero_division=0)), 4)

    return {
        "source": src_name,
        "target": tgt_name,
        "n_source": len(X_src),
        "n_target": len(X_tgt),
        "feature_dim_used": feature_dim,
        "cross_binary_macro_f1": cross_macro_f1,
        "cross_precision": cross_precision,
        "cross_recall": cross_recall,
        "source_indist_macro_f1": indist_f1,
        "delta_f1": round(cross_macro_f1 - indist_f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Physics audit
# ─────────────────────────────────────────────────────────────────────────────

def run_physics_audit(episodes: list[dict], dataset_name: str) -> dict:
    """Run PhysicsPreFilter on all episodes, compute precision/recall vs true labels."""
    pf = PhysicsPreFilter()
    y_true_bin = binary_labels([ep["episode_label"] for ep in episodes])

    physics_preds = []
    flag_counts = Counter()

    for ep in episodes:
        try:
            phys = episode_to_physics_features(ep)
            result = pf.run_all_checks(phys)
            # physics_flags is a list of {"type":..., "timestep":...} dicts
            flags_list = result.get("physics_flags", [])
            confirmed = result.get("physics_confirmed", False)
            for flag_entry in flags_list:
                flag_counts[flag_entry["type"]] += 1
            physics_preds.append(1 if confirmed else 0)
        except Exception:
            physics_preds.append(0)

    y_phys = np.array(physics_preds, dtype=int)
    n_flagged = int(y_phys.sum())
    n_total = len(y_phys)

    prec = round(float(precision_score(y_true_bin, y_phys, zero_division=0)), 4) if n_flagged > 0 else None
    rec = round(float(recall_score(y_true_bin, y_phys, zero_division=0)), 4)

    return {
        "dataset": dataset_name,
        "n_episodes": n_total,
        "physics_flag_rate_pct": round(100 * n_flagged / n_total, 1),
        "physics_precision": prec,
        "physics_recall": rec,
        "physics_flag_breakdown": dict(flag_counts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CLIENT DATASET EVALUATION")
    print("=" * 60)

    # ── Load datasets ─────────────────────────────────────────────────────────
    bio_path = Path("benchmark_output/biotech_client_episodes.json")
    acme_path = Path("benchmark_output/acme_client_episodes.json")

    print(f"\nLoading biotech: {bio_path}")
    biotech = load_client_json(bio_path, "biotech")
    print(f"  → {len(biotech)} episodes loaded")
    print(f"  Labels: {dict(Counter(e['episode_label'] for e in biotech))}")

    print(f"\nLoading acme: {acme_path}")
    acme = load_client_json(acme_path, "acme")
    print(f"  → {len(acme)} episodes loaded")
    print(f"  Labels: {dict(Counter(e['episode_label'] for e in acme))}")

    results = {
        "eval_type": "real_client_labeled_data",
        "datasets": {},
        "cross_client_transfer": [],
        "physics_audit": [],
    }

    # ── In-distribution CV ────────────────────────────────────────────────────
    print("\n── In-distribution CV: biotech ──")
    bio_indist = eval_indist(biotech, "biotech")
    results["datasets"]["biotech"] = bio_indist
    _print_indist(bio_indist)

    print("\n── In-distribution CV: acme ──")
    acme_indist = eval_indist(acme, "acme")
    results["datasets"]["acme"] = acme_indist
    _print_indist(acme_indist)

    # ── Cross-client transfer ─────────────────────────────────────────────────
    print("\n── Cross-client transfer: biotech → acme ──")
    xfer_bio_acme = eval_cross_client(biotech, "biotech", acme, "acme")
    results["cross_client_transfer"].append(xfer_bio_acme)
    _print_xfer(xfer_bio_acme)

    print("\n── Cross-client transfer: acme → biotech ──")
    xfer_acme_bio = eval_cross_client(acme, "acme", biotech, "biotech")
    results["cross_client_transfer"].append(xfer_acme_bio)
    _print_xfer(xfer_acme_bio)

    # ── Physics audit ─────────────────────────────────────────────────────────
    print("\n── Physics audit: biotech ──")
    phys_bio = run_physics_audit(biotech, "biotech")
    results["physics_audit"].append(phys_bio)
    _print_physics(phys_bio)

    print("\n── Physics audit: acme ──")
    phys_acme = run_physics_audit(acme, "acme")
    results["physics_audit"].append(phys_acme)
    _print_physics(phys_acme)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = OUT_DIR / "client_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_indist(r):
    print(f"  Dataset: {r['dataset']}  n={r['n_usable']}  feat_dim={r['feature_dim']}")
    print(f"  Classes: {r['class_list']}")
    iso = r.get("IsolationForest", {})
    rf  = r.get("RandomForest", {})
    hgb = r.get("HistGradientBoosting", {})
    print(f"  IsolationForest  — binary macro-F1={iso.get('macro_f1','?')}  ROC-AUC={iso.get('roc_auc','?')}")
    print(f"  RandomForest     — bin macro-F1={rf.get('binary_macro_f1','?')}  multi macro-F1={rf.get('multiclass_macro_f1','?')}  acc={rf.get('accuracy','?')}")
    print(f"  HistGradBoost    — bin macro-F1={hgb.get('binary_macro_f1','?')}  multi macro-F1={hgb.get('multiclass_macro_f1','?')}  acc={hgb.get('accuracy','?')}")
    print(f"  Per-class F1 (RF): {rf.get('per_class_f1','?')}")

def _print_xfer(r):
    print(f"  {r['source']} → {r['target']}  (feat_dim={r.get('feature_dim_used','?')})")
    print(f"  Cross binary macro-F1={r.get('cross_binary_macro_f1','?')}  prec={r.get('cross_precision','?')}  rec={r.get('cross_recall','?')}")
    print(f"  Source in-dist F1={r.get('source_indist_macro_f1','?')}  Δ={r.get('delta_f1','?')}")

def _print_physics(r):
    print(f"  Dataset: {r['dataset']}  flag_rate={r['physics_flag_rate_pct']}%  prec={r['physics_precision']}  recall={r['physics_recall']}")
    print(f"  Breakdown: {r['physics_flag_breakdown']}")


if __name__ == "__main__":
    main()
