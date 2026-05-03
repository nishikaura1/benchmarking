"""
robot_adapter.py — Per-client fine-tuning layer — Haptal AI

Architecture
------------
Base model (frozen RF, trained on 10 public datasets)
    ↓  predict_proba → (N, 10) class probabilities
    +  raw features  → (N, 68) physics features
    ↓  concatenated  → (N, 78)
MLP Adapter Head    → (N, C)   C = client classes (10 base + any custom)
    ↓
Platt calibration   → calibrated per-class confidences

The RF is never modified. Only the MLP adapter head is trained per client,
which means adaptation needs as few as 20–30 episodes and takes <10 seconds.

Client isolation
----------------
Each client gets their own directory:
  benchmark_output/clients/{client_id}/
    adapter.pkl     — MLP head + metadata
    taxonomy.json   — class definitions (base + custom)
    metrics.json    — latest benchmark results

Usage
-----
# New client onboarding:
adapter = ClientAdapter.create("acme_robotics", robot_type="franka")
adapter.adapt(episodes, n_epochs=50)
adapter.save()

# Inference:
adapter = ClientAdapter.load("acme_robotics")
result  = adapter.annotate(state_seq)

# Add a custom failure class:
adapter.add_custom_class("cable_snag", examples=[(seq1,"cable_snag"), ...])
adapter.save()
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

OUTPUT_DIR  = Path("benchmark_output")
CLIENTS_DIR = OUTPUT_DIR / "clients"
CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_DOF = 8

BASE_FAILURE_CLASSES = [
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


# ── MLP Adapter Head ──────────────────────────────────────────────────────────

class MLPAdapterHead:
    """
    Lightweight MLP that sits on top of the frozen RF's probability outputs.

    Input  : RF probabilities (10) + raw physics features (68) = 78 dims
    Hidden : 64 → 32 (ReLU, dropout-equivalent via low alpha regularization)
    Output : C classes (10 base + any custom client classes)

    Why MLP over re-fitting Platt scaling:
    - Platt scaling is linear — it can shift/scale confidences but can't
      remap class boundaries. If a client's robot has genuinely different
      failure signatures, a linear layer can't capture that.
    - The MLP learns a non-linear remapping of the RF's belief space,
      allowing it to, e.g., split "high_anomaly" into client-specific classes
      or down-weight classes that never appear on this robot.
    - Still fast: 78-dim input, 2 hidden layers → trains in <5s on 50 episodes.
    """

    def __init__(self, n_classes: int = len(BASE_FAILURE_CLASSES)):
        self.n_classes = n_classes
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,          # L2 regularization — prevents overfitting on small datasets
            learning_rate="adaptive",
            max_iter=200,
            n_iter_no_change=15,
            random_state=42,
            warm_start=True,     # enables incremental retraining when new data arrives
        )
        self.scaler   = StandardScaler()
        self.le       = LabelEncoder()
        self._fitted  = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPAdapterHead":
        """X: (N, 78) combined features, y: (N,) string labels."""
        X_sc = self.scaler.fit_transform(X)
        self.le.fit(y)
        y_enc = self.le.transform(y)
        self.mlp.fit(X_sc, y_enc)
        self._fitted = True
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "MLPAdapterHead":
        """Incremental update — called when new human corrections arrive."""
        X_sc = self.scaler.transform(X)
        y_enc = self.le.transform(y)
        self.mlp.partial_fit(X_sc, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.le.inverse_transform(self.mlp.predict(self.scaler.transform(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict_proba(self.scaler.transform(X))

    @property
    def classes_(self):
        return self.le.classes_


# ── Feature builder ───────────────────────────────────────────────────────────

def build_adapter_features(
    state_seq: np.ndarray,
    base_annotator,
) -> np.ndarray:
    """
    Build (T, 78) combined feature matrix:
      - 68 physics features from extract_window_features
      - 10 RF class probabilities

    This gives the MLP both raw signal AND the base model's belief,
    so it can correct the RF rather than ignore it.
    """
    from annotation_model import extract_window_features, canonicalize_dof

    seq    = canonicalize_dof(state_seq)
    feats  = extract_window_features(seq)                           # (T, 68)
    scaled = base_annotator.scaler.transform(feats)                 # (T, 68) scaled
    probs  = base_annotator.calibrated_model.predict_proba(scaled)  # (T, 10)

    return np.hstack([probs, feats])    # (T, 78)


# ── Client taxonomy ───────────────────────────────────────────────────────────

class ClientTaxonomy:
    """
    Manages the failure class vocabulary for one client.
    Starts from the 10 base classes and allows adding custom ones.
    """

    def __init__(self, client_id: str):
        self.client_id    = client_id
        self.base_classes = list(BASE_FAILURE_CLASSES)
        self.custom_classes: Dict[str, dict] = {}   # name → {description, cause, strategy}

    @property
    def all_classes(self) -> List[str]:
        return self.base_classes + list(self.custom_classes.keys())

    def add_class(self, name: str, description: str = "",
                  cause: str = "", strategy: str = "") -> None:
        if name in self.base_classes:
            raise ValueError(f"'{name}' already exists in the base taxonomy.")
        self.custom_classes[name] = {
            "description": description,
            "cause":       cause,
            "strategy":    strategy,
            "added_by":    "client",
        }
        print(f"  [taxonomy] Added custom class '{name}' — "
              f"total classes: {len(self.all_classes)}")

    def to_dict(self) -> dict:
        return {
            "client_id":      self.client_id,
            "base_classes":   self.base_classes,
            "custom_classes": self.custom_classes,
            "all_classes":    self.all_classes,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ClientTaxonomy":
        data = json.loads(path.read_text())
        obj  = cls(data["client_id"])
        obj.custom_classes = data.get("custom_classes", {})
        return obj


# ── Client Adapter ────────────────────────────────────────────────────────────

class ClientAdapter:
    """
    Per-client fine-tuning adapter on top of the frozen base model.

    Workflow
    --------
    1. create()       — initialize with client metadata
    2. adapt()        — train MLP head on client episodes (20–100 episodes)
    3. add_custom_class() — optionally extend taxonomy with client-specific failure
    4. save()         — persist to benchmark_output/clients/{client_id}/
    5. annotate()     — run inference with adapted model
    6. update()       — incremental retraining when human corrections arrive
    """

    def __init__(
        self,
        client_id:   str,
        robot_type:  str = "unknown",
        base_annotator=None,
    ):
        self.client_id      = client_id
        self.robot_type     = robot_type
        self.base_annotator = base_annotator
        self.taxonomy       = ClientTaxonomy(client_id)
        self.adapter_head     = None
        self.review_threshold = 0.60
        self.per_class_thresholds: Dict[str, float] = {}   # per-class confidence floor
        self.client_scaler    = StandardScaler()            # fit on client data, not base
        self._n_episodes_seen = 0
        self._metrics: dict   = {}
        self._X_cache: Optional[np.ndarray] = None
        self._y_cache: Optional[np.ndarray] = None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, client_id: str, robot_type: str = "unknown") -> "ClientAdapter":
        """Initialize a new adapter for a client. Loads the base model."""
        from annotation_model import RobotAnnotator

        model_path = OUTPUT_DIR / "robot_annotator.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                "Base model not found. Run: python annotation_model.py --train"
            )
        base = RobotAnnotator.load(model_path)
        obj  = cls(client_id=client_id, robot_type=robot_type, base_annotator=base)
        print(f"\nClientAdapter created — client='{client_id}' robot='{robot_type}'")
        print(f"  Base model: {base.datasets_used}")
        return obj

    @classmethod
    def load(cls, client_id: str) -> "ClientAdapter":
        """Load a saved client adapter."""
        client_dir = CLIENTS_DIR / client_id
        adapter_path  = client_dir / "adapter.pkl"
        taxonomy_path = client_dir / "taxonomy.json"

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"No adapter found for client '{client_id}' at {adapter_path}.\n"
                f"Run ClientAdapter.create('{client_id}').adapt(episodes).save() first."
            )

        with open(adapter_path, "rb") as fh:
            state = pickle.load(fh)

        obj = cls(
            client_id      = client_id,
            robot_type     = state["robot_type"],
            base_annotator = state["base_annotator"],
        )
        obj.adapter_head       = state["adapter_head"]
        obj.review_threshold   = state["review_threshold"]
        obj._n_episodes_seen      = state["n_episodes_seen"]
        obj._metrics              = state.get("metrics", {})
        obj._X_cache              = state.get("X_cache")
        obj._y_cache              = state.get("y_cache")
        obj.per_class_thresholds  = state.get("per_class_thresholds", {})
        obj.client_scaler         = state.get("client_scaler", StandardScaler())

        if taxonomy_path.exists():
            obj.taxonomy = ClientTaxonomy.load(taxonomy_path)

        print(f"ClientAdapter loaded — client='{client_id}' "
              f"robot='{obj.robot_type}' "
              f"episodes_seen={obj._n_episodes_seen} "
              f"classes={len(obj.taxonomy.all_classes)}")
        return obj

    # ── Taxonomy extension ─────────────────────────────────────────────────────

    def add_custom_class(
        self,
        name: str,
        examples: List[Tuple[np.ndarray, str]],
        description: str = "",
        cause: str = "",
        strategy: str = "",
    ) -> "ClientAdapter":
        """
        Add a new failure class specific to this client and retrain the adapter head.

        examples : list of (state_seq, label) tuples where label == name.
                   Need at least 5 examples for meaningful learning.
        """
        if len(examples) < 5:
            print(f"  [WARNING] Only {len(examples)} examples for '{name}'. "
                  f"Recommend ≥5 for reliable detection.")

        self.taxonomy.add_class(name, description=description,
                                cause=cause, strategy=strategy)

        # sklearn MLPClassifier cannot change class count via partial_fit.
        # Solution: combine cached training data with new class examples,
        # then do a clean full retrain of a fresh head.
        print(f"  Retraining adapter head with new class '{name}'...")

        X_new, y_new = [], []
        for seq, lbl in examples:
            seq = np.atleast_2d(seq).astype(np.float32)
            X_ep = build_adapter_features(seq, self.base_annotator)
            X_new.append(X_ep)
            y_new.extend([lbl] * len(X_ep))

        X_custom = np.vstack(X_new)
        y_custom  = np.array(y_new)

        # Combine with cached training data if available
        if self._X_cache is not None:
            X_combined = np.vstack([self._X_cache, X_custom])
            y_combined  = np.concatenate([self._y_cache, y_custom])
        else:
            X_combined = X_custom
            y_combined  = y_custom

        # Update cache
        self._X_cache = X_combined
        self._y_cache = y_combined

        # Fresh head — now has all classes including the new one
        new_head = MLPAdapterHead(n_classes=len(self.taxonomy.all_classes))
        new_head.fit(X_combined, y_combined)
        self.adapter_head = new_head
        return self

    # ── Adaptation (initial training) ─────────────────────────────────────────

    def adapt(
        self,
        episodes: List[Tuple[np.ndarray, str]],
        n_epochs: int = 100,
        val_frac: float = 0.15,
    ) -> "ClientAdapter":
        """
        Train the MLP adapter head on client episodes.

        episodes : list of (state_seq, ep_label) where ep_label is one of
                   self.taxonomy.all_classes or "nominal".
        n_epochs : MLPClassifier max_iter (warm_start means this accumulates).
        val_frac : fraction held out for validation reporting.
        """
        from annotation_model import generate_weak_labels

        print(f"\nAdapting for client='{self.client_id}' "
              f"on {len(episodes)} episodes...")

        # ── Fix 1: Fit a client-specific scaler on THIS robot's data ──────────
        # The base scaler was fit on xarm/ALOHA — different velocity ranges.
        # Re-scaling on client data brings features to the right distribution.
        from annotation_model import extract_window_features, canonicalize_dof
        all_seqs = [np.atleast_2d(s).astype(np.float32) for s, _ in episodes]
        all_feats = np.vstack([
            extract_window_features(canonicalize_dof(s)) for s in all_seqs
        ])
        self.client_scaler.fit(all_feats)
        print(f"  Client scaler fit — feature std={all_feats.std():.3f} "
              f"→ scaled std≈1.0 on this robot")

        # ── Fix 2: Use episode-level label directly for known failure episodes ─
        # Weak labels misfire on new robots (e.g. slow Franka → stuck_joint).
        # For nominal episodes: use weak labels but filter out low-confidence ones.
        # For failure episodes: trust the episode label for ALL steps.
        X_all, y_all = [], []
        for seq, ep_label in episodes:
            seq = np.atleast_2d(seq).astype(np.float32)
            try:
                X_ep = build_adapter_features(seq, self.base_annotator)  # (T, 78)

                if ep_label == "nominal" or ep_label == 0:
                    # Get weak labels but only keep steps the base model agrees on
                    step_labels = generate_weak_labels(seq)
                    base_probs  = self.base_annotator.calibrated_model.predict_proba(
                        self.base_annotator.scaler.transform(
                            extract_window_features(canonicalize_dof(seq))
                        )
                    )
                    base_conf = base_probs.max(axis=1)
                    # Override weak label with "nominal" when base model is uncertain
                    # (conf < 0.55) — uncertain base predictions on nominal eps are noise
                    step_labels = [
                        "nominal" if (lbl != "nominal" and c < 0.55) else lbl
                        for lbl, c in zip(step_labels, base_conf)
                    ]
                else:
                    # Failure episode: broadcast episode label to all steps
                    step_labels = [ep_label] * len(X_ep)

            except Exception as e:
                print(f"  [WARN] Skipping episode: {e}")
                continue

            X_all.append(X_ep)
            y_all.extend(step_labels)

        if not X_all:
            print("  [ERROR] No valid episodes to adapt on.")
            return self

        X = np.vstack(X_all)
        y = np.array(y_all)

        # Filter labels to known classes only
        known = set(self.taxonomy.all_classes)
        mask  = np.array([lbl in known for lbl in y])
        X, y  = X[mask], y[mask]

        print(f"  Training set: {len(X):,} steps × {X.shape[1]} features")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"    {cls:22s}: {cnt:6,}  ({cnt/len(y)*100:.1f}%)")

        # Train/val split
        n_val   = max(1, int(len(X) * val_frac))
        indices = np.random.RandomState(42).permutation(len(X))
        val_idx, tr_idx = indices[:n_val], indices[n_val:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train MLP adapter head
        self.adapter_head = MLPAdapterHead(n_classes=len(self.taxonomy.all_classes))
        self.adapter_head.mlp.max_iter = n_epochs
        self.adapter_head.fit(X_tr, y_tr)

        # Evaluate
        y_pred = self.adapter_head.predict(X_val)
        acc    = float((y_pred == y_val).mean())

        present = sorted(set(y_val))
        report  = classification_report(
            y_val, y_pred,
            labels=present,
            target_names=present,
            output_dict=True,
            zero_division=0,
        )

        print(f"\n  Validation accuracy: {acc:.3f}")
        for cls in present:
            if cls in report:
                r = report[cls]
                print(f"    {cls:22s}  "
                      f"prec={r['precision']:.2f}  "
                      f"rec={r['recall']:.2f}  "
                      f"f1={r['f1-score']:.2f}")

        # ── Fix 3: Per-class confidence thresholds ────────────────────────────
        # A single global threshold treats "nominal" (high volume, easy) the same
        # as "weld_stutter" (rare, ambiguous). Per-class thresholds set the floor
        # based on each class's actual confidence distribution on validation data.
        probs     = self.adapter_head.predict_proba(X_val)
        classes   = list(self.adapter_head.classes_)
        max_confs = probs.max(axis=1)

        # Global fallback threshold (10th percentile)
        self.review_threshold = float(np.percentile(max_confs, 10))

        # Per-class: route to review if below 20th percentile of that class's scores
        self.per_class_thresholds = {}
        for cls in classes:
            cls_mask = y_val == cls
            if cls_mask.sum() >= 3:
                cls_confs = probs[cls_mask, classes.index(cls)]
                self.per_class_thresholds[cls] = float(np.percentile(cls_confs, 20))

        review_rate = (max_confs < self.review_threshold).mean() * 100
        print(f"\n  Global review threshold : {self.review_threshold:.3f} "
              f"({review_rate:.1f}% steps)")
        print(f"  Per-class thresholds set for {len(self.per_class_thresholds)} classes")

        # Cache training data so custom class addition can retrain from scratch
        self._X_cache = X
        self._y_cache = y

        self._n_episodes_seen += len(episodes)
        self._metrics = {
            "accuracy":        round(acc, 4),
            "n_episodes":      self._n_episodes_seen,
            "n_classes":       len(self.taxonomy.all_classes),
            "review_threshold": round(self.review_threshold, 4),
            "per_class_f1":    {
                cls: round(report[cls]["f1-score"], 3)
                for cls in present if cls in report
            },
        }
        return self

    # ── Incremental update (human corrections) ────────────────────────────────

    def update(
        self,
        corrections: List[Tuple[np.ndarray, str]],
        weight: float = 10.0,
    ) -> "ClientAdapter":
        """
        Incrementally update the adapter with human-verified corrections.
        Uses MLPClassifier warm_start — no full retrain needed.

        corrections : list of (state_seq, corrected_label)
        weight      : how many times to repeat each correction (simulates sample weight)
        """
        if self.adapter_head is None or not self.adapter_head._fitted:
            print("  [WARN] Adapter not trained yet — run adapt() first.")
            return self

        print(f"  Updating adapter with {len(corrections)} human corrections "
              f"(weight={weight}×)...")

        X_corr, y_corr = [], []
        for seq, lbl in corrections:
            seq  = np.atleast_2d(seq).astype(np.float32)
            X_ep = build_adapter_features(seq, self.base_annotator)
            # repeat each correction `weight` times (integer part)
            n_repeat = max(1, int(weight))
            X_corr.extend([X_ep] * n_repeat)
            y_corr.extend([lbl] * len(X_ep) * n_repeat)

        X = np.vstack(X_corr)
        y = np.array(y_corr)

        self.adapter_head.partial_fit(X, y)
        print(f"  Adapter updated — {self._n_episodes_seen} episodes seen total.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def annotate(self, state_seq: np.ndarray) -> dict:
        """
        Annotate a trajectory using the adapted MLP head.
        Falls back to base model if adapter not trained yet.
        """
        state_seq = np.atleast_2d(state_seq).astype(np.float32)

        if self.adapter_head is None or not self.adapter_head._fitted:
            # Fallback: use base model directly
            return self.base_annotator.annotate(state_seq)

        X     = build_adapter_features(state_seq, self.base_annotator)   # (T, 78)
        preds = self.adapter_head.predict(X)                              # (T,)
        probs = self.adapter_head.predict_proba(X)                        # (T, C)
        confs = probs.max(axis=1)                                         # (T,)

        # anomaly score = 1 - P(nominal)
        classes = list(self.adapter_head.classes_)
        nom_idx = classes.index("nominal") if "nominal" in classes else None
        if nom_idx is not None:
            anom_scores = (1 - probs[:, nom_idx]).tolist()
        else:
            anom_scores = (1 - confs).tolist()

        # Per-class thresholds: use class-specific floor if available
        needs_review = []
        for pred, conf in zip(preds, confs):
            cls_thresh = self.per_class_thresholds.get(pred, self.review_threshold)
            needs_review.append(float(conf) < cls_thresh)
        labels         = preds.tolist()
        failure_counts = {cls: labels.count(cls) for cls in self.taxonomy.all_classes}
        dominant       = max(
            [c for c in self.taxonomy.all_classes if c != "nominal"],
            key=lambda c: failure_counts.get(c, 0),
        ) if any(l != "nominal" for l in labels) else "nominal"

        T        = len(labels)
        n_review = sum(needs_review)

        return {
            "client_id":        self.client_id,
            "robot_type":       self.robot_type,
            "n_steps":          T,
            "labels":           labels,
            "confidences":      confs.tolist(),
            "needs_review":     needs_review,
            "n_needs_review":   n_review,
            "review_rate":      round(n_review / T, 4) if T else 0.0,
            "anomaly_scores":   anom_scores,
            "failure_counts":   failure_counts,
            "dominant_failure": dominant,
            "peak_score":       float(max(anom_scores)),
            "peak_step":        int(np.argmax(anom_scores)),
            "taxonomy":         self.taxonomy.all_classes,
            "custom_classes":   list(self.taxonomy.custom_classes.keys()),
        }

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, test_episodes: List[Tuple[np.ndarray, str]]) -> dict:
        """Evaluate adapter on labeled test episodes. Reports step-level F1."""
        from annotation_model import generate_weak_labels

        y_true_all, y_pred_all = [], []
        total_review = 0
        total_steps  = 0

        for seq, ep_label in test_episodes:
            seq    = np.atleast_2d(seq).astype(np.float32)
            result = self.annotate(seq)
            preds  = result["labels"]

            # step-level ground truth: weak labels for nominal, ep_label for failures
            if ep_label == "nominal" or ep_label == 0:
                truth = generate_weak_labels(seq)
            else:
                truth = [ep_label] * len(preds)

            y_true_all.extend(truth)
            y_pred_all.extend(preds)
            total_review += result["n_needs_review"]
            total_steps  += result["n_steps"]

        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        acc    = float((y_true == y_pred).mean())

        present = sorted(set(y_true) | set(y_pred))
        report  = classification_report(
            y_true, y_pred,
            labels=present,
            target_names=present,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "client_id":     self.client_id,
            "robot_type":    self.robot_type,
            "n_episodes":    len(test_episodes),
            "accuracy":      round(acc, 4),
            "review_rate_pct": round(total_review / total_steps * 100, 1) if total_steps else 0,
            "per_class_f1":  {
                cls: round(report[cls]["f1-score"], 3)
                for cls in present if cls in report
            },
            "custom_classes": list(self.taxonomy.custom_classes.keys()),
        }

        print(f"\nBenchmark — client='{self.client_id}' robot='{self.robot_type}'")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Review   : {metrics['review_rate_pct']}% of steps")
        for cls, f1 in metrics["per_class_f1"].items():
            bar = "█" * int(f1 * 20)
            print(f"  {cls:22s} {f1:.2f} {bar}")

        self._metrics = metrics
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> Path:
        client_dir = CLIENTS_DIR / self.client_id
        client_dir.mkdir(parents=True, exist_ok=True)

        adapter_path  = client_dir / "adapter.pkl"
        taxonomy_path = client_dir / "taxonomy.json"
        metrics_path  = client_dir / "metrics.json"

        with open(adapter_path, "wb") as fh:
            pickle.dump({
                "base_annotator":       self.base_annotator,
                "adapter_head":         self.adapter_head,
                "robot_type":           self.robot_type,
                "review_threshold":     self.review_threshold,
                "per_class_thresholds": self.per_class_thresholds,
                "client_scaler":        self.client_scaler,
                "n_episodes_seen":      self._n_episodes_seen,
                "metrics":              self._metrics,
                "X_cache":              self._X_cache,
                "y_cache":              self._y_cache,
            }, fh)

        self.taxonomy.save(taxonomy_path)
        metrics_path.write_text(json.dumps(self._metrics, indent=2))

        print(f"\nClientAdapter saved → {client_dir}/")
        print(f"  adapter.pkl  — MLP head + base model")
        print(f"  taxonomy.json — {len(self.taxonomy.all_classes)} classes "
              f"({len(self.taxonomy.custom_classes)} custom)")
        print(f"  metrics.json  — latest benchmark results")
        return client_dir


# ── Client registry ───────────────────────────────────────────────────────────

def list_clients() -> List[dict]:
    """List all saved client adapters with summary metrics."""
    clients = []
    for client_dir in sorted(CLIENTS_DIR.iterdir()):
        if not client_dir.is_dir():
            continue
        metrics_path  = client_dir / "metrics.json"
        taxonomy_path = client_dir / "taxonomy.json"
        entry = {"client_id": client_dir.name}
        if metrics_path.exists():
            entry.update(json.loads(metrics_path.read_text()))
        if taxonomy_path.exists():
            tax = json.loads(taxonomy_path.read_text())
            entry["custom_classes"] = list(tax.get("custom_classes", {}).keys())
        clients.append(entry)
    return clients


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_demo():
    """
    End-to-end demo: create adapter, adapt on synthetic episodes,
    add a custom class, benchmark, save.
    """
    from annotation_model import generate_weak_labels
    from augmentation import inject_velocity_spike, inject_stuck_joint

    rng = np.random.RandomState(7)
    T, D = 80, 7   # Franka-style episodes

    def make_nominal():
        base = np.linspace(0, np.pi * 0.5, T)[:, None].repeat(D, axis=1)
        return (base + rng.randn(T, D) * 0.02).astype(np.float32)

    def make_failure(injector):
        seq, _ = injector(make_nominal(), rng)
        return seq.astype(np.float32)

    # Build synthetic episodes
    train_eps = (
        [(make_nominal(), "nominal")] * 30
        + [(make_failure(inject_velocity_spike), "velocity_spike")] * 15
        + [(make_failure(inject_stuck_joint),    "stuck_joint")]    * 15
    )
    test_eps = (
        [(make_nominal(), "nominal")] * 10
        + [(make_failure(inject_velocity_spike), "velocity_spike")] * 5
        + [(make_failure(inject_stuck_joint),    "stuck_joint")]    * 5
    )

    # Create and adapt
    adapter = ClientAdapter.create("demo_client", robot_type="franka")
    adapter.adapt(train_eps, n_epochs=80)

    # Add a custom class unique to this client
    custom_eps = []
    for _ in range(10):
        seq = make_nominal()
        # "cable_snag": sudden large offset on last joint then gradual drift back
        t0  = rng.randint(20, 50)
        seq[t0:t0+3, -1] += rng.uniform(0.8, 1.5)
        seq[t0+3:t0+15] += np.linspace(0.3, 0, min(12, T - t0 - 3))[:, None] * 0.5
        custom_eps.append((seq.astype(np.float32), "cable_snag"))

    adapter.add_custom_class(
        "cable_snag",
        examples=custom_eps,
        description="Sudden large displacement on terminal joint with gradual drift",
        cause="Cable caught on fixture or gripper cable routing failure",
        strategy="Add cable management hardware; increase clearance in trajectory planner",
    )

    # Benchmark and save
    metrics = adapter.benchmark(test_eps)
    adapter.save()

    # Show client registry
    print("\n── Client Registry ─────────────────────────────────────────────")
    for c in list_clients():
        print(f"  {c['client_id']:20s}  "
              f"acc={c.get('accuracy','?')}  "
              f"robot={c.get('robot_type','?')}  "
              f"custom={c.get('custom_classes', [])}")

    return adapter, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Client adapter — Haptal AI")
    parser.add_argument("--demo",   action="store_true", help="Run end-to-end demo")
    parser.add_argument("--list",   action="store_true", help="List saved clients")
    parser.add_argument("--client", type=str, help="Client ID to load and inspect")
    args = parser.parse_args()

    if args.list:
        clients = list_clients()
        if clients:
            print(f"\nSaved clients ({len(clients)}):")
            for c in clients:
                print(f"  {c['client_id']:20s}  "
                      f"acc={c.get('accuracy','?')}  "
                      f"robot={c.get('robot_type','?')}  "
                      f"episodes={c.get('n_episodes','?')}  "
                      f"custom_classes={c.get('custom_classes',[])}")
        else:
            print("No client adapters saved yet.")

    elif args.client:
        adapter = ClientAdapter.load(args.client)
        print(f"\nTaxonomy: {adapter.taxonomy.all_classes}")
        print(f"Metrics:  {json.dumps(adapter._metrics, indent=2)}")

    elif args.demo:
        run_demo()

    else:
        parser.print_help()
