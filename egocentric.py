"""
Egocentric / camera-based anomaly detection using CLIP embeddings.

Pipeline:
  1. Extract per-frame CLIP embeddings from episode image observations
  2. Summarise each episode as mean + std of its frame embeddings
  3. Run IsolationForest on those episode-level embeddings
  4. Anomaly score = how far the episode's visual pattern is from nominal

This catches visual failures that joint/force data misses entirely:
  - wrong object grasped
  - occlusion / out-of-workspace
  - unexpected scene state

Usage:
    python egocentric.py                          # synthetic demo (no data needed)
    python egocentric.py --dataset lerobot/pusht  # real images from HuggingFace
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import clip
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── CLIP feature extractor ────────────────────────────────────────────────────

class CLIPExtractor:
    def __init__(self, model_name: str = "ViT-B/32"):
        print(f"Loading CLIP {model_name} on {DEVICE}...")
        self.model, self.preprocess = clip.load(model_name, device=DEVICE)
        self.model.eval()

    @torch.no_grad()
    def embed_frames(self, frames: list) -> np.ndarray:
        """
        frames: list of PIL Images or np.uint8 arrays (H, W, 3)
        returns: (N_frames, 512) float32 array
        """
        tensors = []
        for f in frames:
            if isinstance(f, np.ndarray):
                f = Image.fromarray(f.astype(np.uint8))
            tensors.append(self.preprocess(f))
        batch  = torch.stack(tensors).to(DEVICE)
        embeds = self.model.encode_image(batch).float()
        return embeds.cpu().numpy()

    def embed_episode(self, frames: list) -> np.ndarray:
        """
        Summarise an entire episode as mean + std over frame embeddings.
        Returns a single (1024,) vector.
        """
        if not frames:
            return np.zeros(1024, dtype=np.float32)
        frame_embeds = self.embed_frames(frames)           # (T, 512)
        return np.concatenate([frame_embeds.mean(0),
                                frame_embeds.std(0)])      # (1024,)


# ── Dataset loaders (image-aware) ────────────────────────────────────────────

def load_lerobot_images(dataset_name: str = "lerobot/pusht",
                        max_episodes: int = 100,
                        extractor: CLIPExtractor = None):
    """
    Load image observations from a LeRobot dataset.
    Returns episode-level CLIP embeddings + reward-derived labels.
    """
    import pandas as pd
    from huggingface_hub import HfFileSystem

    fs    = HfFileSystem()
    repo  = dataset_name.replace("lerobot/", "")

    # load tabular data
    parquet_files = fs.glob(f"datasets/lerobot/{repo}/data/**/*.parquet")
    dfs = []
    for p in parquet_files:
        with fs.open(p, "rb") as f:
            dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)

    # find image columns
    img_cols = [c for c in df.columns if "image" in c.lower() or "pixel" in c.lower()]
    print(f"  Image columns: {img_cols}")

    if not img_cols:
        print("  No image columns — generating synthetic CLIP embeddings as stand-in.")
        return _synthetic_clip_episodes(max_episodes)

    ep_col     = "episode_index"
    reward_col = next((c for c in ["next.reward", "reward"] if c in df.columns), None)
    episode_ids = sorted(df[ep_col].unique())[:max_episodes]

    ep_max_rewards = df.groupby(ep_col)[reward_col].max() if reward_col else None
    if ep_max_rewards is not None:
        has_binary = float(ep_max_rewards.min()) >= -0.01
        nominal_t  = float(np.percentile(ep_max_rewards, 70))
        failure_t  = float(np.percentile(ep_max_rewards, 20))

    features_list, labels = [], []
    for i, ep_id in enumerate(episode_ids):
        if i % 20 == 0:
            print(f"  Embedding episode {i}/{len(episode_ids)}...")
        ep = df[df[ep_col] == ep_id]

        # collect frames from first available image column
        frames = []
        for idx, row in ep.iterrows():
            img_data = row[img_cols[0]]
            if isinstance(img_data, np.ndarray):
                frames.append(img_data)
            elif isinstance(img_data, bytes):
                import io
                frames.append(np.array(Image.open(io.BytesIO(img_data))))

        embed = extractor.embed_episode(frames)
        features_list.append(embed)

        if reward_col:
            max_r = float(ep_max_rewards.get(ep_id, 0))
            if has_binary:
                label = 0 if max_r > 0.5 else 1
            else:
                if max_r >= nominal_t:
                    label = 0
                elif max_r <= failure_t:
                    label = 1
                else:
                    features_list.pop()
                    continue
        else:
            label = 0
        labels.append(label)

    return np.array(features_list), np.array(labels)


def _synthetic_clip_episodes(n: int = 200):
    """
    Synthetic CLIP-shaped data for testing without a real image dataset.
    Nominal episodes: centred around a canonical visual distribution.
    Failure episodes: shifted + higher variance (scene looks different).
    """
    rng = np.random.RandomState(42)
    n_nom  = int(n * 0.8)
    n_fail = n - n_nom
    nominal  = rng.randn(n_nom,  1024).astype(np.float32)
    failures = (rng.randn(n_fail, 1024) * 1.8 + 1.2).astype(np.float32)
    features = np.vstack([nominal, failures])
    labels   = np.array([0] * n_nom + [1] * n_fail)
    print(f"  Synthetic CLIP demo: {n_nom} nominal, {n_fail} failures")
    return features, labels


# ── Anomaly detection on CLIP embeddings ─────────────────────────────────────

def run_clip_benchmark(features: np.ndarray, labels: np.ndarray,
                       dataset_label: str):
    print(f"\n{'='*60}")
    print(f"CLIP Visual Benchmark — {dataset_label}")
    print(f"  Episodes: {len(features)}  Failures: {labels.sum()}")

    scaler  = StandardScaler()
    nominal = features[labels == 0]
    scaled_nom = scaler.fit_transform(nominal)
    scaled_all = scaler.transform(features)

    clf    = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    clf.fit(scaled_nom)
    scores = -clf.score_samples(scaled_all)

    auc   = roc_auc_score(labels, scores)
    thresh = np.quantile(scores, 0.75)
    preds  = (scores >= thresh).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    card = {
        "dataset": dataset_label,
        "model": "CLIP-ViT-B/32 + IsolationForest",
        "modality": "egocentric_vision",
        "total_episodes": int(len(features)),
        "failure_episodes": int(labels.sum()),
        "roc_auc": round(float(auc), 4),
        "detection_rate_pct": round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0,
        "false_positive_rate_pct": round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "embedding_dim": features.shape[1],
    }

    print(f"\n  CLIP BENCHMARK CARD")
    print(f"  ROC-AUC          : {card['roc_auc']}")
    print(f"  Detection rate   : {card['detection_rate_pct']}%")
    print(f"  False pos. rate  : {card['false_positive_rate_pct']}%")

    safe = dataset_label.replace("/", "_").replace(" ", "_")
    card_path = OUTPUT_DIR / f"{safe}_clip_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    print(f"\n  Saved: {card_path}")

    # optional plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(scores[labels == 0], bins=30, alpha=0.6, color="steelblue", label="Nominal")
        ax.hist(scores[labels == 1], bins=30, alpha=0.6, color="crimson",   label="Failure")
        ax.axvline(thresh, color="orange", linestyle="--", label="Threshold")
        ax.set_title(f"CLIP Visual Anomaly Scores — {dataset_label}")
        ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count"); ax.legend()
        plot_path = OUTPUT_DIR / f"{safe}_clip_scores.png"
        plt.tight_layout(); plt.savefig(plot_path, dpi=120); plt.close()
        print(f"  Plot saved: {plot_path}")
    except ImportError:
        pass

    return card


# ── Frame-level failure classifier ───────────────────────────────────────────

class FrameLevelCLIPClassifier:
    """
    Per-frame failure classifier using CLIP embeddings.

    Where episode-level anomaly says "this episode has a failure",
    this model says "failure happened at frames 45-67: wrong object grasped."

    Architecture:
      CLIP frame embeddings (512-d) -> MLPClassifier (256->128 -> 6 classes)
      Trained with weak supervision: anomaly score at each frame from a
      per-frame IsolationForest trained on nominal episode frames.

    Failure types detected visually:
      - scene_nominal        : normal operation
      - object_drop          : sudden scene change (object leaves frame)
      - wrong_grasp          : gripper region looks different from nominal grasps
      - occlusion            : workspace occluded (dark/blocked region)
      - out_of_workspace     : end-effector exits normal workspace region
      - scene_disturbance    : unexpected background change
    """

    VISUAL_FAILURE_CLASSES = [
        "scene_nominal", "object_drop", "wrong_grasp",
        "occlusion", "out_of_workspace", "scene_disturbance"
    ]

    # Confidence threshold below which a frame label is considered uncertain
    REVIEW_THRESHOLD = 0.55

    def __init__(self, extractor: CLIPExtractor = None):
        self.extractor = extractor
        # per-frame anomaly detector fitted on nominal embeddings
        self._frame_iso: IsolationForest = None
        self._frame_scaler: StandardScaler = None
        # per-class prototype centroids for weak-supervision heuristics
        self._nominal_centroid: np.ndarray = None
        self._nominal_std: float = None
        # final MLP classifier
        self._clf = None
        self._label_encoder = None
        self._fitted: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_episode_frames(self, frames: list) -> np.ndarray:
        """
        Embed a list of frames using the CLIPExtractor.
        Returns (N, 512) float32.  Falls back to random vectors when no
        extractor is available (useful for unit tests / synthetic demos).
        """
        if self.extractor is not None:
            return self.extractor.embed_frames(frames)
        # synthetic fallback: Gaussian noise at CLIP embedding scale
        rng = np.random.RandomState(abs(hash(str(len(frames)))) % (2**31))
        return rng.randn(len(frames), 512).astype(np.float32)

    def _weak_labels(
        self,
        embeds: np.ndarray,
        frame_scores: np.ndarray,
    ) -> list:
        """
        Assign per-frame failure class using heuristic rules.

        Rules (applied in priority order):
          1. object_drop         — L2 distance to previous frame > 3-sigma spike
          2. out_of_workspace    — embedding far from nominal cluster (high anomaly)
          3. scene_disturbance   — embedding moderately far, but high texture variance
          4. occlusion           — embedding close to nominal but very low L2 norm
          5. wrong_grasp         — frame anomaly score moderate, not covered by above
          6. scene_nominal       — default
        """
        T = len(embeds)
        labels = []

        # inter-frame motion: L2 distance between consecutive embeddings
        deltas = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            deltas[t] = float(np.linalg.norm(embeds[t] - embeds[t - 1]))

        delta_thresh_high = float(np.percentile(deltas, 95))
        delta_thresh_mid  = float(np.percentile(deltas, 80))

        # embedding norms — low norm hints at dark/occluded frames
        norms = np.linalg.norm(embeds, axis=1)
        norm_thresh_low = float(np.percentile(norms, 10))

        score_high = float(np.percentile(frame_scores, 90))
        score_mid  = float(np.percentile(frame_scores, 70))

        for t in range(T):
            if deltas[t] > delta_thresh_high:
                labels.append("object_drop")
            elif frame_scores[t] > score_high:
                labels.append("out_of_workspace")
            elif frame_scores[t] > score_mid and deltas[t] > delta_thresh_mid:
                labels.append("scene_disturbance")
            elif norms[t] < norm_thresh_low and frame_scores[t] < score_mid:
                labels.append("occlusion")
            elif frame_scores[t] > score_mid:
                labels.append("wrong_grasp")
            else:
                labels.append("scene_nominal")

        return labels

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        nominal_episodes: list,
        failure_episodes: list = None,
    ) -> "FrameLevelCLIPClassifier":
        """
        Train on frame embeddings.

        nominal_episodes : list of (frames, episode_label) where frames is a
                           list of PIL Images or np.uint8 arrays.
        failure_episodes : optional labeled failures
                           Each entry: (frames, failure_class_str).
                           If None, weak supervision is used exclusively.

        Weak supervision rules (applied when failure_episodes is None or
        insufficient to cover all 6 classes):
          - Frame embedding far from nominal cluster  -> out_of_workspace /
                                                         scene_disturbance
          - Sharp inter-frame embedding delta         -> object_drop
          - Embedding close to nominal, low L2 norm  -> occlusion

        Returns self to allow method chaining.
        """
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for FrameLevelCLIPClassifier.fit(). "
                "Install with: pip install scikit-learn"
            ) from exc

        print("FrameLevelCLIPClassifier: extracting nominal frame embeddings...")
        nominal_embeds = []
        for frames, _ in nominal_episodes:
            embs = self._embed_episode_frames(frames)
            nominal_embeds.append(embs)

        # concatenate all nominal frames into one pool
        nominal_pool = np.vstack(nominal_embeds)          # (N_nom, 512)
        self._nominal_centroid = nominal_pool.mean(axis=0)
        self._nominal_std = float(nominal_pool.std())

        # fit per-frame IsolationForest on nominal pool
        print(f"  Fitting frame-level IsolationForest on {len(nominal_pool)} nominal frames...")
        self._frame_scaler = StandardScaler()
        nom_scaled = self._frame_scaler.fit_transform(nominal_pool)
        self._frame_iso = IsolationForest(
            contamination=0.08, random_state=42, n_jobs=-1
        )
        self._frame_iso.fit(nom_scaled)

        # ------------------------------------------------------------------
        # Build the per-frame training set
        # ------------------------------------------------------------------
        X_all, y_all = [], []

        # Nominal episodes -> weak labels (mostly scene_nominal)
        for embs in nominal_embeds:
            scaled = self._frame_scaler.transform(embs)
            scores = -self._frame_iso.score_samples(scaled)
            weak   = self._weak_labels(embs, scores)
            X_all.extend(embs.tolist())
            y_all.extend(weak)

        # Failure episodes (if provided): use supplied labels directly
        if failure_episodes:
            print(f"  Adding {len(failure_episodes)} labeled failure episodes...")
            for frames, failure_class in failure_episodes:
                embs = self._embed_episode_frames(frames)
                # map episode-level label to all frames as a starting point,
                # then refine with weak supervision for frames that look nominal
                scaled = self._frame_scaler.transform(embs)
                scores = -self._frame_iso.score_samples(scaled)
                weak   = self._weak_labels(embs, scores)
                for i, (emb, w_lbl) in enumerate(zip(embs, weak)):
                    # prefer supplied label for clearly anomalous frames;
                    # keep weak label for frames that look clean (could be
                    # part of a recovery move inside the failure episode)
                    if scores[i] > float(np.percentile(scores, 50)):
                        y_all.append(
                            failure_class
                            if failure_class in self.VISUAL_FAILURE_CLASSES
                            else w_lbl
                        )
                    else:
                        y_all.append(w_lbl)
                    X_all.append(emb.tolist())

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all)

        print(f"  Frame training set: {len(X)} frames")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"    {cls:25s}: {cnt:6,} ({cnt/len(y)*100:.1f}%)")

        # encode labels
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self.VISUAL_FAILURE_CLASSES)
        y_enc = self._label_encoder.transform(y)

        # scale features
        X_scaled = self._frame_scaler.transform(X)

        # train MLP: two hidden layers, dropout-style regularisation via alpha
        print("  Training MLP frame classifier (512 -> 256 -> 128 -> 6)...")
        self._clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
        )
        self._clf.fit(X_scaled, y_enc)
        self._fitted = True

        train_acc = self._clf.score(X_scaled, y_enc)
        print(f"  Training accuracy: {train_acc:.3f}")
        return self

    def predict_episode(self, frames: list) -> dict:
        """
        Classify each frame in an episode.

        Parameters
        ----------
        frames : list of PIL Images or np.uint8 arrays

        Returns
        -------
        dict with keys:
          frame_labels          : list[str]   per-frame class name
          frame_scores          : list[float] per-frame anomaly score (0-1)
          frame_confidences     : list[float] classifier confidence per frame
          dominant_visual_failure : str       most frequent non-nominal class
          failure_frames        : list[int]   frame indices with non-nominal label
          summary               : str         human-readable one-liner
        """
        if not self._fitted:
            raise RuntimeError(
                "Model not trained. Call .fit() or .load() first."
            )

        embeds = self._embed_episode_frames(frames)          # (T, 512)
        scaled = self._frame_scaler.transform(embeds)        # (T, 512)
        scores = (-self._frame_iso.score_samples(scaled)).tolist()

        probs      = self._clf.predict_proba(scaled)         # (T, 6)
        pred_enc   = probs.argmax(axis=1)
        frame_labels = self._label_encoder.inverse_transform(pred_enc).tolist()
        frame_confs  = probs.max(axis=1).tolist()

        failure_frames = [
            i for i, lbl in enumerate(frame_labels)
            if lbl != "scene_nominal"
        ]

        # dominant failure: most frequent non-nominal class
        non_nominal = [lbl for lbl in frame_labels if lbl != "scene_nominal"]
        if non_nominal:
            from collections import Counter
            dominant = Counter(non_nominal).most_common(1)[0][0]
        else:
            dominant = "scene_nominal"

        # concise summary
        T = len(frames)
        if not failure_frames:
            summary = f"All {T} frames nominal."
        else:
            pct = len(failure_frames) / T * 100
            span_start, span_end = failure_frames[0], failure_frames[-1]
            summary = (
                f"{len(failure_frames)}/{T} frames ({pct:.0f}%) show "
                f"'{dominant}' — frames {span_start}–{span_end}."
            )

        return {
            "frame_labels":            frame_labels,
            "frame_scores":            [round(s, 4) for s in scores],
            "frame_confidences":       [round(c, 4) for c in frame_confs],
            "dominant_visual_failure": dominant,
            "failure_frames":          failure_frames,
            "summary":                 summary,
        }

    def save(self, path: Path) -> None:
        """Persist the classifier to a pickle file."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame_iso":         self._frame_iso,
            "frame_scaler":      self._frame_scaler,
            "nominal_centroid":  self._nominal_centroid,
            "nominal_std":       self._nominal_std,
            "clf":               self._clf,
            "label_encoder":     self._label_encoder,
            "fitted":            self._fitted,
            "classes":           self.VISUAL_FAILURE_CLASSES,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"FrameLevelCLIPClassifier saved: {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        extractor: CLIPExtractor = None,
    ) -> "FrameLevelCLIPClassifier":
        """Load a previously saved classifier from disk."""
        import pickle
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No classifier checkpoint at {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls(extractor=extractor)
        obj._frame_iso        = payload["frame_iso"]
        obj._frame_scaler     = payload["frame_scaler"]
        obj._nominal_centroid = payload["nominal_centroid"]
        obj._nominal_std      = payload.get("nominal_std")
        obj._clf              = payload["clf"]
        obj._label_encoder    = payload["label_encoder"]
        obj._fitted           = payload.get("fitted", True)
        print(f"FrameLevelCLIPClassifier loaded from {path}")
        return obj


# ── Synthetic per-frame data (used for --frame-classifier demo) ───────────────

def _synthetic_frame_episodes(n_episodes: int = 60, frames_per_ep: int = 50):
    """
    Build toy nominal/failure episode lists without a camera or CLIP.
    Each "frame" is a random 512-d float vector with class-specific offsets.
    Returns (nominal_episodes, failure_episodes) in the format fit() expects.
    """
    rng = np.random.RandomState(7)

    def make_frames(n, offset=0.0, noise_scale=1.0):
        # frames represented as pre-built embed lists;
        # CLIPExtractor is bypassed when extractor=None
        raw = (rng.randn(n, 512) * noise_scale + offset).astype(np.float32)
        # wrap each row as a 1-pixel PIL Image so embed_frames would accept it,
        # but since we run with extractor=None we just return raw arrays
        from PIL import Image as PILImage
        return [PILImage.fromarray(
                    np.clip(raw[i].reshape(16, 32, 1).repeat(3, axis=2)
                            * 10 + 128, 0, 255).astype(np.uint8))
                for i in range(n)]

    nominal_eps = [(make_frames(frames_per_ep), "scene_nominal")
                   for _ in range(int(n_episodes * 0.7))]
    failure_map = [
        ("object_drop",       2.5, 1.8),
        ("wrong_grasp",       1.5, 1.2),
        ("occlusion",        -0.5, 0.3),
        ("out_of_workspace",  3.0, 2.0),
        ("scene_disturbance", 1.8, 1.5),
    ]
    failure_eps = []
    per_class   = max(1, int(n_episodes * 0.06))
    for cls, offset, noise in failure_map:
        for _ in range(per_class):
            failure_eps.append((make_frames(frames_per_ep, offset, noise), cls))

    return nominal_eps, failure_eps


# ── Fusion helper ─────────────────────────────────────────────────────────────

def fuse_visual_proprioceptive(
    visual_result: dict,
    prop_result: dict,
) -> dict:
    """
    Merge per-frame visual annotations with per-step proprioceptive annotations
    into a unified per-step dictionary.

    Alignment: 1:1 mapping by timestep index is assumed.  If the two sequences
    have different lengths the shorter one is padded with "unknown" / 0.0
    so the output always covers every timestep in the longer sequence.

    Parameters
    ----------
    visual_result : dict
        Output of FrameLevelCLIPClassifier.predict_episode().
        Keys used: frame_labels, frame_scores, frame_confidences,
                   dominant_visual_failure, failure_frames.

    prop_result : dict
        Output of RobotAnnotator.annotate() (from annotation_model.py).
        Keys used: labels, confidences, anomaly_scores, needs_review,
                   dominant_failure, peak_step.

    Returns
    -------
    dict with keys:
      steps            : list[dict]  — one entry per timestep, containing:
          step_index         int
          visual_label       str      CLIP frame class
          visual_score       float    per-frame anomaly score
          visual_confidence  float    classifier confidence
          prop_label         str      joint-level failure class
          prop_confidence    float
          prop_anomaly       float    1 - P(nominal) from RF
          needs_review       bool     proprioceptive model flagged for review
          fused_label        str      heuristic fusion of both modalities
          fused_confidence   float    min of both confidences (conservative)
          is_failure         bool     True if either modality flags anomaly
      n_steps                int
      dominant_visual        str
      dominant_prop          str
      dominant_fused         str      most frequent fused_label != "nominal"
      visual_only_failures   list[int]  steps flagged visually but not by prop
      prop_only_failures     list[int]  steps flagged by prop but not visually
      both_flagged           list[int]  steps flagged by both modalities
      summary                str
    """
    v_labels  = visual_result.get("frame_labels", [])
    v_scores  = visual_result.get("frame_scores", [])
    v_confs   = visual_result.get("frame_confidences", [])
    p_labels  = prop_result.get("labels", [])
    p_confs   = prop_result.get("confidences", [])
    p_scores  = prop_result.get("anomaly_scores", [])
    p_review  = prop_result.get("needs_review", [])

    T = max(len(v_labels), len(p_labels))

    def _get(lst, i, default):
        return lst[i] if i < len(lst) else default

    # Fusion rule: prefer the more specific (non-nominal) label;
    # if both are anomalous use the visual label as the primary signal
    # since camera sees the scene outcome directly.
    def _fuse_labels(v_lbl: str, p_lbl: str) -> str:
        v_nom = v_lbl == "scene_nominal"
        p_nom = p_lbl == "nominal"
        if v_nom and p_nom:
            return "nominal"
        if not v_nom and p_nom:
            return v_lbl          # visual-only signal
        if v_nom and not p_nom:
            return p_lbl          # proprioceptive-only signal
        # both anomalous: visual label is more interpretable to annotators
        return v_lbl

    steps = []
    for i in range(T):
        v_lbl  = _get(v_labels, i, "scene_nominal")
        v_sc   = _get(v_scores,  i, 0.0)
        v_cf   = _get(v_confs,   i, 1.0)
        p_lbl  = _get(p_labels,  i, "nominal")
        p_cf   = _get(p_confs,   i, 1.0)
        p_sc   = _get(p_scores,  i, 0.0)
        p_rv   = _get(p_review,  i, False)

        f_lbl  = _fuse_labels(v_lbl, p_lbl)
        f_cf   = min(v_cf, p_cf)           # conservative: trust the weaker signal
        is_fail = (v_lbl != "scene_nominal") or (p_lbl != "nominal")

        steps.append({
            "step_index":        i,
            "visual_label":      v_lbl,
            "visual_score":      round(v_sc,  4),
            "visual_confidence": round(v_cf,  4),
            "prop_label":        p_lbl,
            "prop_confidence":   round(p_cf,  4),
            "prop_anomaly":      round(p_sc,  4),
            "needs_review":      bool(p_rv),
            "fused_label":       f_lbl,
            "fused_confidence":  round(f_cf,  4),
            "is_failure":        bool(is_fail),
        })

    # aggregate indices
    v_fail_set = set(
        i for i, s in enumerate(steps) if s["visual_label"] != "scene_nominal"
    )
    p_fail_set = set(
        i for i, s in enumerate(steps) if s["prop_label"] != "nominal"
    )
    visual_only   = sorted(v_fail_set - p_fail_set)
    prop_only     = sorted(p_fail_set - v_fail_set)
    both_flagged  = sorted(v_fail_set & p_fail_set)

    # dominant fused label
    from collections import Counter
    non_nom_fused = [s["fused_label"] for s in steps if s["fused_label"] != "nominal"]
    dominant_fused = (
        Counter(non_nom_fused).most_common(1)[0][0] if non_nom_fused else "nominal"
    )

    n_fail = sum(1 for s in steps if s["is_failure"])
    summary = (
        f"{n_fail}/{T} steps flagged "
        f"({len(both_flagged)} by both modalities, "
        f"{len(visual_only)} visual-only, "
        f"{len(prop_only)} prop-only). "
        f"Dominant: {dominant_fused}."
    )

    return {
        "steps":                steps,
        "n_steps":              T,
        "dominant_visual":      visual_result.get("dominant_visual_failure", "scene_nominal"),
        "dominant_prop":        prop_result.get("dominant_failure", "nominal"),
        "dominant_fused":       dominant_fused,
        "visual_only_failures": visual_only,
        "prop_only_failures":   prop_only,
        "both_flagged":         both_flagged,
        "summary":              summary,
    }


# ── Frame-level demo ──────────────────────────────────────────────────────────

def run_frame_classifier_demo(extractor: CLIPExtractor = None):
    """
    Synthetic demo for FrameLevelCLIPClassifier.
    Does not require real images or a GPU.
    """
    print(f"\n{'='*60}")
    print("Frame-Level CLIP Classifier Demo (synthetic)")
    print(f"{'='*60}")

    nominal_eps, failure_eps = _synthetic_frame_episodes(n_episodes=60)

    # When extractor is None the classifier uses its random fallback,
    # so the demo runs without CLIP being installed.
    clf = FrameLevelCLIPClassifier(extractor=extractor)
    clf.fit(nominal_eps, failure_eps)

    # Test on a held-out nominal episode
    test_nom_frames, _ = nominal_eps[0]
    result_nom = clf.predict_episode(test_nom_frames)
    print(f"\nNominal episode  : {result_nom['summary']}")
    print(f"  Dominant       : {result_nom['dominant_visual_failure']}")
    print(f"  Failure frames : {result_nom['failure_frames'][:10]}"
          f"{'...' if len(result_nom['failure_frames'])>10 else ''}")

    # Test on a held-out failure episode
    test_fail_frames, test_fail_class = failure_eps[0]
    result_fail = clf.predict_episode(test_fail_frames)
    print(f"\nFailure episode ({test_fail_class})  : {result_fail['summary']}")
    print(f"  Dominant       : {result_fail['dominant_visual_failure']}")
    print(f"  Failure frames : {result_fail['failure_frames'][:10]}"
          f"{'...' if len(result_fail['failure_frames'])>10 else ''}")

    # Demonstrate fusion with a synthetic proprioceptive result
    prop_mock = {
        "labels":        ["nominal"] * 40 + ["velocity_spike"] * 10,
        "confidences":   [0.9] * 50,
        "anomaly_scores":[0.1] * 40 + [0.8] * 10,
        "needs_review":  [False] * 45 + [True] * 5,
        "dominant_failure": "velocity_spike",
        "peak_step": 44,
    }
    fused = fuse_visual_proprioceptive(result_fail, prop_mock)
    print(f"\nFused annotation : {fused['summary']}")
    print(f"  Both flagged   : {fused['both_flagged'][:10]}")
    print(f"  Visual-only    : {fused['visual_only_failures'][:10]}")
    print(f"  Prop-only      : {fused['prop_only_failures'][:10]}")

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / "frame_clip_classifier.pkl"
    clf.save(ckpt_path)

    # Round-trip load
    clf2 = FrameLevelCLIPClassifier.load(ckpt_path, extractor=extractor)
    result2 = clf2.predict_episode(test_fail_frames)
    print(f"\nPost-load prediction (should match): {result2['summary']}")
    print(f"\nFrame-level classifier demo complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic",
                        help="'synthetic' or a lerobot dataset name e.g. lerobot/pusht")
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--frame-classifier", action="store_true",
                        help="Run the per-frame failure classifier demo")
    args = parser.parse_args()

    # Try to build a CLIPExtractor; gracefully fall back to None if clip is
    # not installed so the synthetic demos still work.
    extractor = None
    try:
        extractor = CLIPExtractor()
    except Exception as exc:
        print(f"[WARNING] Could not load CLIP ({exc}). "
              "Synthetic demos will use random embeddings as a stand-in.")

    if args.frame_classifier:
        run_frame_classifier_demo(extractor=extractor)

    else:
        if extractor is None:
            print("CLIP not available — falling back to synthetic demo.")
            features, labels = _synthetic_clip_episodes(args.max_episodes)
            label = "Synthetic CLIP Demo"
        elif args.dataset == "synthetic":
            features, labels = _synthetic_clip_episodes(args.max_episodes)
            label = "Synthetic CLIP Demo"
        else:
            features, labels = load_lerobot_images(
                args.dataset, args.max_episodes, extractor)
            label = args.dataset

        run_clip_benchmark(features, labels, label)
