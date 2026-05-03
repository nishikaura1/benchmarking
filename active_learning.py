"""
Active Learning Query Strategy — Haptal AI Annotation Pipeline

Instead of routing ALL low-confidence steps to humans, this module selects
the MOST INFORMATIVE ones: steps where human input gives the greatest signal
boost to the next training iteration.

Two complementary strategies are combined:
  1. Uncertainty sampling — steps where the model is most confused
       • Entropy:  H = -Σ p_i log(p_i)  (high = uniform distribution = lost)
       • Margin:   1 - (p₁ - p₂)        (high = top two classes nearly tied)
  2. Diversity selection — greedy k-center in PCA feature space
       • Avoids reviewing 15 identical velocity spikes
       • Surfaces underrepresented failure modes
       • Each selected step covers a different region of the decision boundary

The result: a ranked review list where the first N steps give the most
information gain per human-minute spent labeling.

Usage:
  from active_learning import ActiveLearningSelector, score_uncertainty

  selector = ActiveLearningSelector(budget=20, diversity_frac=0.4)
  ranked   = selector.rank(feats, probs, step_indices=low_conf_indices)
  # ranked → [{"step": 67, "informativeness": 0.91, "reason": "...", "priority": 1}, ...]
"""

import numpy as np
from sklearn.decomposition import PCA

ENTROPY_WEIGHT = 0.5
MARGIN_WEIGHT  = 0.5


# ── Uncertainty scores ────────────────────────────────────────────────────────

def score_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Per-step informativeness score from predicted class probabilities.

    probs : (N, C) — calibrated probability matrix
    Returns (N,) scores in [0, 1]; higher = more uncertain = more valuable to label.
    """
    eps    = 1e-10
    probs  = np.clip(probs, eps, 1 - eps)
    C      = probs.shape[1]

    # Entropy: -Σ p log(p), normalised to [0,1] by dividing by log(C)
    entropy = -np.sum(probs * np.log(probs), axis=1) / np.log(max(C, 2))

    # Margin: 1 - (top1 prob - top2 prob); high when model can't choose
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    margin   = 1.0 - (sorted_p[:, 0] - sorted_p[:, 1])

    return ENTROPY_WEIGHT * entropy + MARGIN_WEIGHT * margin


# ── Diversity selection via greedy k-center ───────────────────────────────────

def greedy_kcenter(feats: np.ndarray, uncertainty: np.ndarray,
                   n_select: int) -> np.ndarray:
    """
    Greedy k-center selection in PCA(3) embedding space.

    Seeds with the highest-uncertainty point, then iteratively adds the
    point that maximises the minimum distance to already-selected points.
    This spreads selections across the feature space, covering diverse modes.

    Returns indices (into feats) of selected points.
    """
    N = len(feats)
    if N == 0 or n_select == 0:
        return np.array([], dtype=int)
    n_select = min(n_select, N)

    # Embed in 3-D PCA space for distance computation
    try:
        n_comp = min(3, feats.shape[1], N)
        emb    = PCA(n_components=n_comp).fit_transform(feats)
    except Exception:
        emb = feats[:, :min(3, feats.shape[1])]

    # Seed: highest-uncertainty point
    selected  = [int(np.argmax(uncertainty))]
    remaining = list(set(range(N)) - {selected[0]})

    while len(selected) < n_select and remaining:
        sel_emb  = emb[selected]                        # (S, d)
        rem_emb  = emb[remaining]                       # (R, d)

        # For each remaining point: min dist to any selected
        diffs    = rem_emb[:, None, :] - sel_emb[None, :, :]  # (R, S, d)
        dists    = np.linalg.norm(diffs, axis=2).min(axis=1)  # (R,)

        best     = int(np.argmax(dists))
        selected.append(remaining[best])
        remaining.pop(best)

    return np.array(selected, dtype=int)


# ── Main selector ─────────────────────────────────────────────────────────────

class ActiveLearningSelector:
    """
    Rank uncertain steps by expected information gain for human review.

    Parameters
    ----------
    budget        : max steps to surface per episode (default 20)
    diversity_frac: fraction of budget filled by k-center diversity
                    (0 = pure uncertainty; 0.5 = half uncertainty, half diverse)
    """

    def __init__(self, budget: int = 20, diversity_frac: float = 0.4):
        self.budget         = budget
        self.diversity_frac = max(0.0, min(1.0, diversity_frac))

    def rank(self, feats: np.ndarray, probs: np.ndarray,
             step_indices: list = None) -> list:
        """
        Rank steps by informativeness.

        feats        : (N, F) feature matrix for the N uncertain steps
        probs        : (N, C) calibrated probability matrix for those steps
        step_indices : original step numbers (default 0..N-1)

        Returns
        -------
        list of dicts, sorted by priority (1 = most informative):
          {
            "step":            int,   # original step index in the episode
            "informativeness": float, # combined uncertainty score [0,1]
            "reason":          str,   # explanation shown to human reviewer
            "priority":        int,   # rank (1 = label this first)
            "strategy":        str,   # "uncertainty" | "diversity"
          }
        """
        N = len(feats)
        if N == 0:
            return []

        if step_indices is None:
            step_indices = list(range(N))

        unc_scores = score_uncertainty(probs)

        # Split budget
        n_diverse = min(int(self.budget * self.diversity_frac), N)
        n_unc     = min(self.budget - n_diverse, N)

        # Uncertainty-only indices (top n_unc by score)
        unc_idx = set(np.argsort(-unc_scores)[:n_unc].tolist())

        # Diversity indices via k-center
        total    = min(self.budget, N)
        selected = greedy_kcenter(feats, unc_scores, total)

        # Tag each as uncertainty vs diversity
        results = []
        for rank_i, local_i in enumerate(selected):
            score    = float(unc_scores[local_i])
            strategy = "uncertainty" if local_i in unc_idx else "diversity"
            if strategy == "uncertainty":
                reason = (
                    "Model is near the decision boundary — "
                    "labeling this step directly improves confidence calibration"
                )
            else:
                reason = (
                    "Covers an under-sampled region of the failure space — "
                    "labeling this exposes a failure mode not yet seen"
                )
            results.append({
                "step":            step_indices[local_i],
                "local_idx":       int(local_i),
                "informativeness": round(score, 4),
                "reason":          reason,
                "priority":        rank_i + 1,
                "strategy":        strategy,
            })

        return sorted(results, key=lambda x: x["priority"])


# ── Episode-level prioritisation ──────────────────────────────────────────────

def rank_episodes_for_review(episodes_meta: list) -> list:
    """
    Sort episodes by expected label information gain.

    episodes_meta : list of dicts, each with:
      - "episode_id"     : str
      - "n_needs_review" : int
      - "review_rate"    : float  (fraction of uncertain steps)
      - "quality_score"  : float  (0-1, higher = cleaner)
      - "anomaly_score"  : float

    Returns the same list sorted by priority (highest first).
    Score = 0.5*review_rate + 0.3*(1 - quality_score) + 0.2*anomaly_score
    Episodes with many uncertain steps AND low quality have highest gain.
    """
    def ep_score(ep):
        rr  = ep.get("review_rate", 0.0)
        qs  = ep.get("quality_score", 0.5)
        ans = ep.get("anomaly_score", 0.0)
        return 0.5 * rr + 0.3 * (1 - qs) + 0.2 * ans

    return sorted(episodes_meta, key=ep_score, reverse=True)


# ── CLI (standalone scoring) ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(0)
    N, C, F = 30, 6, 36
    probs = np.random.dirichlet(np.ones(C) * 0.5, size=N)
    feats = np.random.randn(N, F)

    sel = ActiveLearningSelector(budget=10, diversity_frac=0.4)
    ranked = sel.rank(feats, probs)
    print(f"Selected {len(ranked)} steps:")
    for r in ranked:
        print(f"  [{r['priority']:2d}] step={r['step']:3d}  info={r['informativeness']:.3f}"
              f"  [{r['strategy']:11s}]  {r['reason'][:60]}...")
