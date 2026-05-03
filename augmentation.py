"""
augmentation.py — Synthetic failure injection for rare class oversampling.

For each rare failure class, we take nominal episodes and surgically inject
the known physics signature of that failure into a random window of steps.
This gives the model hundreds of realistic training examples for classes
that barely appear in real data (self_collision: 0.1%, overshoot: 0.1%,
perception_failure: 0.3%, gripper_event: 2.1%).

Each injector returns (state_seq, labels) where:
  - state_seq : (T, D) modified trajectory
  - labels    : (T,) list of strings — only the injected window changes label
"""

import numpy as np
from typing import Tuple, List

# ── Per-class injection functions ─────────────────────────────────────────────

def inject_velocity_spike(seq: np.ndarray, rng: np.random.RandomState,
                          window: int = 4) -> Tuple[np.ndarray, List[str]]:
    """Slam one or two joints to 5-8× their normal velocity range for a short burst."""
    seq = seq.copy()
    T, D = seq.shape
    t0   = rng.randint(5, max(6, T - window - 1))
    j    = rng.randint(0, D)
    mag  = float(np.std(np.diff(seq[:, j], axis=0)) or 0.1)
    for t in range(t0, min(t0 + window, T)):
        seq[t, j] += rng.choice([-1, 1]) * mag * rng.uniform(5, 8)
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "velocity_spike"
    return seq, labels


def inject_position_jerk(seq: np.ndarray, rng: np.random.RandomState,
                         window: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Inject a sharp direction reversal — high jerk / acceleration discontinuity."""
    seq = seq.copy()
    T, D = seq.shape
    t0   = rng.randint(5, max(6, T - window - 1))
    j    = rng.randint(0, D)
    delta = float(np.std(seq[:, j]) or 0.05)
    # abrupt bump then snap back
    seq[t0,     j] += delta * rng.uniform(2, 4)
    seq[t0 + 1, j] -= delta * rng.uniform(2, 4) if t0 + 1 < T else 0
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "position_jerk"
    return seq, labels


def inject_self_collision(seq: np.ndarray, rng: np.random.RandomState,
                          window: int = 8) -> Tuple[np.ndarray, List[str]]:
    """
    Self-collision signature: multiple pairs of adjacent joints simultaneously
    moving in strongly opposing directions, with velocities well above the normal
    range. This creates the cross-joint velocity opposition pattern that the
    weak-label rule and the RF both look for.

    Improvements over v1:
    - Affects ceil(D/3) joint pairs instead of just one
    - Uses sign-alternating increments (+=, -=, +=, ...) for realism
    - Magnitude scales with per-joint velocity std (adaptive to each episode)
    - Builds up over 2 steps then sustains, matching real collision profiles
    """
    seq = seq.copy()
    T, D = seq.shape
    if D < 2:
        return seq, ["nominal"] * T

    t0 = rng.randint(5, max(6, T - window - 1))

    # per-joint velocity std for adaptive magnitude
    vel = np.diff(seq, axis=0)
    vstd = np.std(vel, axis=0) + 0.05

    # choose ceil(D/3) non-overlapping adjacent pairs
    n_pairs = max(1, int(np.ceil(D / 3)))
    starts  = rng.choice(D - 1, size=min(n_pairs, D - 1), replace=False)

    for step_i, t in enumerate(range(t0, min(t0 + window, T))):
        ramp = min(1.0, (step_i + 1) / 2)   # ramp up over first 2 steps
        for j in starts:
            mag_j  = vstd[j]     * rng.uniform(4.0, 7.0) * ramp
            mag_j1 = vstd[j + 1] * rng.uniform(4.0, 7.0) * ramp
            seq[t, j]     += mag_j
            seq[t, j + 1] -= mag_j1

    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "self_collision"
    return seq, labels


def inject_overshoot(seq: np.ndarray, rng: np.random.RandomState,
                     window: int = 5) -> Tuple[np.ndarray, List[str]]:
    """
    Large fast motion on one joint followed immediately by direction reversal —
    classic control overshoot.
    """
    seq = seq.copy()
    T, D = seq.shape
    t0   = rng.randint(5, max(6, T - window - 2))
    j    = rng.randint(0, D)
    mag  = float(np.std(seq[:, j]) or 0.05) * 2.5
    direction = rng.choice([-1, 1])
    half = window // 2
    # fast motion
    for t in range(t0, t0 + half):
        if t < T:
            seq[t, j] += direction * mag * (1 + (t - t0) * 0.4)
    # reversal
    for t in range(t0 + half, min(t0 + window, T)):
        seq[t, j] -= direction * mag * 0.8
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "overshoot"
    return seq, labels


def inject_stuck_joint(seq: np.ndarray, rng: np.random.RandomState,
                       window: int = 15) -> Tuple[np.ndarray, List[str]]:
    """Freeze one joint at its current value while others keep moving."""
    seq = seq.copy()
    T, D = seq.shape
    t0  = rng.randint(5, max(6, T - window - 1))
    j   = rng.randint(0, D)
    val = seq[t0, j]
    for t in range(t0, min(t0 + window, T)):
        seq[t, j] = val + rng.uniform(-0.001, 0.001)   # tiny noise to avoid perfect flat
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "stuck_joint"
    return seq, labels


def inject_gripper_event(seq: np.ndarray, rng: np.random.RandomState,
                         window: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Sudden large state change on the last joint (proxy for gripper open/close)."""
    seq = seq.copy()
    T, D = seq.shape
    t0   = rng.randint(5, max(6, T - window - 1))
    j    = D - 1    # last joint = gripper
    rng_val = float(np.ptp(seq[:, j]) or 0.5)
    seq[t0, j] += rng.choice([-1, 1]) * rng_val * rng.uniform(0.6, 1.0)
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "gripper_event"
    return seq, labels


def inject_perception_failure(seq: np.ndarray, rng: np.random.RandomState,
                              window: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Near-stillness (all joints barely moving) followed by a sudden large displacement —
    mimics pose estimation drift / perception failure.
    """
    seq  = seq.copy()
    T, D = seq.shape
    t0   = rng.randint(5, max(6, T - window - 1))
    half = window // 2
    # stillness phase
    for t in range(t0, t0 + half):
        if t < T:
            seq[t] = seq[t0] + rng.uniform(-0.002, 0.002, D)
    # sudden jump (perception correction)
    if t0 + half < T:
        jump = np.std(seq, axis=0) * rng.uniform(1.5, 3.0)
        seq[t0 + half] = seq[t0] + rng.choice([-1, 1], D) * jump
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "perception_failure"
    return seq, labels


def inject_trajectory_deviation(seq: np.ndarray, rng: np.random.RandomState,
                                window: int = 20) -> Tuple[np.ndarray, List[str]]:
    """Gradually drift all joints away from their mean position."""
    seq  = seq.copy()
    T, D = seq.shape
    t0   = rng.randint(5, max(6, T - window - 1))
    drift_rate = np.std(seq, axis=0) * 0.15
    for i, t in enumerate(range(t0, min(t0 + window, T))):
        seq[t] += drift_rate * (i + 1)
    labels = ["nominal"] * T
    for t in range(t0, min(t0 + window, T)):
        labels[t] = "trajectory_deviation"
    return seq, labels


# ── Injector registry ─────────────────────────────────────────────────────────

INJECTORS = {
    "velocity_spike":       inject_velocity_spike,
    "position_jerk":        inject_position_jerk,
    "self_collision":       inject_self_collision,
    "overshoot":            inject_overshoot,
    "stuck_joint":          inject_stuck_joint,
    "gripper_event":        inject_gripper_event,
    "perception_failure":   inject_perception_failure,
    "trajectory_deviation": inject_trajectory_deviation,
}


# ── Main augmentation function ────────────────────────────────────────────────

def augment_rare_classes(
    episodes: list,
    target_counts: dict,
    current_counts: dict,
    seed: int = 123,
) -> list:
    """
    Generate synthetic (state_seq, label_list) pairs to bring rare classes up
    to target_counts.

    Args:
        episodes       : list of (state_seq, ep_label, ds_name)
        target_counts  : {class_name: desired_step_count}
        current_counts : {class_name: current_step_count}
        seed           : random seed for reproducibility

    Returns:
        augmented_episodes : list of (state_seq, label_list) — synthetic only
    """
    rng = np.random.RandomState(seed)
    # Only use nominal episodes as source material
    nominal_eps = [seq for seq, ep_label, _ in episodes if ep_label == 0]
    if not nominal_eps:
        nominal_eps = [seq for seq, _, _ in episodes]   # fallback
        print("  [augment] No nominal episodes found — using all episodes as source")

    augmented = []
    for cls, target in target_counts.items():
        current = current_counts.get(cls, 0)
        deficit = target - current
        if deficit <= 0 or cls not in INJECTORS:
            continue

        injector = INJECTORS[cls]
        generated = 0
        attempts  = 0
        while generated < deficit and attempts < deficit * 5:
            attempts += 1
            src = nominal_eps[rng.randint(len(nominal_eps))]
            if len(src) < 20:
                continue
            aug_seq, aug_labels = injector(src, rng)
            # only keep if we actually got some of the target label
            if cls in aug_labels:
                augmented.append((aug_seq, aug_labels))
                generated += aug_labels.count(cls)

        print(f"  [augment] {cls:22s}: +{generated:,} synthetic steps "
              f"({len([e for e in augmented if cls in e[1]])} episodes)")

    return augmented


def compute_class_counts(X_rows: list, y_rows: list) -> dict:
    """Count current step-level label distribution."""
    from collections import Counter
    return dict(Counter(y_rows))
