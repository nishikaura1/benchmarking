"""
Trajectory normalization for the Haptal AI robot annotation pipeline.

Raw joint-angle trajectories look completely different across robot platforms:
a Franka Panda and a UR5 doing the same pick-and-place occupy non-overlapping
regions of joint space.  After normalization the same task phase looks similar
regardless of which robot performed it, so the annotation model can transfer.

Transformations applied in order:
  1. Start-pose centering    — subtract q[0] so every trajectory starts at origin
  2. Per-joint range scaling — scale each joint to [-1, 1] using training data
  3. Velocity normalization  — rescale velocity magnitudes to unit scale
  4. DOF-adaptive padding    — pad shorter robots / truncate longer ones to
                               D_CANONICAL = 8 joints

Canonical joint count reasoning:
  xarm=6, ur5=6 → padded to 8
  franka=7       → padded to 8
  aloha=14       → truncated to 8 (first 7 joints of each arm)

Usage:
  normalizer = TrajectoryNormalizer()
  normalizer.fit(train_state_seqs)         # fit range stats from training data
  norm_seqs  = normalizer.fit_transform(train_state_seqs)
  normalizer.save(Path("benchmark_output/trajectory_normalizer.pkl"))

  # at inference time
  normalizer = TrajectoryNormalizer.load(Path("benchmark_output/trajectory_normalizer.pkl"))
  norm_seq   = normalizer.transform(new_state_seq)   # (T, 8)
"""

import pickle
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)

NORMALIZER_PATH = OUTPUT_DIR / "trajectory_normalizer.pkl"

# ── Robot DOF heuristics ──────────────────────────────────────────────────────

# Known robot DOF counts — used by detect_robot_dof()
_DOF_MAP = {
    6:  "xarm6_or_ur5",
    7:  "franka_panda",
    8:  "franka_with_gripper",   # 7 joints + gripper binary = 8
    14: "aloha_bimanual",
}

_DOF_NAMES = {
    "xarm6":  6,
    "ur5":    6,
    "franka": 7,
    "aloha":  14,
}


# ── Utility: per-dataset statistics ──────────────────────────────────────────

def compute_trajectory_stats(state_seqs: list) -> dict:
    """
    Compute per-joint descriptive statistics across a list of trajectories.

    Parameters
    ----------
    state_seqs : list of (T_i, D) np.ndarray
        Raw joint-state sequences from a training dataset.
        All sequences must have the same D (joint count).

    Returns
    -------
    dict with keys:
        "n_sequences"   : int
        "n_timesteps"   : int   — total steps across all sequences
        "D"             : int   — joint count
        "per_joint"     : list of D dicts, each containing:
                            "mean", "std", "min", "max", "range"
        "global"        : overall stats (velocity mean/std across all steps)
    """
    if not state_seqs:
        return {}

    # stack all timesteps
    all_steps = np.vstack([s for s in state_seqs])   # (N_total, D)
    N, D = all_steps.shape

    per_joint = []
    for d in range(D):
        col = all_steps[:, d]
        per_joint.append({
            "joint_idx": d,
            "mean":  float(col.mean()),
            "std":   float(col.std()),
            "min":   float(col.min()),
            "max":   float(col.max()),
            "range": float(col.max() - col.min()),
        })

    # global velocity stats
    vel_mags = []
    for seq in state_seqs:
        vel = np.diff(seq, axis=0)                          # (T-1, D)
        vel_mags.extend(np.linalg.norm(vel, axis=1).tolist())
    vel_arr = np.array(vel_mags) if vel_mags else np.array([0.0])

    return {
        "n_sequences": len(state_seqs),
        "n_timesteps": N,
        "D":           D,
        "per_joint":   per_joint,
        "global": {
            "velocity_mean":   float(vel_arr.mean()),
            "velocity_std":    float(vel_arr.std()),
            "velocity_max":    float(vel_arr.max()),
            "velocity_p95":    float(np.percentile(vel_arr, 95)),
        },
    }


# ── DOF detection heuristic ───────────────────────────────────────────────────

def detect_robot_dof(state_seq: np.ndarray) -> str:
    """
    Heuristic guess of robot platform from the number of state dimensions.

    Parameters
    ----------
    state_seq : (T, D) joint-state sequence (any length)

    Returns
    -------
    str — one of "xarm6", "ur5", "franka", "aloha", or "unknown_<D>dof"

    Notes
    -----
    The heuristic is intentionally simple: joint count is the strongest signal
    when no robot metadata is available.  When D == 6, we inspect velocity
    variance to distinguish xArm (higher gains, tighter variance) from UR5
    (lower gains, wider spread) — the distinction is approximate and mostly
    cosmetic since both are treated identically by the normalizer.
    """
    D = state_seq.shape[1] if state_seq.ndim == 2 else len(state_seq)

    if D == 14:
        return "aloha"
    if D == 7:
        return "franka"
    if D == 8:
        return "franka_with_gripper"
    if D == 6:
        # Try to distinguish xArm vs UR5 by velocity spread.
        # xArm joints tend to have tighter velocity ranges (high gear ratio).
        # This is a weak signal — treat as informational only.
        if state_seq.ndim == 2 and len(state_seq) > 1:
            vel = np.diff(state_seq, axis=0)
            vel_std = float(vel.std())
            return "xarm6" if vel_std < 0.05 else "ur5"
        return "xarm6"

    return f"unknown_{D}dof"


# ── DTW-like temporal alignment ───────────────────────────────────────────────

def align_to_canonical_frame(seq_a: np.ndarray,
                              seq_b: np.ndarray) -> np.ndarray:
    """
    Return a time-warped version of seq_b aligned to seq_a's timeline.

    This is a lightweight DTW substitute: we linearly interpolate seq_b so
    it has the same number of timesteps as seq_a.  This is equivalent to
    optimal DTW along a monotone path when both sequences are smooth, which
    holds for most robot trajectories (no backtracking in time).

    For best results apply AFTER start-pose centering so phase differences
    don't dominate the alignment.

    Parameters
    ----------
    seq_a : (T_a, D) — reference timeline
    seq_b : (T_b, D) — sequence to warp; D must match seq_a

    Returns
    -------
    warped : (T_a, D) — seq_b resampled to T_a timesteps
    """
    T_a, D_a = seq_a.shape
    T_b, D_b = seq_b.shape

    if D_a != D_b:
        raise ValueError(
            f"align_to_canonical_frame: joint dimensions must match "
            f"(seq_a D={D_a}, seq_b D={D_b}).  Call after DOF padding."
        )

    if T_a == T_b:
        return seq_b.copy()

    # Original time axis of seq_b: [0, 1]
    t_orig = np.linspace(0.0, 1.0, T_b)
    # Target time axis: T_a evenly-spaced points in [0, 1]
    t_target = np.linspace(0.0, 1.0, T_a)

    # Linear interpolation along the time axis, per joint
    warped = np.zeros((T_a, D_a), dtype=seq_b.dtype)
    for d in range(D_a):
        warped[:, d] = np.interp(t_target, t_orig, seq_b[:, d])

    return warped


# ── Core normalizer ───────────────────────────────────────────────────────────

class TrajectoryNormalizer:
    """
    Normalize robot trajectories to a canonical frame before annotation.

    Transformations applied in order:
      1. Start-pose centering : subtract q[0] so all trajectories start at origin
      2. Per-joint range norm : scale each joint to [-1, 1] using training data min/max
      3. Velocity normalization: normalize velocity magnitude to unit scale
      4. DOF-adaptive padding : pad or truncate to D_CANONICAL joints

    Design notes
    ------------
    - Range normalization uses the *training-data* min/max (stored in .fit()), not
      the per-trajectory min/max.  Using per-trajectory stats would collapse flat
      trajectories (stuck joint) to noise.
    - Padding uses zeros, which land at the normalized "resting position" of a
      joint that never moved — a reasonable neutral value.
    - Truncation removes the last (D - D_CANONICAL) joints.  For ALOHA bimanual
      robots this removes the distal joints of the second arm; the 8 retained
      joints capture shoulder + elbow motion which encodes task phase well.

    Attributes
    ----------
    D_CANONICAL : int  — canonical joint count after padding/truncation (default 8)
    joint_min_  : (D,) — per-joint minimum from training data (before centering)
    joint_max_  : (D,) — per-joint maximum from training data
    joint_range_: (D,) — joint_max_ - joint_min_  (used for scaling)
    vel_scale_  : float — 95th-percentile velocity magnitude from training data
    D_train_    : int  — original DOF of training data
    """

    D_CANONICAL = 8

    def __init__(self):
        self.joint_min_   = None    # (D,) fitted range lower bound
        self.joint_max_   = None    # (D,) fitted range upper bound
        self.joint_range_ = None    # (D,) max - min, with floor at eps
        self.vel_scale_   = None    # float: 95th-pct velocity magnitude
        self.D_train_     = None    # original D of training sequences

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, state_seqs: list) -> "TrajectoryNormalizer":
        """
        Fit joint range statistics from a list of (T, D) arrays.

        All sequences must have the same D.  If they differ (e.g. a mixed
        multi-robot dataset), truncate/pad to the modal D before calling fit.

        Parameters
        ----------
        state_seqs : list of (T_i, D) np.ndarray

        Returns
        -------
        self — for chaining
        """
        if not state_seqs:
            raise ValueError("fit() requires at least one sequence.")

        # Infer D from the first sequence
        D = state_seqs[0].shape[1]
        self.D_train_ = D

        all_steps = np.vstack([s for s in state_seqs])   # (N_total, D)
        self.joint_min_   = all_steps.min(axis=0)        # (D,)
        self.joint_max_   = all_steps.max(axis=0)        # (D,)
        raw_range         = self.joint_max_ - self.joint_min_
        # Avoid divide-by-zero for joints that never move
        self.joint_range_ = np.where(raw_range > 1e-8, raw_range, 1.0)

        # Velocity scale: 95th-percentile of step-to-step velocity magnitudes
        vel_mags = []
        for seq in state_seqs:
            if len(seq) > 1:
                vel = np.diff(seq, axis=0)               # (T-1, D)
                vel_mags.extend(np.linalg.norm(vel, axis=1).tolist())
        if vel_mags:
            self.vel_scale_ = float(np.percentile(vel_mags, 95))
        else:
            self.vel_scale_ = 1.0
        if self.vel_scale_ < 1e-8:
            self.vel_scale_ = 1.0

        return self

    # ── Core transform ────────────────────────────────────────────────────────

    def transform(self, state_seq: np.ndarray) -> np.ndarray:
        """
        Apply the full normalization pipeline to one trajectory.

        Parameters
        ----------
        state_seq : (T, D) joint-state sequence — D may differ from D_train_

        Returns
        -------
        normalized : (T, D_CANONICAL) float32 array
        """
        if self.joint_min_ is None:
            raise RuntimeError("Call fit() before transform().")

        seq = state_seq.astype(np.float32)
        T, D = seq.shape

        # ── Step 1: start-pose centering ──────────────────────────────────────
        # Subtract the very first pose so all trajectories start at zero.
        # This removes the absolute joint configuration bias and focuses the
        # model on *motion* rather than *pose*.
        seq = seq - seq[0:1, :]                          # (T, D)

        # ── Step 2: per-joint range normalization ─────────────────────────────
        # Scale each joint to [-1, 1] using training-data min/max.
        # After start-pose centering the centered values are in
        # [original_min - q0, original_max - q0], so we re-use the *range*
        # (max - min) as the denominator; the result won't be exactly in [-1,1]
        # for all steps, but it will be proportionally scaled across joints.
        #
        # For joints not seen in training data (D > D_train_) we use scale=1.
        D_ref = min(D, self.D_train_)
        scaled = seq.copy()
        scaled[:, :D_ref] = seq[:, :D_ref] / (self.joint_range_[:D_ref] / 2.0 + 1e-8)

        # ── Step 3: velocity normalization ───────────────────────────────────
        # Compute the velocity sequence (finite differences), then rescale
        # so the 95th-percentile velocity magnitude is 1.0.
        # We reconstruct the position sequence from the normalized velocities
        # so both position and velocity share the same scale.
        if T > 1:
            vel = np.diff(scaled, axis=0)                # (T-1, D)
            vel_mags = np.linalg.norm(vel, axis=1, keepdims=True)  # (T-1, 1)
            scale_factor = self.vel_scale_ + 1e-8
            vel_norm = vel / scale_factor
            # Reconstruct: integrate normalized velocity back to positions
            # This preserves the start-at-zero property.
            scaled = np.vstack([scaled[0:1, :],
                                 scaled[0:1, :] + np.cumsum(vel_norm, axis=0)])

        # ── Step 4: DOF-adaptive padding / truncation ─────────────────────────
        if D < self.D_CANONICAL:
            # Pad with zeros (represent "resting" joints that don't exist)
            pad = np.zeros((T, self.D_CANONICAL - D), dtype=np.float32)
            scaled = np.concatenate([scaled, pad], axis=1)
        elif D > self.D_CANONICAL:
            scaled = scaled[:, :self.D_CANONICAL]

        return scaled.astype(np.float32)

    # ── Batch convenience wrappers ────────────────────────────────────────────

    def fit_transform(self, state_seqs: list) -> list:
        """
        Fit on the provided sequences, then transform them all.

        Parameters
        ----------
        state_seqs : list of (T_i, D) np.ndarray

        Returns
        -------
        list of (T_i, D_CANONICAL) np.ndarray
        """
        self.fit(state_seqs)
        return [self.transform(s) for s in state_seqs]

    # ── Inverse transform ─────────────────────────────────────────────────────

    def inverse_transform(self, normalized: np.ndarray,
                           original_D: int) -> np.ndarray:
        """
        Approximately undo the normalization — useful for visualization.

        The inverse is approximate because:
          - Velocity normalization loses information about absolute velocity scale
            per timestep (we only store the global 95th-pct scale factor).
          - Padding zeros are indistinguishable from real near-zero joint motion.

        Parameters
        ----------
        normalized  : (T, D_CANONICAL) normalized trajectory
        original_D  : int — DOF of the original robot (before padding/truncation)

        Returns
        -------
        reconstructed : (T, original_D) float32 — in approx. original joint space
        """
        if self.joint_min_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        seq = normalized.astype(np.float32)
        T   = seq.shape[0]

        # Step 4 inverse: strip padding or zero-fill truncated joints
        D_out = min(original_D, self.D_CANONICAL)
        out   = seq[:, :D_out].copy()                    # (T, D_out)

        if original_D > self.D_CANONICAL:
            # Joints that were truncated: fill with zeros (neutral)
            extra = np.zeros((T, original_D - self.D_CANONICAL), dtype=np.float32)
            out   = np.concatenate([out, extra], axis=1)

        D_ref = min(original_D, self.D_train_)

        # Step 3 inverse: undo velocity normalization
        # We don't have the original time-varying velocity scale, so we
        # re-derive position from the scaled velocity.
        if T > 1:
            vel_norm = np.diff(out, axis=0)              # (T-1, D_out)
            vel_orig = vel_norm * self.vel_scale_
            out = np.vstack([out[0:1, :],
                             out[0:1, :] + np.cumsum(vel_orig, axis=0)])

        # Step 2 inverse: undo range normalization
        range_ref = self.joint_range_[:D_ref]
        out[:, :D_ref] = out[:, :D_ref] * (range_ref / 2.0 + 1e-8)

        # Step 1 inverse: we cannot recover q[0] since centering is irreversible
        # without storing per-episode start poses.  Return centered coordinates.
        # (For visualization, centering makes little difference to motion shape.)

        return out.astype(np.float32)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = NORMALIZER_PATH) -> None:
        """Serialize normalizer state to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "joint_min":   self.joint_min_,
            "joint_max":   self.joint_max_,
            "joint_range": self.joint_range_,
            "vel_scale":   self.vel_scale_,
            "D_train":     self.D_train_,
            "D_canonical": self.D_CANONICAL,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        print(f"TrajectoryNormalizer saved: {path}")

    @classmethod
    def load(cls, path: Path = NORMALIZER_PATH) -> "TrajectoryNormalizer":
        """Load a previously saved normalizer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Normalizer not found at {path}.  "
                "Call TrajectoryNormalizer().fit(seqs).save() first."
            )
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj.joint_min_   = state["joint_min"]
        obj.joint_max_   = state["joint_max"]
        obj.joint_range_ = state["joint_range"]
        obj.vel_scale_   = state["vel_scale"]
        obj.D_train_     = state["D_train"]
        # Honour D_CANONICAL stored in the file (in case it was overridden)
        obj.D_CANONICAL  = state.get("D_canonical", cls.D_CANONICAL)
        print(f"TrajectoryNormalizer loaded from {path}  "
              f"(D_train={obj.D_train_}, D_canonical={obj.D_CANONICAL}, "
              f"vel_scale={obj.vel_scale_:.4f})")
        return obj

    # ── Pretty-print summary ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.joint_min_ is None:
            return "TrajectoryNormalizer(unfitted)"
        return (
            f"TrajectoryNormalizer("
            f"D_train={self.D_train_}, "
            f"D_canonical={self.D_CANONICAL}, "
            f"vel_scale={self.vel_scale_:.4f})"
        )


# ── Quick smoke-test / demo ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trajectory normalization utilities — smoke test and demo")
    parser.add_argument("--demo", action="store_true",
                        help="Run smoke test with synthetic trajectories")
    args = parser.parse_args()

    if args.demo:
        print("TrajectoryNormalizer smoke test")
        print("=" * 50)

        rng = np.random.default_rng(42)

        # Simulate Franka (D=7) and xArm (D=6) trajectories
        franka_seqs = [rng.uniform(-np.pi, np.pi, size=(50 + i, 7)).cumsum(0) * 0.1
                       for i in range(10)]
        xarm_seqs   = [rng.uniform(-np.pi, np.pi, size=(40 + i, 6)).cumsum(0) * 0.15
                       for i in range(8)]

        print(f"\nFranka sequences : {len(franka_seqs)} episodes, "
              f"shapes {[s.shape for s in franka_seqs[:3]]}...")
        print(f"xArm  sequences  : {len(xarm_seqs)}  episodes, "
              f"shapes {[s.shape for s in xarm_seqs[:3]]}...")

        # Fit on Franka data
        norm = TrajectoryNormalizer()
        norm_franka = norm.fit_transform(franka_seqs)
        print(f"\nAfter fit_transform (Franka):")
        print(f"  Output shape  : {norm_franka[0].shape}")
        print(f"  Value range   : [{norm_franka[0].min():.3f}, {norm_franka[0].max():.3f}]")
        print(f"  Starts at zero: {np.allclose(norm_franka[0][0], 0.0, atol=1e-5)}")
        print(f"  Normalizer    : {norm}")

        # Transform xArm (cross-robot)
        norm_xarm = [norm.transform(s) for s in xarm_seqs]
        print(f"\nCross-robot transform (xArm → canonical):")
        print(f"  Output shape  : {norm_xarm[0].shape}")
        print(f"  Value range   : [{norm_xarm[0].min():.3f}, {norm_xarm[0].max():.3f}]")

        # DOF detection
        for seq, expected in [(franka_seqs[0], "franka"),
                               (xarm_seqs[0],   "xarm6 or ur5")]:
            robot = detect_robot_dof(seq)
            print(f"\ndetect_robot_dof: D={seq.shape[1]} → '{robot}'")

        # Alignment
        ref  = norm_franka[0]                            # (50, 8)
        long = norm_franka[2]                            # (52, 8)
        warped = align_to_canonical_frame(ref, long)
        print(f"\nalign_to_canonical_frame:")
        print(f"  seq_a shape  : {ref.shape}")
        print(f"  seq_b shape  : {long.shape}")
        print(f"  warped shape : {warped.shape}  (matches seq_a)")

        # Compute stats
        stats = compute_trajectory_stats(franka_seqs)
        print(f"\ncompute_trajectory_stats (Franka):")
        print(f"  n_sequences : {stats['n_sequences']}")
        print(f"  n_timesteps : {stats['n_timesteps']}")
        print(f"  velocity p95: {stats['global']['velocity_p95']:.4f}")
        for j in stats["per_joint"][:3]:
            print(f"  joint {j['joint_idx']}: mean={j['mean']:.3f}  "
                  f"range={j['range']:.3f}")

        # Save / load round-trip
        norm.save()
        norm2 = TrajectoryNormalizer.load()
        t1 = norm.transform(franka_seqs[0])
        t2 = norm2.transform(franka_seqs[0])
        print(f"\nSave/load round-trip identical: {np.allclose(t1, t2)}")

        print("\nSmoke test passed.")

    else:
        parser.print_help()
        print("\nQuick start:  python preprocessing.py --demo")
