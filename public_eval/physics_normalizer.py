"""
public_eval/physics_normalizer.py
====================================
Change 1 — Per-dataset z-score normalization (RobotDataNormalizer)
Change 2 — Physics-based pre-filter before ML model (PhysicsPreFilter)
Change 3 — Combined physics + ML prediction (predict_episode)

Motivation (from cross-dataset evaluation):
  Cross-dataset transfer dropped 0.35–0.58 F1 points.
  Root cause: absolute sensor values are robot-specific.
  A velocity of 1 rad/s means completely different things on a UR5 vs xArm vs
  Orangewood arm. Without per-dataset z-score normalization the model learns
  sensor magnitudes, not behavioral patterns.

Physics filter basis:
  - Velocity spike: |v| > mean + 3σ  → Coulomb collision / loss of control
  - Stuck joint: |v| < threshold for N consecutive steps + commanded motion
  - Trajectory deviation: accumulated |Δpos| integral > threshold
  - Grasp slip: rapid drop in gripper force proxy (Coulomb friction model)

Usage:
    from public_eval.physics_normalizer import (
        RobotDataNormalizer, PhysicsPreFilter, predict_episode,
        extract_physics_features
    )
"""

import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Change 1 — Per-dataset z-score normalization
# ─────────────────────────────────────────────────────────────────────────────

class RobotDataNormalizer:
    """
    Normalizes robot sensor features relative to each dataset's own distribution.
    Critical for cross-robot generalization — a velocity of 1 rad/s means
    completely different things on a UR5 vs an xArm vs an Orangewood arm.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on training data and transform."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit_transform first.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def fit_transform_new_client(self, X_client: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
        """
        For a new client robot with unknown distribution.
        Fit a fresh scaler on their data only.
        This is the client adapter normalization step.
        Returns (X_scaled, client_scaler) so the scaler can be saved per-customer.
        """
        X_client = np.nan_to_num(X_client, nan=0.0, posinf=0.0, neginf=0.0)
        client_scaler = StandardScaler()
        return client_scaler.fit_transform(X_client), client_scaler


# ─────────────────────────────────────────────────────────────────────────────
# Change 2 — Physics-based pre-filter before ML model
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsPreFilter:
    """
    Applies physics-based heuristics before ML model.
    These are not ML predictions — they are physics constraints.
    If a physics constraint is violated we flag it regardless of ML confidence.

    Based on:
    - Coulomb friction model for grasp slip detection
    - Velocity limit violations for velocity spike detection
    - Position error accumulation for trajectory deviation
    - Torque-velocity relationship for stuck joint detection
    """

    def __init__(self, robot_profile: dict | None = None):
        # Default thresholds — override with robot-specific values
        self.vel_spike_sigma     = 3.0   # z-scores above mean
        self.stuck_joint_threshold = 0.05  # rad/s minimum movement
        self.stuck_joint_window  = 10    # timesteps
        self.traj_error_integral = 0.15  # rad*s accumulated error
        self.grip_force_drop_pct = 0.4   # 40% drop in grip force proxy

        if robot_profile:
            self.load_robot_profile(robot_profile)

    def load_robot_profile(self, profile: dict) -> None:
        """Load robot-specific thresholds from hardware profile dict."""
        self.vel_spike_sigma       = profile.get("vel_spike_sigma",       self.vel_spike_sigma)
        self.stuck_joint_threshold = profile.get("stuck_threshold",       self.stuck_joint_threshold)
        self.stuck_joint_window    = profile.get("stuck_window",          self.stuck_joint_window)
        self.traj_error_integral   = profile.get("traj_error_integral",   self.traj_error_integral)
        self.grip_force_drop_pct   = profile.get("grip_force_drop_pct",   self.grip_force_drop_pct)

    def detect_velocity_spike(self, velocities: np.ndarray) -> tuple[bool, int | None]:
        """
        Physics basis: instantaneous velocity exceeds 3 sigma above episode mean.
        Indicates loss of control or collision response.
        """
        velocities = np.asarray(velocities, dtype=np.float32).flatten()
        if len(velocities) < 3:
            return False, None
        abs_v  = np.abs(velocities)
        mean_v = abs_v.mean()
        std_v  = abs_v.std()
        if std_v < 1e-8:
            return False, None
        z_scores = (abs_v - mean_v) / std_v
        spike_ts = np.where(z_scores > self.vel_spike_sigma)[0]
        if len(spike_ts) > 0:
            return True, int(spike_ts[0])
        return False, None

    def detect_stuck_joint(
        self,
        velocities: np.ndarray,
        commanded_velocities: np.ndarray | None = None,
    ) -> tuple[bool, int | None]:
        """
        Physics basis: joint velocity near zero despite commanded motion.
        Indicates mechanical obstruction or motor stall.
        """
        velocities = np.asarray(velocities, dtype=np.float32).flatten()
        W = self.stuck_joint_window
        for i in range(len(velocities) - W):
            window = velocities[i : i + W]
            if np.all(np.abs(window) < self.stuck_joint_threshold):
                if commanded_velocities is not None:
                    cmd = np.asarray(commanded_velocities, dtype=np.float32).flatten()
                    cmd_window = cmd[i : i + W] if len(cmd) >= i + W else cmd[i:]
                    if len(cmd_window) and np.mean(np.abs(cmd_window)) > self.stuck_joint_threshold * 2:
                        return True, i
                else:
                    return True, i
        return False, None

    def detect_trajectory_deviation(
        self,
        actual_positions: np.ndarray,
        reference_positions: np.ndarray | None = None,
    ) -> tuple[bool, int | None]:
        """
        Physics basis: accumulated position error exceeds threshold.
        Indicates controller failure or external disturbance.
        """
        actual = np.asarray(actual_positions, dtype=np.float32).flatten()
        if len(actual) < 5:
            return False, None

        if reference_positions is None:
            # Without reference use Savitzky-Golay smoothed trajectory as proxy
            try:
                from scipy.signal import savgol_filter  # type: ignore
                wl = min(11, len(actual) // 2 * 2 - 1)
                if wl >= 3:
                    reference = savgol_filter(actual, wl, min(3, wl - 1))
                else:
                    reference = actual.copy()
            except Exception:
                # Fall back to moving average if scipy unavailable
                k = min(5, len(actual))
                reference = np.convolve(actual, np.ones(k) / k, mode="same")
        else:
            reference = np.asarray(reference_positions, dtype=np.float32).flatten()
            if len(reference) != len(actual):
                reference = actual.copy()

        errors = np.abs(actual - reference)
        error_integral = float(np.trapz(errors)) / len(errors)

        if error_integral > self.traj_error_integral:
            return True, int(np.argmax(errors))
        return False, None

    def detect_grasp_slip(self, grip_force_proxy: np.ndarray) -> tuple[bool, int | None]:
        """
        Physics basis: Coulomb friction model.
        Sudden drop in grip force proxy indicates slip initiation.
        F_tangential > mu * F_normal => slip
        We detect this as rapid grip force decrease.
        """
        g = np.asarray(grip_force_proxy, dtype=np.float32).flatten()
        if len(g) < 5:
            return False, None
        for i in range(1, len(g)):
            if g[i - 1] > 0.1:  # only when actively gripping
                drop_pct = (g[i - 1] - g[i]) / g[i - 1]
                if drop_pct > self.grip_force_drop_pct:
                    return True, i
        return False, None

    def run_all_checks(self, episode_features: dict) -> dict:
        """
        Run all physics checks on an episode.
        Returns dict of detected failures with timesteps.
        Any physics-confirmed failure overrides ML confidence threshold.

        episode_features keys (all optional — unrecognized keys silently ignored):
          velocities          np.ndarray  (T,) or (T, D) — joint velocities
          positions           np.ndarray  (T,) or (T, D) — joint positions
          grip_force_proxy    np.ndarray  (T,)            — gripper channel
          commanded_velocities np.ndarray (T,) or (T, D) — commanded vels
        """
        results: dict = {
            "physics_flags": [],
            "physics_failure_timestep": None,
            "physics_confirmed": False,
        }

        def _flatten_first(arr):
            """Use first column if multi-dim, else keep as-is."""
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                return arr[:, 0]
            return arr

        velocities  = _flatten_first(episode_features.get("velocities",        np.array([])))
        positions   = _flatten_first(episode_features.get("positions",         np.array([])))
        grip_force  = _flatten_first(episode_features.get("grip_force_proxy",  np.array([])))
        cmd_vel     = episode_features.get("commanded_velocities", None)
        if cmd_vel is not None:
            cmd_vel = _flatten_first(cmd_vel)

        checks = [
            ("velocity_spike",       self.detect_velocity_spike(velocities)),
            ("stuck_joint",          self.detect_stuck_joint(velocities, cmd_vel)),
            ("trajectory_deviation", self.detect_trajectory_deviation(positions)),
            ("grasp_slip",           self.detect_grasp_slip(grip_force)),
        ]

        for failure_type, (detected, timestep) in checks:
            if detected:
                results["physics_flags"].append({"type": failure_type, "timestep": timestep})
                if results["physics_failure_timestep"] is None:
                    results["physics_failure_timestep"] = timestep
                results["physics_confirmed"] = True

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build physics features dict from a schema episode
# ─────────────────────────────────────────────────────────────────────────────

def episode_to_physics_features(ep: dict) -> dict:
    """
    Map a schema episode dict to the format expected by PhysicsPreFilter.run_all_checks.

    Heuristics for column assignment (no ground-truth metadata):
      - velocities:       np.diff(state_seq, axis=0)  — first-order differencing
      - positions:        state_seq[:, 0]              — first state channel
      - grip_force_proxy: state_seq[:, -1]             — last channel (often gripper)
      - commanded_velocities: action_seq[:, 0] if available
    """
    states = np.asarray(ep["state_seq"], dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    T, D = states.shape

    velocities = np.diff(states, axis=0) if T > 1 else np.zeros((1, D), dtype=np.float32)
    positions  = states

    grip_force_proxy = states[:, -1] if D >= 1 else np.zeros(T)

    feats = {
        "velocities":       velocities,
        "positions":        positions,
        "grip_force_proxy": grip_force_proxy,
    }

    if ep.get("action_seq") is not None:
        actions = np.asarray(ep["action_seq"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        feats["commanded_velocities"] = actions

    return feats


def extract_physics_features(ep: dict, physics_filter: "PhysicsPreFilter") -> np.ndarray:
    """
    Run PhysicsPreFilter on one episode and return a 5-element binary/continuous
    feature vector to append to the statistical feature vector:

      [vel_spike_flag, stuck_joint_flag, traj_deviation_flag,
       grasp_slip_flag, physics_failure_step_norm]

    physics_failure_step_norm = failure_step / n_timesteps ∈ [0, 1]
    (0 if no physics failure detected)
    """
    feats = episode_to_physics_features(ep)
    results = physics_filter.run_all_checks(feats)

    flags_map = {
        "velocity_spike":       0,
        "stuck_joint":          0,
        "trajectory_deviation": 0,
        "grasp_slip":           0,
    }
    for flag in results["physics_flags"]:
        if flag["type"] in flags_map:
            flags_map[flag["type"]] = 1

    T = ep["timesteps"] or 1
    failure_step_norm = 0.0
    if results["physics_failure_timestep"] is not None:
        failure_step_norm = results["physics_failure_timestep"] / T

    return np.array([
        flags_map["velocity_spike"],
        flags_map["stuck_joint"],
        flags_map["trajectory_deviation"],
        flags_map["grasp_slip"],
        failure_step_norm,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Change 3 — Combine physics pre-filter with ML model
# ─────────────────────────────────────────────────────────────────────────────

def predict_episode(
    episode_features: dict,
    ml_model,
    normalizer: "RobotDataNormalizer",
    physics_filter: "PhysicsPreFilter",
    extract_features_fn,
) -> dict:
    """
    Two-stage prediction:
    1. Physics pre-filter — hard rules based on physics constraints
    2. ML model — learned patterns from training data

    If physics confirms a failure, trust it regardless of ML confidence.
    If physics is silent, use ML confidence with 0.75 threshold.

    Args:
        episode_features: schema episode dict
        ml_model:         fitted sklearn classifier with predict_proba
        normalizer:       fitted RobotDataNormalizer
        physics_filter:   PhysicsPreFilter instance
        extract_features_fn: callable(ep) -> np.ndarray (base features)

    Returns dict:
        failure_class, use_for_policy, confidence, failure_timestep,
        source ('physics_confirmed' | 'ml_model' | 'uncertain'), needs_review
    """
    # Stage 1: Physics check
    physics_results = physics_filter.run_all_checks(
        episode_to_physics_features(episode_features))

    # Stage 2: ML prediction
    try:
        X_base = extract_features_fn(episode_features)
        if X_base is None or not np.isfinite(X_base).all():
            raise ValueError("Feature extraction failed")
        phys_feat = extract_physics_features(episode_features, physics_filter)
        X = np.concatenate([X_base, phys_feat]).reshape(1, -1)
        X_norm = normalizer.transform(X)
        ml_proba   = ml_model.predict_proba(X_norm)[0]
        ml_class   = ml_model.classes_[int(ml_proba.argmax())]
        ml_confidence = float(ml_proba.max())
    except Exception as e:
        # ML failed — fall back to physics only
        if physics_results["physics_confirmed"]:
            return {
                "failure_class":     physics_results["physics_flags"][0]["type"],
                "use_for_policy":    False,
                "confidence":        0.75,
                "failure_timestep":  physics_results["physics_failure_timestep"],
                "source":            "physics_only_ml_failed",
                "needs_review":      True,
                "ml_error":          str(e),
            }
        return {
            "failure_class":     None,
            "use_for_policy":    None,
            "confidence":        0.0,
            "failure_timestep":  None,
            "source":            "error",
            "needs_review":      True,
            "ml_error":          str(e),
        }

    # Stage 3: Combine
    if physics_results["physics_confirmed"]:
        primary_failure = physics_results["physics_flags"][0]["type"]
        failure_timestep = physics_results["physics_failure_timestep"]

        # ML refines the classification if it agrees with physics
        if ml_class == primary_failure and ml_confidence > 0.6:
            final_confidence = min(0.95, (ml_confidence + 0.85) / 2)
        else:
            final_confidence = 0.80  # Physics confirmed — trust physics

        return {
            "failure_class":     primary_failure,
            "use_for_policy":    False,
            "confidence":        final_confidence,
            "failure_timestep":  failure_timestep,
            "source":            "physics_confirmed",
            "needs_review":      False,
        }

    elif ml_confidence >= 0.75:
        # ML is confident, physics found nothing obvious
        return {
            "failure_class":     ml_class if ml_class != "nominal" else None,
            "use_for_policy":    ml_class == "nominal",
            "confidence":        ml_confidence,
            "failure_timestep":  None,
            "source":            "ml_model",
            "needs_review":      False,
        }

    else:
        # Neither is confident — send to human review
        return {
            "failure_class":     ml_class,
            "use_for_policy":    None,
            "confidence":        ml_confidence,
            "failure_timestep":  physics_results.get("physics_failure_timestep"),
            "source":            "uncertain",
            "needs_review":      True,
        }
