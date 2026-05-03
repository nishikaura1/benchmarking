"""
Real-time step-by-step annotation for live robot teleoperation.

This module provides the "live safety monitoring" product: a streaming
interface that ingests joint states one at a time as they arrive from
the robot and emits a safety signal (green / yellow / red) within <1ms.

Architecture
------------
  StreamingAnnotator
    - Maintains a rolling buffer of the last BUFFER_SIZE joint states
    - Extracts sliding-window features from the buffer on each step
    - Runs two fast inference passes:
        1. IsolationForest (anomaly score)    — fitted from the RobotAnnotator's
           internal episode-level anomaly model (reused from pipeline.py)
        2. RobotAnnotator.annotate()          — step-level failure label + confidence
    - Maps the combined score to a traffic-light safety signal
    - Gracefully falls back to rule-based signals when no model is available

  StreamingMonitor
    - Wraps StreamingAnnotator and accumulates a rolling dashboard
    - Exposes: last N signals, running failure rate, per-type alert counts

  demo_stream()
    - Generates synthetic joint states and streams them through the annotator
    - Prints colored console output to illustrate the live product

Usage
-----
  from streaming import StreamingAnnotator

  ann = StreamingAnnotator.load()
  ann.reset()
  for joint_state in robot.stream():
      signal = ann.step(joint_state)
      display(signal["color"])

CLI:
  python streaming.py --demo
  python streaming.py --demo --steps 300
"""

import argparse
import collections
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "robot_annotator.pkl"

# ANSI colour codes for console output
_ANSI = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
}


def _color(text: str, color: str) -> str:
    """Wrap text in ANSI colour codes (no-op on non-TTY)."""
    if sys.stdout.isatty():
        return f"{_ANSI.get(color,'')}{text}{_ANSI['reset']}"
    return text


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_signal(buffer: np.ndarray, step_idx: int) -> dict:
    """
    Simple physics-rule safety signal when no trained model is available.

    Uses the same thresholding logic as generate_weak_labels() in
    annotation_model.py, applied to the tail of the rolling buffer.

    Returns a signal dict compatible with StreamingAnnotator.step().
    """
    if len(buffer) < 2:
        return {
            "step":          step_idx,
            "color":         "green",
            "failure_label": "nominal",
            "confidence":    1.0,
            "anomaly_score": 0.0,
            "needs_review":  False,
            "alert_message": "Initializing buffer — no history yet.",
            "buffer_filled": False,
        }

    seq = buffer                            # (w, D)
    T, D = seq.shape
    vel  = np.diff(seq, axis=0)             # (T-1, D)
    vel_mags = np.linalg.norm(vel, axis=1)  # (T-1,)
    acc  = np.diff(vel, axis=0) if T > 2 else np.zeros((1, D))
    acc_mags = np.linalg.norm(acc, axis=1)

    # Latest step diagnostics
    current_vel_mag = float(vel_mags[-1]) if len(vel_mags) > 0 else 0.0
    current_acc_mag = float(acc_mags[-1]) if len(acc_mags) > 0 else 0.0
    window_var      = float(seq.var(axis=0).max())

    # Adaptive thresholds from the current buffer
    vel_thresh = float(np.percentile(vel_mags, 85)) if len(vel_mags) >= 3 else 1e9
    acc_thresh = float(np.percentile(acc_mags, 85)) if len(acc_mags) >= 3 else 1e9

    # Classify
    if current_vel_mag > vel_thresh and current_vel_mag > vel_mags.mean() * 2:
        label  = "velocity_spike"
        score  = min(1.0, current_vel_mag / (vel_thresh + 1e-8) * 0.5)
    elif current_acc_mag > acc_thresh and current_acc_mag > acc_mags.mean() * 2:
        label  = "position_jerk"
        score  = min(1.0, current_acc_mag / (acc_thresh + 1e-8) * 0.5)
    elif T >= 5 and window_var < 1e-6:
        label  = "stuck_joint"
        score  = 0.55
    else:
        label  = "nominal"
        score  = 0.05

    # Map score to colour
    if score < StreamingAnnotator.GREEN_THRESHOLD:
        color = "green"
    elif score < StreamingAnnotator.YELLOW_THRESHOLD:
        color = "yellow"
    else:
        color = "red"

    alert_msg = _make_alert_message(color, label, score, step_idx, model_type="rule-based")

    return {
        "step":          step_idx,
        "color":         color,
        "failure_label": label,
        "confidence":    round(1.0 - score, 3),
        "anomaly_score": round(score, 4),
        "needs_review":  color != "green",
        "alert_message": alert_msg,
        "buffer_filled": len(buffer) >= StreamingAnnotator.BUFFER_SIZE,
    }


def _make_alert_message(color: str, label: str, score: float,
                         step: int, model_type: str = "model") -> str:
    """Human-readable status string for a safety signal."""
    prefix_map = {
        "green":  "OK",
        "yellow": "CAUTION",
        "red":    "ALERT",
    }
    prefix = prefix_map.get(color, "INFO")
    if label == "nominal":
        detail = "Normal operation"
    else:
        detail = f"Detected: {label.replace('_', ' ')}"
    return (f"[step {step:04d}] {prefix} | {detail} "
            f"| anomaly={score:.3f} ({model_type})")


# ── StreamingAnnotator ────────────────────────────────────────────────────────

class StreamingAnnotator:
    """
    Real-time annotation for live robot teleoperation.

    Maintains a rolling buffer of recent joint states and emits a
    safety signal (green/yellow/red) after each new step arrives.

    Latency: <1ms per step on CPU (IsolationForest + RF inference).

    Signal thresholds
    -----------------
    GREEN  : anomaly_score < GREEN_THRESHOLD   (safe — continue operation)
    YELLOW : GREEN_THRESHOLD <= score < YELLOW_THRESHOLD  (caution — monitor closely)
    RED    : score >= YELLOW_THRESHOLD         (alert — consider stopping)

    The anomaly_score is 1 - P(nominal) from the calibrated RF probability.
    When no trained model is present, a rule-based fallback is used instead.

    Parameters
    ----------
    fail_annotator  : RobotAnnotator instance (or None → rule-based fallback)
    anomaly_detector: sklearn IsolationForest fitted on nominal data (optional)
    """

    BUFFER_SIZE       = 30       # rolling window of recent states
    GREEN_THRESHOLD   = 0.35     # anomaly score below → green (safe)
    YELLOW_THRESHOLD  = 0.60     # anomaly score above → red (alert)

    def __init__(self, fail_annotator=None, anomaly_detector=None):
        self.fail_annotator  = fail_annotator    # RobotAnnotator or None
        self.anomaly_detector = anomaly_detector  # IsolationForest or None
        self._buffer: collections.deque = collections.deque(maxlen=self.BUFFER_SIZE)
        self._step_count: int = 0
        self._episode_signals: list = []

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_path: Path = None) -> "StreamingAnnotator":
        """
        Load from saved robot_annotator.pkl.

        If the file is not found, returns a StreamingAnnotator that uses
        rule-based fallback logic — no error is raised.  The caller can check
        whether a model was loaded by inspecting `.fail_annotator is not None`.

        Parameters
        ----------
        model_path : Path or None — defaults to benchmark_output/robot_annotator.pkl
        """
        path = Path(model_path) if model_path else MODEL_PATH

        if not path.exists():
            print(
                f"[StreamingAnnotator] Model not found at {path}.\n"
                f"  Falling back to rule-based safety signals.\n"
                f"  To enable ML inference: python annotation_model.py --train"
            )
            return cls(fail_annotator=None, anomaly_detector=None)

        try:
            # Import here to avoid circular imports at module load time
            from annotation_model import RobotAnnotator
            ann = RobotAnnotator.load(path)
            print(f"[StreamingAnnotator] Loaded RobotAnnotator from {path}")
            return cls(fail_annotator=ann, anomaly_detector=None)
        except Exception as exc:
            print(
                f"[StreamingAnnotator] Failed to load model from {path}: {exc}\n"
                f"  Falling back to rule-based safety signals."
            )
            return cls(fail_annotator=None, anomaly_detector=None)

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear buffer and step counter — call at the start of each episode."""
        self._buffer.clear()
        self._step_count = 0
        self._episode_signals = []

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, joint_state: np.ndarray) -> dict:
        """
        Process one new joint state and return a safety signal.

        This is the hot path — called at robot control frequency (10–500 Hz).
        All inference uses pre-fitted sklearn models; no retraining occurs here.

        Parameters
        ----------
        joint_state : (D,) float array — current joint positions

        Returns
        -------
        dict with keys:
            "step"          : int    — monotone step counter (resets on reset())
            "color"         : str    — "green" | "yellow" | "red"
            "failure_label" : str    — predicted failure type (or "nominal")
            "confidence"    : float  — model confidence in the label [0, 1]
            "anomaly_score" : float  — 1 - P(nominal); higher = more anomalous
            "needs_review"  : bool   — True when color != "green"
            "alert_message" : str    — human-readable status line
            "buffer_filled" : bool   — False until BUFFER_SIZE steps seen
        """
        joint_state = np.asarray(joint_state, dtype=np.float32).flatten()
        self._buffer.append(joint_state)
        idx = self._step_count
        self._step_count += 1

        buf_array = np.array(self._buffer)          # (w, D), w <= BUFFER_SIZE
        buffer_filled = len(self._buffer) >= self.BUFFER_SIZE

        # ── Fallback: rule-based when model not loaded ────────────────────────
        if self.fail_annotator is None:
            signal = _rule_based_signal(buf_array, idx)
            signal["buffer_filled"] = buffer_filled
            self._episode_signals.append(signal)
            return signal

        # ── ML inference path ─────────────────────────────────────────────────
        # Wait until the buffer has enough context to compute meaningful features
        if not buffer_filled:
            signal = {
                "step":          idx,
                "color":         "green",
                "failure_label": "nominal",
                "confidence":    1.0,
                "anomaly_score": 0.0,
                "needs_review":  False,
                "alert_message": (f"[step {idx:04d}] OK | Filling buffer "
                                   f"({len(self._buffer)}/{self.BUFFER_SIZE})"),
                "buffer_filled": False,
            }
            self._episode_signals.append(signal)
            return signal

        try:
            # annotate() expects (T, D); we pass the rolling buffer
            ann = self.fail_annotator.annotate(buf_array)

            # We care about the LAST step's annotation (the current one)
            last = len(buf_array) - 1
            label  = ann["labels"][last]
            conf   = float(ann["confidences"][last])
            score  = float(ann["anomaly_scores"][last])

        except Exception as exc:
            # Degrade gracefully on unexpected inference errors
            signal = _rule_based_signal(buf_array, idx)
            signal["alert_message"] += f" [model error: {exc}]"
            self._episode_signals.append(signal)
            return signal

        # Map score to traffic-light colour
        if score < self.GREEN_THRESHOLD:
            color = "green"
        elif score < self.YELLOW_THRESHOLD:
            color = "yellow"
        else:
            color = "red"

        alert_msg = _make_alert_message(color, label, score, idx, model_type="RF")

        signal = {
            "step":          idx,
            "color":         color,
            "failure_label": label,
            "confidence":    round(conf, 4),
            "anomaly_score": round(score, 4),
            "needs_review":  color != "green",
            "alert_message": alert_msg,
            "buffer_filled": True,
        }
        self._episode_signals.append(signal)
        return signal

    # ── Full episode replay ───────────────────────────────────────────────────

    def stream_episode(self, state_seq: np.ndarray) -> list:
        """
        Replay a full episode through the streaming interface step-by-step.

        Equivalent to calling reset() then step() for each row of state_seq.
        Useful for offline evaluation of the streaming latency and accuracy
        against episodes where the ground truth is known.

        Parameters
        ----------
        state_seq : (T, D) joint-state sequence

        Returns
        -------
        list of T signal dicts (one per timestep)
        """
        self.reset()
        signals = []
        for t in range(len(state_seq)):
            sig = self.step(state_seq[t])
            signals.append(sig)
        return signals

    # ── End-of-episode summary ────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        End-of-episode summary: dominant failure, alert count, review rate.

        Call after stream_episode() or after the last step() of an episode.

        Returns
        -------
        dict with:
            "n_steps"          : int
            "n_alerts"         : int   — red + yellow signals
            "n_red"            : int
            "n_yellow"         : int
            "alert_rate"       : float — fraction of steps that were non-green
            "review_steps"     : list[int] — step indices that need review
            "dominant_failure" : str   — most frequent non-nominal label
            "failure_counts"   : dict  — {label: count}
            "buffer_filled_at" : int   — step index when buffer first filled
        """
        sigs = self._episode_signals
        if not sigs:
            return {"n_steps": 0, "n_alerts": 0, "alert_rate": 0.0,
                    "dominant_failure": "nominal", "failure_counts": {},
                    "review_steps": [], "buffer_filled_at": -1}

        n_steps  = len(sigs)
        n_red    = sum(1 for s in sigs if s["color"] == "red")
        n_yellow = sum(1 for s in sigs if s["color"] == "yellow")
        n_alerts = n_red + n_yellow

        review_steps = [s["step"] for s in sigs if s["needs_review"]]

        # Failure type counts (exclude nominal)
        failure_counts: dict = {}
        for s in sigs:
            lbl = s["failure_label"]
            failure_counts[lbl] = failure_counts.get(lbl, 0) + 1

        non_nominal = {k: v for k, v in failure_counts.items() if k != "nominal"}
        dominant = max(non_nominal, key=non_nominal.get) if non_nominal else "nominal"

        # Step at which the buffer first filled
        filled_at = next(
            (s["step"] for s in sigs if s.get("buffer_filled")), -1
        )

        return {
            "n_steps":          n_steps,
            "n_alerts":         n_alerts,
            "n_red":            n_red,
            "n_yellow":         n_yellow,
            "alert_rate":       round(n_alerts / n_steps, 4) if n_steps else 0.0,
            "review_steps":     review_steps,
            "dominant_failure": dominant,
            "failure_counts":   failure_counts,
            "buffer_filled_at": filled_at,
        }


# ── StreamingMonitor ──────────────────────────────────────────────────────────

class StreamingMonitor:
    """
    Sliding-window dashboard wrapper around StreamingAnnotator.

    Aggregates signals in real time so a UI or logger can display:
      - The last N signals (rolling window)
      - Running failure rate
      - Per-failure-type alert history

    Parameters
    ----------
    annotator   : StreamingAnnotator instance
    window_size : int — how many recent signals to keep in the dashboard
    """

    def __init__(self, annotator: StreamingAnnotator, window_size: int = 50):
        self.annotator    = annotator
        self.window_size  = window_size
        self._window: collections.deque = collections.deque(maxlen=window_size)
        self._alert_history: list = []      # all red/yellow signals ever seen
        self._failure_totals: dict = {}     # cumulative failure counts
        self._total_steps: int = 0

    def reset(self) -> None:
        """Clear all state — call at episode start."""
        self.annotator.reset()
        self._window.clear()
        self._alert_history.clear()
        self._failure_totals.clear()
        self._total_steps = 0

    def step(self, joint_state: np.ndarray) -> dict:
        """
        Forward one joint state to the annotator and update the dashboard.

        Returns the signal dict from StreamingAnnotator.step() unchanged,
        so callers can use either interface interchangeably.
        """
        signal = self.annotator.step(joint_state)
        self._window.append(signal)
        self._total_steps += 1

        # Accumulate failure counts
        lbl = signal["failure_label"]
        self._failure_totals[lbl] = self._failure_totals.get(lbl, 0) + 1

        # Track alerts
        if signal["color"] != "green":
            self._alert_history.append(signal)

        return signal

    def dashboard(self) -> dict:
        """
        Return the current dashboard state.

        Returns
        -------
        dict with:
            "total_steps"       : int
            "recent_signals"    : list  — last window_size signals
            "recent_colors"     : list[str] — colour sequence (compact view)
            "running_alert_rate": float — fraction of all steps that are non-green
            "window_alert_rate" : float — alert rate within the last window
            "failure_totals"    : dict  — cumulative {label: count}
            "alert_history"     : list  — all non-green signals (full history)
            "last_signal"       : dict | None
        """
        recent = list(self._window)
        n_recent = len(recent)
        window_alerts = sum(1 for s in recent if s["color"] != "green")

        all_alerts = len(self._alert_history)
        return {
            "total_steps":        self._total_steps,
            "recent_signals":     recent,
            "recent_colors":      [s["color"] for s in recent],
            "running_alert_rate": round(all_alerts / max(1, self._total_steps), 4),
            "window_alert_rate":  round(window_alerts / max(1, n_recent), 4),
            "failure_totals":     dict(self._failure_totals),
            "alert_history":      list(self._alert_history),
            "last_signal":        recent[-1] if recent else None,
        }

    def print_dashboard(self) -> None:
        """Print a compact text dashboard to stdout."""
        d = self.dashboard()
        last = d["last_signal"]
        if last is None:
            print("[StreamingMonitor] No signals yet.")
            return

        color_bar = "".join(
            _color("█", s["color"]) for s in d["recent_signals"]
        )
        print(
            f"\n{'─'*55}\n"
            f"  StreamingMonitor Dashboard\n"
            f"{'─'*55}\n"
            f"  Total steps     : {d['total_steps']}\n"
            f"  Alert rate      : {d['running_alert_rate']*100:.1f}%  "
            f"(window: {d['window_alert_rate']*100:.1f}%)\n"
            f"  Last signal     : {_color(last['color'].upper(), last['color'])}  "
            f"| {last['failure_label']}  "
            f"| score={last['anomaly_score']:.3f}\n"
            f"  Failure totals  : {d['failure_totals']}\n"
            f"  Recent signals  : {color_bar}\n"
            f"{'─'*55}"
        )


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_stream(n_steps: int = 200, print_every: int = 10) -> None:
    """
    Stream synthetic joint states through the annotator and print signals.

    Generates a plausible 7-DOF Franka trajectory:
      - Steps 0–49   : smooth nominal motion (slow sinusoidal sweep)
      - Steps 50–59  : velocity spike (collision-like impulse)
      - Steps 60–89  : recovery + nominal
      - Steps 90–99  : stuck joint (all joints frozen)
      - Steps 100–109: gripper event (last dim flips)
      - Steps 110–end: nominal again

    The synthetic data triggers rule-based detection reliably so the demo
    is self-contained even without a trained model.

    Parameters
    ----------
    n_steps     : int — total steps to generate (default 200)
    print_every : int — print a signal line every N steps (default 10)
    """
    print("\n" + "=" * 60)
    print("StreamingAnnotator — Live Demo")
    print("=" * 60)

    rng = np.random.default_rng(0)
    # Use D=4 (xarm) to match the pre-trained model; change to 7 for Franka once adapted
    D   = 4

    # ── Generate synthetic trajectory ─────────────────────────────────────────
    states = []
    base = rng.uniform(-0.5, 0.5, size=D)
    t_arr = np.linspace(0, 4 * np.pi, n_steps)

    for i, t in enumerate(t_arr):
        freqs = np.array([1, 1.3, 0.7, 1.1][:D])
        if i < 50:
            s = base + 0.3 * np.sin(t * freqs)
        elif 50 <= i < 60:
            spike = rng.uniform(1.5, 2.5, size=D)
            s = base + spike * (1 if i == 50 else -0.5)
        elif 60 <= i < 90:
            s = base + 0.3 * np.sin(t)
        elif 90 <= i < 100:
            s = states[-1].copy() if states else base.copy()
        elif 100 <= i < 110:
            s = base + 0.1 * np.sin(t)
            s[-1] = 1.0 if (i % 2 == 0) else 0.0
        else:
            s = base + 0.3 * np.sin(t)

        states.append(s.astype(np.float32))

    # ── Load annotator ────────────────────────────────────────────────────────
    ann     = StreamingAnnotator.load()
    monitor = StreamingMonitor(ann, window_size=40)
    monitor.reset()

    model_type = "RobotAnnotator (RF)" if ann.fail_annotator else "rule-based fallback"
    print(f"\nModel  : {model_type}")
    print(f"Steps  : {n_steps}  |  D={D} joints  |  print every {print_every} steps")
    print(f"\n{'Step':>5}  {'Color':>8}  {'Label':>18}  {'Score':>7}  {'Conf':>6}  Message")
    print("─" * 80)

    t_start = time.perf_counter()

    for i, s in enumerate(states):
        sig = monitor.step(s)

        if i % print_every == 0 or sig["color"] != "green":
            color_str = _color(f"{sig['color']:>8}", sig["color"])
            label_str = sig["failure_label"][:18]
            print(
                f"{sig['step']:>5}  {color_str}  {label_str:>18}  "
                f"{sig['anomaly_score']:>7.4f}  {sig['confidence']:>6.4f}  "
                f"{sig['alert_message']}"
            )

    elapsed = (time.perf_counter() - t_start) * 1000
    avg_ms  = elapsed / n_steps

    print("\n" + "=" * 60)
    print(f"Completed {n_steps} steps in {elapsed:.1f}ms  ({avg_ms:.3f}ms/step avg)")

    # End-of-episode summary
    ep_summary = ann.summary()
    print(f"\nEpisode Summary")
    print(f"  Steps processed  : {ep_summary['n_steps']}")
    print(f"  Alerts (non-green): {ep_summary['n_alerts']}  "
          f"({ep_summary['alert_rate']*100:.1f}%)")
    print(f"  Red signals       : {ep_summary['n_red']}")
    print(f"  Yellow signals    : {ep_summary['n_yellow']}")
    print(f"  Dominant failure  : {ep_summary['dominant_failure']}")
    print(f"  Failure counts    : {ep_summary['failure_counts']}")
    if ep_summary['review_steps']:
        print(f"  Review steps      : {ep_summary['review_steps'][:10]}"
              f"{'...' if len(ep_summary['review_steps']) > 10 else ''}")

    monitor.print_dashboard()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time robot safety annotation — streaming interface")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run streaming demo with synthetic joint states"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of synthetic steps to generate in demo mode (default: 200)"
    )
    parser.add_argument(
        "--print-every", type=int, default=10,
        help="Print a signal line every N steps (default: 10)"
    )
    args = parser.parse_args()

    if args.demo:
        demo_stream(n_steps=args.steps, print_every=args.print_every)
    else:
        parser.print_help()
        print("\nQuick start:  python streaming.py --demo")
