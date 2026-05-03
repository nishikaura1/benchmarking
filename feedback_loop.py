"""
feedback_loop.py — Automatic human-correction retraining pipeline.

Every time a human corrects a label it flows into a queue.
When the queue hits the threshold (default 50 corrections), a retrain
is triggered automatically and the model version is bumped.

Without this loop the model is static — you're just an annotation service.
With this loop every human correction makes the model better.

Usage
-----
# Log a correction from the review UI / API:
from feedback_loop import on_human_correction
on_human_correction("ep_0042", step=17,
                    original_label="nominal",
                    corrected_label="velocity_spike",
                    reviewer_id="reviewer_01")

# Run the weekly check manually (or via cron):
from feedback_loop import weekly_retrain
weekly_retrain()

# Or trigger immediately regardless of queue size:
from feedback_loop import force_retrain
force_retrain()
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

OUTPUT_DIR    = Path("benchmark_output")
QUEUE_PATH    = OUTPUT_DIR / "retraining_queue.json"
CORRECTIONS_PATH = OUTPUT_DIR / "corrections.json"
VERSION_PATH  = OUTPUT_DIR / "model_version.json"

RETRAIN_THRESHOLD = 50   # trigger a retrain after this many queued corrections

OUTPUT_DIR.mkdir(exist_ok=True)


# ── Queue helpers ─────────────────────────────────────────────────────────────

def _load_queue() -> list:
    if QUEUE_PATH.exists():
        return json.loads(QUEUE_PATH.read_text())
    return []


def _save_queue(queue: list):
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def _load_corrections() -> list:
    if CORRECTIONS_PATH.exists():
        return json.loads(CORRECTIONS_PATH.read_text())
    return []


def _save_corrections(corrections: list):
    CORRECTIONS_PATH.write_text(json.dumps(corrections, indent=2))


# ── Model version tracking ────────────────────────────────────────────────────

def get_model_version() -> dict:
    if VERSION_PATH.exists():
        return json.loads(VERSION_PATH.read_text())
    return {"version": "1.0.0", "trained_at": None, "total_corrections": 0, "history": []}


def _bump_version(corrections_used: int) -> str:
    """Bump minor version on each retrain. Major version = manual bump."""
    v = get_model_version()
    parts = v["version"].split(".")
    parts[1] = str(int(parts[1]) + 1)
    new_ver  = ".".join(parts)
    now      = datetime.utcnow().isoformat()
    v["history"].append({
        "version":          new_ver,
        "trained_at":       now,
        "corrections_used": corrections_used,
    })
    v["version"]           = new_ver
    v["trained_at"]        = now
    v["total_corrections"] = v.get("total_corrections", 0) + corrections_used
    VERSION_PATH.write_text(json.dumps(v, indent=2))
    return new_ver


# ── Public API ────────────────────────────────────────────────────────────────

def on_human_correction(
    episode_id:      str,
    step:            int,
    original_label:  str,
    corrected_label: str,
    reviewer_id:     str     = "anonymous",
    confidence:      float   = 1.0,
    features:        Optional[list] = None,
) -> dict:
    """
    Record a human label correction and add it to the retraining queue.

    Parameters
    ----------
    episode_id      : unique episode identifier
    step            : timestep index within the episode
    original_label  : what the model predicted
    corrected_label : what the human says it actually is
    reviewer_id     : who made the correction (for auditing)
    confidence      : reviewer's confidence in their label (0-1)
    features        : raw feature vector for this step (stored for retrain injection)

    Returns
    -------
    dict with status, queue size, and whether a retrain was triggered
    """
    now = datetime.utcnow().isoformat()

    entry = {
        "id":               hashlib.md5(f"{episode_id}:{step}:{now}".encode()).hexdigest()[:8],
        "episode_id":       episode_id,
        "step":             step,
        "original_label":   original_label,
        "corrected_label":  corrected_label,
        "reviewer_id":      reviewer_id,
        "confidence":       confidence,
        "timestamp":        now,
        "used_in_retrain":  False,
    }
    if features is not None:
        entry["features"] = features   # stored for direct injection into training

    # Write to corrections.json (used by _load_human_corrections in annotation_model)
    corrections = _load_corrections()
    corrections.append({
        "episode_id":       episode_id,
        "step":             step,
        "original_label":   original_label,
        "corrected_label":  corrected_label,
        "reviewer_id":      reviewer_id,
        "timestamp":        now,
    })
    _save_corrections(corrections)

    # Write to retraining queue (tracked separately for threshold logic)
    queue = _load_queue()
    queue.append(entry)
    _save_queue(queue)

    print(f"  [feedback] Correction recorded: ep={episode_id} step={step} "
          f"{original_label!r} → {corrected_label!r}  "
          f"(queue: {len(queue)}/{RETRAIN_THRESHOLD})")

    # Auto-trigger retrain if queue is full
    retrain_triggered = False
    if len(queue) >= RETRAIN_THRESHOLD:
        print(f"  [feedback] Queue threshold reached ({len(queue)} corrections) — triggering retrain")
        retrain_triggered = True
        weekly_retrain(force=True)

    return {
        "status":            "recorded",
        "queue_size":        len(_load_queue()),
        "retrain_triggered": retrain_triggered,
        "entry_id":          entry["id"],
    }


def add_bulk_corrections(corrections: list) -> int:
    """
    Add multiple corrections at once (e.g. from a batch review session).

    corrections : list of dicts with keys:
        episode_id, step, original_label, corrected_label, reviewer_id
    Returns number of corrections added.
    """
    added = 0
    for c in corrections:
        on_human_correction(
            episode_id=c["episode_id"],
            step=c.get("step", 0),
            original_label=c.get("original_label", "nominal"),
            corrected_label=c["corrected_label"],
            reviewer_id=c.get("reviewer_id", "batch"),
            features=c.get("features"),
        )
        added += 1
    return added


def get_queue_status() -> dict:
    """Return current queue size and progress toward next retrain."""
    queue    = _load_queue()
    pending  = [e for e in queue if not e.get("used_in_retrain")]
    version  = get_model_version()
    return {
        "queue_size":          len(pending),
        "threshold":           RETRAIN_THRESHOLD,
        "pct_to_retrain":      round(len(pending) / RETRAIN_THRESHOLD * 100, 1),
        "total_corrections":   version.get("total_corrections", 0),
        "current_version":     version["version"],
        "last_trained":        version.get("trained_at"),
        "corrections_by_class": _count_by_class(pending),
    }


def _count_by_class(queue: list) -> dict:
    from collections import Counter
    return dict(Counter(e["corrected_label"] for e in queue))


def weekly_retrain(
    force: bool = False,
    dataset_names: Optional[list] = None,
) -> dict:
    """
    Check the retraining queue and retrain the model if the threshold is met.

    Parameters
    ----------
    force          : retrain even if below threshold (for manual triggers)
    dataset_names  : datasets to train on (defaults to last run's datasets)

    Returns
    -------
    dict with retrain status and new model version
    """
    queue   = _load_queue()
    pending = [e for e in queue if not e.get("used_in_retrain")]

    if not force and len(pending) < RETRAIN_THRESHOLD:
        msg = (f"  [feedback] Queue has {len(pending)}/{RETRAIN_THRESHOLD} corrections — "
               f"retrain not triggered yet")
        print(msg)
        return {"status": "skipped", "queue_size": len(pending), "reason": msg}

    print(f"\n{'='*55}")
    print(f"AUTOMATED RETRAIN — {len(pending)} corrections in queue")
    print(f"{'='*55}")

    # ── Load existing model to get dataset list ───────────────────────────────
    from annotation_model import RobotAnnotator, MODEL_PATH
    try:
        existing = RobotAnnotator.load()
        ds_names = dataset_names or existing.datasets_used
    except Exception:
        from annotation_model import FAILURE_CLASSES
        ds_names = dataset_names or [
            "lerobot/xarm_lift_medium_replay",
            "lerobot/xarm_push_medium_replay",
            "lerobot/aloha_sim_transfer_cube_human",
            "lerobot/aloha_sim_insertion_human",
        ]

    # ── Retrain ───────────────────────────────────────────────────────────────
    start    = time.time()
    ann      = RobotAnnotator()
    ann.le.fit(["nominal", "velocity_spike", "position_jerk", "stuck_joint",
                "gripper_event", "high_anomaly", "self_collision", "overshoot",
                "trajectory_deviation", "perception_failure"])
    report   = ann.train(ds_names)
    elapsed  = time.time() - start

    # ── Bump version + mark corrections as used ───────────────────────────────
    new_version = _bump_version(len(pending))
    for entry in queue:
        entry["used_in_retrain"] = True
    _save_queue(queue)

    # ── Write retrain log ─────────────────────────────────────────────────────
    log_path = OUTPUT_DIR / f"retrain_{new_version.replace('.','_')}.json"
    log_path.write_text(json.dumps({
        "version":          new_version,
        "trained_at":       datetime.utcnow().isoformat(),
        "corrections_used": len(pending),
        "datasets":         ds_names,
        "elapsed_sec":      round(elapsed, 1),
        "accuracy":         ann.train_accuracy,
    }, indent=2))

    print(f"\n  ✅ Retrain complete — model {new_version} ({elapsed:.0f}s)")
    print(f"  Accuracy: {ann.train_accuracy:.3f}")
    print(f"  Log: {log_path}")

    return {
        "status":           "retrained",
        "new_version":      new_version,
        "corrections_used": len(pending),
        "accuracy":         ann.train_accuracy,
        "elapsed_sec":      round(elapsed, 1),
    }


def force_retrain(dataset_names: Optional[list] = None) -> dict:
    """Trigger a retrain immediately regardless of queue size."""
    return weekly_retrain(force=True, dataset_names=dataset_names)


def clear_queue():
    """Clear the retraining queue (called after a successful retrain)."""
    _save_queue([])
    print("  [feedback] Retraining queue cleared")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Haptal feedback loop manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status",        help="Show queue status")
    sub.add_parser("retrain",       help="Run weekly retrain check")
    sub.add_parser("force-retrain", help="Force retrain now")
    sub.add_parser("clear",         help="Clear the queue")

    p_corr = sub.add_parser("correct", help="Log a single correction")
    p_corr.add_argument("--episode",   required=True)
    p_corr.add_argument("--step",      type=int, default=0)
    p_corr.add_argument("--original",  required=True)
    p_corr.add_argument("--corrected", required=True)
    p_corr.add_argument("--reviewer",  default="cli")

    args = parser.parse_args()

    if args.cmd == "status":
        s = get_queue_status()
        print(f"\nQueue status")
        print(f"  Pending corrections : {s['queue_size']} / {s['threshold']}")
        print(f"  Progress to retrain : {s['pct_to_retrain']}%")
        print(f"  Current version     : {s['current_version']}")
        print(f"  Last trained        : {s['last_trained'] or 'never'}")
        print(f"  Total corrections   : {s['total_corrections']}")
        print(f"  By class            : {s['corrections_by_class']}")

    elif args.cmd == "retrain":
        weekly_retrain()

    elif args.cmd == "force-retrain":
        force_retrain()

    elif args.cmd == "clear":
        clear_queue()

    elif args.cmd == "correct":
        on_human_correction(
            episode_id=args.episode,
            step=args.step,
            original_label=args.original,
            corrected_label=args.corrected,
            reviewer_id=args.reviewer,
        )

    else:
        parser.print_help()
