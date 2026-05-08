"""
robot_viewer/server.py — Robot Vacuum Episode Visualization & Human Review Server

Simulates a differential-drive robot vacuum navigating cleaning runs,
with injected failure modes. Humans watch episode playback and submit
Approve / Reject / Flag decisions (with reason labels for flagged episodes).

Run:  python robot_viewer/server.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"
DT = 0.02   # 50 Hz

AMR_FAILURES = {
    "nominal":      "Successful cleaning run — full coverage achieved",
    "stuck_corner": "Robot trapped in corner, spinning in place",
    "missed_zone":  "Large cleaning area skipped due to path error",
    "cliff_error":  "Cliff sensor false positive caused erratic avoidance",
    "low_battery":  "Battery depleted mid-run, coverage incomplete",
    "wheel_slip":   "Wheel slip on smooth floor caused odometry drift",
    "tangled":      "Side brush tangled on debris, robot stalled",
    "return_fail":  "Dock return path blocked, robot navigation lost",
}

# Map semantic failure names → simulation physics behavior
_PHYS = {
    "nominal":      "nominal",
    "stuck_corner": "oscillation",
    "missed_zone":  "path_deviation",
    "cliff_error":  "e_stop",
    "low_battery":  "timeout",
    "wheel_slip":   "path_deviation",
    "tangled":      "timeout",
    "return_fail":  "oscillation",
}

# Exactly 8 episodes: (id, failure_class, pre_decision, pre_notes)
EPISODE_PLAN = [
    ("ep_001", "nominal",      "approve", "Full coverage complete — 98% area efficiency, clean dock return"),
    ("ep_002", "stuck_corner", "flag",    "Robot trapped in NE corner for ~47 steps; inspect cliff sensors and corner escape logic"),
    ("ep_003", "missed_zone",  "reject",  "Skipped ~35% of zone B — unacceptable coverage gap, re-run required"),
    ("ep_004", "nominal",      "approve", "Clean dock return, no obstacles encountered, coverage optimal"),
    ("ep_005", "cliff_error",  "flag",    "False cliff detection triggered 3× mid-run — sensor calibration needed"),
    ("ep_006", "low_battery",  "reject",  "Run abandoned at 31% coverage — battery reached 8%, schedule re-run"),
    ("ep_007", "nominal",      "approve", "Optimal spiral coverage pattern, no incidents, fast dock return"),
    ("ep_008", "tangled",      "flag",    "Side brush snagged on cable at step ~89 — mechanical inspection required"),
]


# ── Navigation simulation ─────────────────────────────────────────────────────

def simulate_navigation(
    waypoints: list[list[float]],
    failure_class: str,
    n_steps: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, int]:
    """
    Pure-pursuit differential-drive simulation.
    Returns (states, failure_step) where states has shape (n_steps, 3) = [x, y, theta].
    """
    phys = _PHYS.get(failure_class, "nominal")
    wp   = np.array(waypoints, dtype=np.float32)
    states = np.zeros((n_steps, 3), dtype=np.float32)
    init_heading = np.arctan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0])
    states[0] = [wp[0, 0], wp[0, 1], init_heading]

    failure_step = int(n_steps * rng.uniform(0.30, 0.60)) if phys != "nominal" else n_steps
    current_wp = 1
    WP_RADIUS = 0.25
    MAX_V     = 0.55   # robot vacuum moves slower than AMR
    MAX_OMEGA = 2.2

    for i in range(1, n_steps):
        x, y, theta = states[i - 1]
        post_failure = (i >= failure_step)

        if current_wp >= len(wp):
            states[i] = states[i - 1]
            continue

        target = wp[current_wp]
        dx, dy = target[0] - x, target[1] - y
        dist = np.hypot(dx, dy)

        if dist < WP_RADIUS:
            current_wp = min(current_wp + 1, len(wp) - 1)
            target = wp[current_wp]
            dx, dy = target[0] - x, target[1] - y
            dist = np.hypot(dx, dy)

        desired = np.arctan2(dy, dx)
        err = (desired - theta + np.pi) % (2 * np.pi) - np.pi

        v     = float(np.clip(dist * 0.45, 0.0, MAX_V))
        omega = float(np.clip(2.5 * err, -MAX_OMEGA, MAX_OMEGA))
        v    *= max(0.15, 1.0 - abs(err) * 0.5)

        if post_failure:
            t = float(i - failure_step)
            if phys == "e_stop":
                v     = max(0.0, v * (1.0 - t / 6.0))
                omega = 0.0
            elif phys == "oscillation":
                omega += 2.4 * np.sin(t * 1.1)
            elif phys == "path_deviation":
                v     += float(rng.randn()) * 0.35
                omega += float(rng.randn()) * 0.65
            elif phys == "timeout":
                v *= max(0.0, 1.0 - t / 28.0)

        v     += float(rng.randn()) * 0.008
        omega += float(rng.randn()) * 0.018

        new_theta = float(theta + omega * DT)
        states[i] = [
            x + v * np.cos(new_theta) * DT,
            y + v * np.sin(new_theta) * DT,
            new_theta,
        ]

    return states, failure_step if phys != "nominal" else -1


def build_episode(episode_id: str, failure_class: str, rng: np.random.RandomState) -> dict:
    # Cleaning run: compact bowing path through a ~6 m × 6 m room
    n_wp  = rng.randint(5, 8)
    start = rng.uniform(-2.0, 2.0, 2)
    goal  = start + rng.uniform(2.5, 5.0, 2) * rng.choice([-1, 1], 2)
    goal  = np.clip(goal, -4, 4)

    wps = [start.tolist()]
    for k in range(1, n_wp - 1):
        alpha   = k / (n_wp - 1)
        base    = start + (goal - start) * alpha
        perturb = rng.randn(2) * 0.7
        wps.append(np.clip(base + perturb, -4, 4).tolist())
    wps.append(goal.tolist())

    n_steps = rng.randint(180, 320)
    states, failure_step = simulate_navigation(wps, failure_class, n_steps, rng)

    dpos   = np.diff(states[:, :2], axis=0, prepend=states[:1, :2]) / DT
    v_lin  = np.linalg.norm(dpos, axis=1).tolist()
    dtheta = np.diff(states[:, 2], prepend=[states[0, 2]]) / DT
    v_ang  = dtheta.tolist()

    planned_pts = []
    for a, b in zip(wps, wps[1:]):
        for t in np.linspace(0, 1, 10):
            planned_pts.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])

    return {
        "id":            episode_id,
        "failure_class": failure_class,
        "failure_step":  int(failure_step),
        "n_timesteps":   n_steps,
        "waypoints":     wps,
        "planned_path":  planned_pts,
        "states":        states.tolist(),
        "v_linear":      v_lin,
        "v_angular":     v_ang,
        "timestamps":    (np.arange(n_steps) * DT).tolist(),
        "description":   AMR_FAILURES[failure_class],
    }


def generate_all_episodes() -> list[dict]:
    rng = np.random.RandomState(7)
    return [build_episode(ep_id, fc, rng) for ep_id, fc, _, _ in EPISODE_PLAN]


def _seed_reviews() -> dict:
    ts = datetime.utcnow().isoformat() + "Z"
    return {
        ep_id: {"decision": decision, "notes": notes, "at": ts}
        for ep_id, _, decision, notes in EPISODE_PLAN
    }


_EPISODES: list[dict] = generate_all_episodes()
_MAP: dict[str, dict]  = {ep["id"]: ep for ep in _EPISODES}
_REVIEWS: dict[str, dict] = _seed_reviews()


# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Robot Vacuum Episode Viewer")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/api/episodes")
def list_episodes():
    return [
        {
            "id":            ep["id"],
            "failure_class": ep["failure_class"],
            "failure_step":  ep["failure_step"],
            "n_timesteps":   ep["n_timesteps"],
            "description":   ep["description"],
            "review":        _REVIEWS.get(ep["id"], {}).get("decision", "pending"),
        }
        for ep in _EPISODES
    ]


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str):
    ep = _MAP.get(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found")
    return {**ep, "prev_review": _REVIEWS.get(episode_id)}


class ReviewIn(BaseModel):
    decision: str     # "approve" | "reject" | "flag"
    notes: str = ""


@app.post("/api/review/{episode_id}")
def submit_review(episode_id: str, body: ReviewIn):
    if episode_id not in _MAP:
        raise HTTPException(status_code=404)
    if body.decision not in ("approve", "reject", "flag"):
        raise HTTPException(status_code=400, detail="decision must be approve|reject|flag")
    _REVIEWS[episode_id] = {
        "decision": body.decision,
        "notes":    body.notes,
        "at":       datetime.utcnow().isoformat() + "Z",
    }
    return {"ok": True, "episode_id": episode_id}


@app.get("/api/reviews")
def get_all_reviews():
    pending  = sum(1 for ep in _EPISODES if _REVIEWS.get(ep["id"], {}).get("decision") not in ("approve", "reject", "flag"))
    approved = sum(1 for r in _REVIEWS.values() if r["decision"] == "approve")
    rejected = sum(1 for r in _REVIEWS.values() if r["decision"] == "reject")
    flagged  = sum(1 for r in _REVIEWS.values() if r["decision"] == "flag")
    return {
        "summary":   {"pending": pending, "approved": approved, "rejected": rejected, "flagged": flagged},
        "decisions": _REVIEWS,
    }


@app.post("/api/load_file")
async def load_file(path: str):
    """Load a real episode from a Parquet or HDF5 file on disk."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        if p.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(p)
            state_cols = [c for c in df.columns if any(k in c for k in ("state", "joint", "pos"))]
            if not state_cols:
                state_cols = df.select_dtypes("number").columns.tolist()[:3]
            pos = df[state_cols[:3]].values.astype(np.float32)
        elif p.suffix in (".hdf5", ".h5"):
            import h5py
            with h5py.File(p, "r") as f:
                key = "observation.state" if "observation.state" in f else list(f.keys())[0]
                pos = f[key][()].astype(np.float32)
                if pos.ndim == 1:
                    pos = pos.reshape(-1, 1)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format (use .parquet or .hdf5)")

        n_steps, n_cols = pos.shape
        if n_cols >= 3:
            states = pos[:, :3]
        elif n_cols == 2:
            states = np.column_stack([pos, np.zeros(n_steps)])
        else:
            states = np.column_stack([pos[:, 0], np.zeros((n_steps, 2))])

        dpos   = np.diff(states[:, :2], axis=0, prepend=states[:1, :2]) / DT
        v_lin  = np.linalg.norm(dpos, axis=1).tolist()
        dtheta = np.diff(states[:, 2], prepend=[states[0, 2]]) / DT

        ep = {
            "id":            p.stem,
            "failure_class": "unknown",
            "failure_step":  -1,
            "n_timesteps":   n_steps,
            "waypoints":     [states[0, :2].tolist(), states[-1, :2].tolist()],
            "planned_path":  [],
            "states":        states.tolist(),
            "v_linear":      v_lin,
            "v_angular":     dtheta.tolist(),
            "timestamps":    (np.arange(n_steps) * DT).tolist(),
            "description":   "Imported episode",
            "prev_review":   None,
        }
        _MAP[ep["id"]] = ep
        return ep

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
