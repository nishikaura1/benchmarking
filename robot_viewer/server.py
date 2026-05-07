"""
robot_viewer/server.py — 3D Robot Episode Visualization Server
FastAPI backend: serves episode data + static frontend

Run: uvicorn robot_viewer.server:app --reload --port 8765
  or: python robot_viewer/server.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
DT = 0.02          # 50 Hz
N_EPISODES = 20    # synthetic episodes to generate
RNG = np.random.RandomState(42)

app = FastAPI(title="Robot Episode Viewer")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── xArm 4-DOF trajectory generation ──────────────────────────────────────────

# Waypoints for a pick-and-place: (q0, q1, q2, q3) in radians
WAYPOINTS = np.array([
    [ 0.00,  0.30, -0.50,  0.20],   # home
    [ 0.40,  0.55, -0.95,  0.40],   # approach
    [ 0.40,  0.35, -0.75,  0.40],   # grasp
    [ 0.40,  0.65, -1.10,  0.40],   # lift
    [-0.30,  0.50, -0.90,  0.30],   # place above
    [-0.30,  0.30, -0.70,  0.20],   # place
    [ 0.00,  0.30, -0.50,  0.20],   # return home
], dtype=np.float32)


def smooth_interp(waypoints: np.ndarray, n_steps: int, noise: float = 0.005) -> np.ndarray:
    """Interpolate through waypoints with moving-average smoothing."""
    n_wp, n_joints = waypoints.shape
    wp_x = np.linspace(0, n_steps - 1, n_wp)
    out = np.zeros((n_steps, n_joints), dtype=np.float32)
    for j in range(n_joints):
        raw = np.interp(np.arange(n_steps), wp_x, waypoints[:, j])
        # 9-point moving average for smooth joint motion
        k = 9
        padded = np.pad(raw, k // 2, mode="edge")
        out[:, j] = np.convolve(padded, np.ones(k) / k, mode="valid")
    out += RNG.randn(*out.shape).astype(np.float32) * noise
    return out


def inject_velocity_spike(pos: np.ndarray, t: int) -> np.ndarray:
    pos = pos.copy()
    j = RNG.randint(0, pos.shape[1])
    pos[t, j] += RNG.uniform(0.4, 0.8) * RNG.choice([-1, 1])
    return pos


def inject_stuck_joint(pos: np.ndarray, t: int) -> np.ndarray:
    pos = pos.copy()
    j = RNG.randint(0, pos.shape[1])
    pos[t:, j] = pos[t, j]  # joint freezes
    return pos


def inject_position_jerk(pos: np.ndarray, t: int) -> np.ndarray:
    pos = pos.copy()
    j = RNG.randint(0, pos.shape[1])
    delta = RNG.uniform(0.3, 0.6) * RNG.choice([-1, 1])
    pos[t:t + 5, j] += np.linspace(delta, 0, 5)
    return pos


def inject_overshoot(pos: np.ndarray, t: int) -> np.ndarray:
    pos = pos.copy()
    j = RNG.randint(0, pos.shape[1])
    amp = RNG.uniform(0.25, 0.5)
    decay = np.exp(-np.arange(20) * 0.3) * np.sin(np.arange(20) * 1.2)
    end = min(t + 20, len(pos))
    pos[t:end, j] += (decay[:end - t] * amp).astype(np.float32)
    return pos


def inject_deviation(pos: np.ndarray, t: int) -> np.ndarray:
    pos = pos.copy()
    n = len(pos) - t
    drift = np.linspace(0, RNG.uniform(0.2, 0.4), n)[:, None] * RNG.randn(1, pos.shape[1])
    pos[t:] += drift.astype(np.float32)
    return pos


FAILURE_INJECTORS = {
    "velocity_spike":      inject_velocity_spike,
    "stuck_joint":         inject_stuck_joint,
    "position_jerk":       inject_position_jerk,
    "overshoot":           inject_overshoot,
    "trajectory_deviation": inject_deviation,
}

FAILURE_CLASSES = list(FAILURE_INJECTORS.keys())


def build_episode(episode_id: str, failure_class: str, n_steps: int) -> dict:
    pos = smooth_interp(WAYPOINTS, n_steps)

    if failure_class == "nominal":
        failure_step = -1
    else:
        failure_step = int(n_steps * RNG.uniform(0.25, 0.65))
        injector = FAILURE_INJECTORS[failure_class]
        pos = injector(pos, failure_step)

    # Compute velocities (rad/s)
    vel = np.diff(pos, axis=0) / DT
    vel = np.vstack([vel[:1], vel]).astype(np.float32)  # same length as pos

    timestamps = (np.arange(n_steps) * DT).tolist()

    return {
        "id":               episode_id,
        "failure_class":    failure_class,
        "failure_step":     failure_step,
        "n_timesteps":      n_steps,
        "timestamps":       timestamps,
        "joint_positions":  pos.tolist(),
        "joint_velocities": vel.tolist(),
    }


def _generate_all_episodes() -> list[dict]:
    episodes = []
    failure_cycle = (FAILURE_CLASSES * 4)[:N_EPISODES - 4]
    labels = failure_cycle + ["nominal"] * 4
    RNG.shuffle(labels)

    for i, label in enumerate(labels):
        n_steps = int(RNG.uniform(80, 160))
        ep = build_episode(f"ep_{i+1:03d}", label, n_steps)
        episodes.append(ep)
    return episodes


# Build once at startup
_EPISODES: list[dict] = _generate_all_episodes()
_EPISODE_MAP: dict[str, dict] = {ep["id"]: ep for ep in _EPISODES}


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/episodes")
def list_episodes():
    """Return lightweight episode index for the left panel."""
    return [
        {
            "id":            ep["id"],
            "failure_class": ep["failure_class"],
            "failure_step":  ep["failure_step"],
            "n_timesteps":   ep["n_timesteps"],
        }
        for ep in _EPISODES
    ]


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str):
    """Return full episode data including joint positions/velocities."""
    ep = _EPISODE_MAP.get(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found")
    return ep


@app.post("/api/load_file")
async def load_file(path: str):
    """
    Load a LeRobot episode from disk.
    Supports: .parquet (columns: observation.state cols) or .hdf5
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if p.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(p)
            # find joint position columns
            state_cols = [c for c in df.columns if "state" in c.lower() or "joint" in c.lower()]
            if not state_cols:
                state_cols = df.select_dtypes("number").columns.tolist()[:8]
            pos = df[state_cols].values.astype(np.float32)

        elif p.suffix in (".hdf5", ".h5"):
            import h5py
            with h5py.File(p, "r") as f:
                # LeRobot HDF5 layout: /data/frame_index/observation.state
                if "data" in f:
                    keys = list(f["data"].keys())
                    pos = np.stack([f["data"][k]["observation.state"][()] for k in keys])
                else:
                    # try flat layout
                    pos = f["observation.state"][()].astype(np.float32)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        pos = pos.astype(np.float32)
        n_steps, n_joints = pos.shape
        n_joints = min(n_joints, 4)   # use first 4 joints
        pos = pos[:, :n_joints]

        vel = np.diff(pos, axis=0) / DT
        vel = np.vstack([vel[:1], vel]).astype(np.float32)

        ep = {
            "id":               p.stem,
            "failure_class":    "unknown",
            "failure_step":     -1,
            "n_timesteps":      n_steps,
            "timestamps":       (np.arange(n_steps) * DT).tolist(),
            "joint_positions":  pos.tolist(),
            "joint_velocities": vel.tolist(),
        }
        _EPISODE_MAP[ep["id"]] = ep
        return ep

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Static files ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
