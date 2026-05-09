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


# ── Datasets (20 open-source / synthetic) ─────────────────────────────────────

DATASETS = [
    {"id":"ds_robocleaner","name":"RoboCleaner Vacuums","org":"Haptal Synthetic Lab","year":2024,"n_episodes":8,"tags":["cleaning","vacuum","coverage"],"description":"Synthetic multi-room vacuum cleaning corpus — basis for the live Haptal Engine demo. Includes diverse failure modes.","color":"#26bfb0","license":"Proprietary","failure_mix":["nominal","stuck_corner","missed_zone","cliff_error","low_battery","wheel_slip","tangled","return_fail"]},
    {"id":"ds_bridge_v2","name":"Bridge Data V2","org":"UC Berkeley","year":2023,"n_episodes":60064,"tags":["manipulation","household","multi-task"],"description":"60k+ real robot demonstrations of household manipulation across 24 environments.","color":"#f4b400","license":"CC BY-4.0","failure_mix":["nominal","nominal","tangled","missed_zone","nominal","stuck_corner","nominal","cliff_error"]},
    {"id":"ds_droid","name":"DROID","org":"Stanford RAIL","year":2024,"n_episodes":76000,"tags":["manipulation","dexterous","generalizable"],"description":"Large-scale robot manipulation dataset with 76k trajectories across 564 real-world scenes.","color":"#db4437","license":"Apache-2.0","failure_mix":["nominal","nominal","nominal","wheel_slip","missed_zone","tangled","return_fail"]},
    {"id":"ds_openx_fractal","name":"Open X Fractal","org":"Google DeepMind","year":2023,"n_episodes":87212,"tags":["manipulation","tabletop","grasping"],"description":"Tabletop pick-and-place with a Franka arm across diverse objects and grasping configurations.","color":"#4285f4","license":"Apache-2.0","failure_mix":["nominal","nominal","nominal","missed_zone","stuck_corner","wheel_slip","nominal"]},
    {"id":"ds_sacson","name":"SACSoN","org":"Stanford / CMU","year":2023,"n_episodes":8200,"tags":["navigation","outdoor","social"],"description":"Social navigation in crowded outdoor environments — stop-and-go human traffic, uncertain intentions.","color":"#0f9d58","license":"MIT","failure_mix":["nominal","nominal","return_fail","stuck_corner","nominal","cliff_error"]},
    {"id":"ds_roboagent","name":"RoboAgent","org":"CMU","year":2023,"n_episodes":7500,"tags":["manipulation","semantic","generalization"],"description":"Multi-task semantic manipulation — 12 tasks, 1 robot, tested on novel object instances.","color":"#9c27b0","license":"MIT","failure_mix":["nominal","stuck_corner","nominal","missed_zone","nominal"]},
    {"id":"ds_calvin","name":"CALVIN","org":"Freiburg / TU Berlin","year":2022,"n_episodes":24000,"tags":["manipulation","language-conditioned","long-horizon"],"description":"Language-conditioned long-horizon manipulation across 34 tasks in 4 scene configurations.","color":"#00bcd4","license":"MIT","failure_mix":["nominal","nominal","return_fail","cliff_error","missed_zone","nominal"]},
    {"id":"ds_habitat_hm3d","name":"Habitat HM3D Nav","org":"Meta AI","year":2022,"n_episodes":36000,"tags":["navigation","indoor","photorealistic"],"description":"Object-goal navigation in 1,000 photorealistic HM3D environments.","color":"#3f51b5","license":"CC BY-4.0","failure_mix":["nominal","nominal","return_fail","stuck_corner","cliff_error","nominal","return_fail","nominal"]},
    {"id":"ds_rh20t","name":"RH20T","org":"Shanghai AI Lab","year":2023,"n_episodes":110000,"tags":["manipulation","contact-rich","force-torque"],"description":"Rich haptic manipulation dataset with force/torque sensing — peg-in-hole, assembly, wiping.","color":"#e91e63","license":"CC BY-NC-4.0","failure_mix":["nominal","nominal","stuck_corner","missed_zone","tangled","nominal","wheel_slip"]},
    {"id":"ds_tiago","name":"TIAGo Domestic Tasks","org":"PAL Robotics / EU-OpenDR","year":2023,"n_episodes":4200,"tags":["manipulation","mobile","domestic"],"description":"Domestic service robot demonstrations: fetching, door-opening, tray transport in cluttered homes.","color":"#795548","license":"Apache-2.0","failure_mix":["nominal","tangled","nominal","low_battery","stuck_corner"]},
    {"id":"ds_locobot","name":"LocoBot Indoor Nav","org":"Berkeley Embodied AI Lab","year":2023,"n_episodes":9100,"tags":["navigation","legged","indoor"],"description":"Quadruped indoor navigation with social-aware path planning in real office environments.","color":"#009688","license":"MIT","failure_mix":["nominal","nominal","cliff_error","wheel_slip","return_fail","nominal"]},
    {"id":"ds_openbot","name":"OpenBot Racing","org":"ETH Zurich","year":2022,"n_episodes":3400,"tags":["navigation","racing","speed"],"description":"High-speed autonomous racing on outdoor tracks with visual odometry and depth cameras.","color":"#ff9800","license":"MIT","failure_mix":["nominal","wheel_slip","nominal","cliff_error","low_battery"]},
    {"id":"ds_iamlab","name":"IAMLab Deformable","org":"CMU IAM Lab","year":2023,"n_episodes":6200,"tags":["manipulation","deformable","cloth"],"description":"Deformable object manipulation — cloth folding, rope manipulation, bag packing.","color":"#607d8b","license":"MIT","failure_mix":["nominal","tangled","nominal","tangled","missed_zone","return_fail"]},
    {"id":"ds_spot","name":"Spot Inspection Corpus","org":"Boston Dynamics / MIT","year":2023,"n_episodes":1800,"tags":["navigation","quadruped","inspection"],"description":"Facility inspection runs with Spot: stairs, uneven terrain, door traversal, outdoor environments.","color":"#ffc107","license":"Apache-2.0","failure_mix":["nominal","nominal","return_fail","wheel_slip","cliff_error","nominal","low_battery"]},
    {"id":"ds_robopen_drawer","name":"Open X-E Drawer","org":"Open X-Embodiment","year":2023,"n_episodes":18000,"tags":["manipulation","articulated","multi-robot"],"description":"Cross-robot drawer opening/closing — 7 different robot arms, standardized data format.","color":"#8bc34a","license":"Apache-2.0","failure_mix":["nominal","nominal","stuck_corner","nominal","missed_zone"]},
    {"id":"ds_oxe_kitchen","name":"OXE Kitchen Demos","org":"Open X-Embodiment","year":2023,"n_episodes":22000,"tags":["manipulation","kitchen","multi-robot"],"description":"Kitchen manipulation across 22 robot morphologies — cutting, stacking, wiping, pouring.","color":"#cddc39","license":"Apache-2.0","failure_mix":["nominal","nominal","tangled","missed_zone","nominal","cliff_error"]},
    {"id":"ds_gtc_outdoor","name":"GTC Outdoor Nav","org":"Georgia Tech","year":2022,"n_episodes":5600,"tags":["navigation","outdoor","terrain"],"description":"Outdoor navigation on uneven terrain: grass, gravel, sidewalk transitions, rain conditions.","color":"#4caf50","license":"CC BY-4.0","failure_mix":["nominal","wheel_slip","cliff_error","nominal","return_fail"]},
    {"id":"ds_tidybot","name":"TidyBot","org":"Princeton NLP","year":2023,"n_episodes":2100,"tags":["navigation","manipulation","llm-guided"],"description":"LLM-guided household tidying — picking, sorting, and placing objects into furniture.","color":"#2196f3","license":"MIT","failure_mix":["nominal","return_fail","nominal","low_battery","tangled"]},
    {"id":"ds_robomimic","name":"RoboMimic","org":"Stanford / UT Austin","year":2021,"n_episodes":4500,"tags":["manipulation","imitation","benchmark"],"description":"Human demonstration dataset for imitation learning with multiple quality tiers per task.","color":"#ff5722","license":"MIT","failure_mix":["nominal","nominal","wheel_slip","stuck_corner","nominal"]},
    {"id":"ds_anymal","name":"ANYmal Field Data","org":"ANYbotics","year":2023,"n_episodes":2800,"tags":["quadruped","terrain","legged"],"description":"ANYmal quadruped field deployment data across diverse outdoor terrains and industrial sites.","color":"#a29bfe","license":"CC BY-SA 4.0","failure_mix":["nominal","nominal","wheel_slip","cliff_error","return_fail","stuck_corner"]},
]

_DS_MAP: dict[str, dict] = {ds["id"]: ds for ds in DATASETS}
_DATASET_EPISODE_CACHE: dict[str, list[dict]] = {}


def get_dataset_episodes(dataset_id: str) -> list[dict]:
    if dataset_id not in _DATASET_EPISODE_CACHE:
        ds = _DS_MAP.get(dataset_id)
        if not ds:
            return []
        seed = abs(hash(dataset_id)) % (2 ** 31)
        rng  = np.random.RandomState(seed)
        eps  = []
        for i, fc in enumerate(ds["failure_mix"]):
            ep_id  = f"{dataset_id}_ep{i + 1:02d}"
            ep_rng = np.random.RandomState(int(rng.randint(0, 99999)))
            ep     = build_episode(ep_id, fc, ep_rng)
            ep["dataset_id"] = dataset_id
            _MAP[ep_id] = ep
            eps.append(ep)
        _DATASET_EPISODE_CACHE[dataset_id] = eps
    return _DATASET_EPISODE_CACHE[dataset_id]


# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Robot Vacuum Episode Viewer")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/api/datasets")
def list_datasets_endpoint():
    return [
        {k: v for k, v in ds.items() if k != "failure_mix"}
        for ds in DATASETS
    ]


@app.get("/api/datasets/{dataset_id}/episodes")
def list_dataset_episodes(dataset_id: str):
    if dataset_id not in _DS_MAP:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    eps = get_dataset_episodes(dataset_id)
    return {
        "dataset": {k: v for k, v in _DS_MAP[dataset_id].items() if k != "failure_mix"},
        "episodes": [
            {
                "id":            ep["id"],
                "failure_class": ep["failure_class"],
                "failure_step":  ep["failure_step"],
                "n_timesteps":   ep["n_timesteps"],
                "description":   ep["description"],
                "dataset_id":    ep["dataset_id"],
                "review":        _REVIEWS.get(ep["id"], {}).get("decision", "pending"),
            }
            for ep in eps
        ],
    }


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
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
