"""
generate_amr_dataset.py
=======================
Generates synthetic AMR (Autonomous Mobile Robot) navigation dataset.
500 episodes across 5 failure classes with realistic physics simulation.

Classes: nominal (200), path_deviation (75), velocity_spike (75),
         stuck (75), overcorrect (75)

Output: amr_train.parquet (400 eps), amr_test.parquet (100 eps)
"""

import json
from math import sin, cos, atan2, sqrt, pi
from pathlib import Path

import numpy as np
import pandas as pd

ROOM_SIZE   = 10.0   # metres
DT          = 0.1    # seconds per timestep
MAX_V       = 0.55   # m/s  max linear velocity
MAX_OMEGA   = 2.2    # rad/s max angular velocity
WP_RADIUS   = 0.30   # waypoint capture radius


# ── Pure-pursuit controller ───────────────────────────────────────────────────

def _pursue(x, y, theta, tx, ty, rng, noise_v=0.008, noise_w=0.018):
    dx, dy = tx - x, ty - y
    dist   = sqrt(dx**2 + dy**2)
    desired = atan2(dy, dx)
    err     = atan2(sin(desired - theta), cos(desired - theta))

    v     = float(np.clip(dist * 0.4, 0.0, MAX_V))
    omega = float(np.clip(2.5 * err, -MAX_OMEGA, MAX_OMEGA))
    v     += float(rng.normal(0, noise_v))
    omega += float(rng.normal(0, noise_w))
    return v, omega, dist


def _step(x, y, theta, v, omega):
    theta = theta + omega * DT
    x = float(np.clip(x + v * cos(theta) * DT, 0.15, ROOM_SIZE - 0.15))
    y = float(np.clip(y + v * sin(theta) * DT, 0.15, ROOM_SIZE - 0.15))
    return x, y, theta


def _waypoints(rng, n=4):
    pts = [(float(rng.uniform(1.5, 8.5)), float(rng.uniform(1.5, 8.5))) for _ in range(n)]
    return pts


def _base_nav(x0, y0, theta0, waypoints, rng, n_steps, noise_v=0.008, noise_w=0.018):
    """Follow waypoints with pure-pursuit. Returns list of (x,y,theta,v,omega)."""
    x, y, theta = x0, y0, theta0
    wp_idx = 0
    path   = []
    for _ in range(n_steps):
        if wp_idx >= len(waypoints):
            v, omega = 0.0, 0.0
        else:
            tx, ty = waypoints[wp_idx]
            v, omega, dist = _pursue(x, y, theta, tx, ty, rng, noise_v, noise_w)
            if dist < WP_RADIUS:
                wp_idx += 1
        x, y, theta = _step(x, y, theta, v, omega)
        path.append((x, y, theta % (2*pi), max(0.0, v), omega))
    return path


# ── Episode generators ────────────────────────────────────────────────────────

def gen_nominal(ep_id, rng):
    n_steps = rng.randint(90, 160)
    wps     = _waypoints(rng, n=rng.randint(3, 6))
    x0, y0  = wps[0]
    theta0  = float(rng.uniform(0, 2*pi))
    path    = _base_nav(x0, y0, theta0, wps[1:], rng, n_steps)
    return _rows(ep_id, path, 'nominal', None)


def gen_path_deviation(ep_id, rng):
    n_steps   = rng.randint(100, 160)
    wps       = _waypoints(rng, n=4)
    x0, y0    = wps[0]
    theta0    = float(rng.uniform(0, 2*pi))
    fail_step = int(n_steps * rng.uniform(0.25, 0.50))

    # Drift direction — random unit vector
    drift_dir = float(rng.uniform(-1, 1))
    drift_rate = float(rng.uniform(0.008, 0.020))   # metres per step, cumulative

    path = []
    x, y, theta = x0, y0, theta0
    wp_idx = 0
    drift_accum = 0.0

    for t in range(n_steps):
        if wp_idx >= len(wps) - 1:
            v, omega = 0.0, 0.0
        else:
            tx, ty = wps[wp_idx + 1]
            v, omega, dist = _pursue(x, y, theta, tx, ty, rng)
            if dist < WP_RADIUS:
                wp_idx += 1

        if t >= fail_step:
            drift_accum += drift_rate
            # Lateral drift — perpendicular to heading
            x += cos(theta + pi/2) * drift_accum * drift_dir * DT
            y += sin(theta + pi/2) * drift_accum * drift_dir * DT
            x = float(np.clip(x, 0.15, ROOM_SIZE - 0.15))
            y = float(np.clip(y, 0.15, ROOM_SIZE - 0.15))

        x, y, theta = _step(x, y, theta, v, omega)
        path.append((x, y, theta % (2*pi), max(0.0, v), omega))

    return _rows(ep_id, path, 'path_deviation', fail_step)


def gen_velocity_spike(ep_id, rng):
    n_steps   = rng.randint(80, 150)
    wps       = _waypoints(rng, n=4)
    x0, y0    = wps[0]
    theta0    = float(rng.uniform(0, 2*pi))
    fail_step = int(n_steps * rng.uniform(0.30, 0.65))
    spike_mag = float(rng.uniform(3.5, 6.0))
    spike_len = rng.randint(2, 5)

    path    = []
    x, y, theta = x0, y0, theta0
    wp_idx  = 0

    for t in range(n_steps):
        if wp_idx >= len(wps) - 1:
            v, omega = 0.0, 0.0
        else:
            tx, ty = wps[wp_idx + 1]
            v, omega, dist = _pursue(x, y, theta, tx, ty, rng)
            if dist < WP_RADIUS:
                wp_idx += 1

        # Spike window
        if fail_step <= t < fail_step + spike_len:
            v = v * spike_mag
            omega = omega * float(rng.uniform(0.5, 1.5))

        x, y, theta = _step(x, y, theta, v, omega)
        path.append((x, y, theta % (2*pi), abs(v), omega))

    return _rows(ep_id, path, 'velocity_spike', fail_step)


def gen_stuck(ep_id, rng):
    n_steps   = rng.randint(100, 160)
    wps       = _waypoints(rng, n=4)
    x0, y0    = wps[0]
    theta0    = float(rng.uniform(0, 2*pi))
    fail_step = int(n_steps * rng.uniform(0.30, 0.60))

    path    = []
    x, y, theta = x0, y0, theta0
    wp_idx  = 0

    for t in range(n_steps):
        if t >= fail_step:
            # Robot stops — tiny vibration only
            v     = float(rng.normal(0, 0.003))
            omega = float(rng.normal(0, 0.005))
        else:
            if wp_idx >= len(wps) - 1:
                v, omega = 0.0, 0.0
            else:
                tx, ty = wps[wp_idx + 1]
                v, omega, dist = _pursue(x, y, theta, tx, ty, rng)
                if dist < WP_RADIUS:
                    wp_idx += 1

        x, y, theta = _step(x, y, theta, v, omega)
        path.append((x, y, theta % (2*pi), max(0.0, abs(v)), omega))

    return _rows(ep_id, path, 'stuck', fail_step)


def gen_overcorrect(ep_id, rng):
    n_steps   = rng.randint(100, 160)
    wps       = _waypoints(rng, n=4)
    x0, y0    = wps[0]
    theta0    = float(rng.uniform(0, 2*pi))
    fail_step = int(n_steps * rng.uniform(0.30, 0.55))
    amplify   = float(rng.uniform(2.8, 5.0))
    osc_freq  = float(rng.uniform(0.4, 0.8))

    path    = []
    x, y, theta = x0, y0, theta0
    wp_idx  = 0

    for t in range(n_steps):
        if wp_idx >= len(wps) - 1:
            v, omega = 0.0, 0.0
        else:
            tx, ty = wps[wp_idx + 1]
            v, omega, dist = _pursue(x, y, theta, tx, ty, rng)
            if dist < WP_RADIUS:
                wp_idx += 1

        if t >= fail_step:
            # Oscillating overcorrection — decaying sinusoid in omega
            decay = float(np.exp(-0.03 * (t - fail_step)))
            omega = omega * amplify * sin(osc_freq * (t - fail_step)) * decay

        x, y, theta = _step(x, y, theta, v, omega)
        path.append((x, y, theta % (2*pi), max(0.0, v), omega))

    return _rows(ep_id, path, 'overcorrect', fail_step)


def _rows(ep_id, path, failure_class, failure_timestep):
    rows = []
    for t, (x, y, theta, v, omega) in enumerate(path):
        rows.append({
            'episode_id':       ep_id,
            'timestep':         t,
            'x':                round(x, 5),
            'y':                round(y, 5),
            'theta':            round(theta, 5),
            'velocity':         round(v, 5),
            'angular_velocity': round(omega, 5),
            'failure_class':    failure_class,
            'failure_timestep': float(failure_timestep) if failure_timestep is not None else None,
            'synthetic':        True,
        })
    return rows


# ── Main generation ───────────────────────────────────────────────────────────

GENERATORS = {
    'nominal':        (gen_nominal,        200),
    'path_deviation': (gen_path_deviation,  75),
    'velocity_spike': (gen_velocity_spike,  75),
    'stuck':          (gen_stuck,           75),
    'overcorrect':    (gen_overcorrect,     75),
}


def generate_all(seed=42):
    rng  = np.random.RandomState(seed)
    all_rows = []
    ep_counts = {}

    for fc, (gen_fn, n_eps) in GENERATORS.items():
        ep_counts[fc] = 0
        for i in range(n_eps):
            ep_id     = f"{fc}_{i:04d}"
            ep_seed   = int(rng.randint(0, 2**31))
            ep_rng    = np.random.RandomState(ep_seed)
            rows      = gen_fn(ep_id, ep_rng)
            all_rows.extend(rows)
            ep_counts[fc] += 1
        print(f"  {fc}: {n_eps} episodes generated")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total episodes: {df['episode_id'].nunique()}")
    return df, ep_counts


def split_and_save(df, output_dir='.'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratified 80/20 split by episode
    episodes = df[['episode_id','failure_class']].drop_duplicates()
    train_ids, test_ids = [], []

    for fc, grp in episodes.groupby('failure_class'):
        ids = grp['episode_id'].tolist()
        np.random.RandomState(42).shuffle(ids)
        cut = int(len(ids) * 0.8)
        train_ids.extend(ids[:cut])
        test_ids.extend(ids[cut:])

    train_df = df[df['episode_id'].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df['episode_id'].isin(test_ids)].reset_index(drop=True)

    train_path = output_dir / 'amr_train.parquet'
    test_path  = output_dir / 'amr_test.parquet'
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)

    print(f"\nSaved:")
    print(f"  {train_path}  ({train_df['episode_id'].nunique()} episodes, {len(train_df):,} rows)")
    print(f"  {test_path}   ({test_df['episode_id'].nunique()} episodes, {len(test_df):,} rows)")
    return train_df, test_df


def write_dataset_readme(output_dir='.'):
    readme = """---
title: Haptal AMR Navigation Failure Dataset
emoji: 🤖
colorFrom: red
colorTo: gray
tags:
  - robotics
  - failure-detection
  - navigation
  - synthetic
  - amr
license: apache-2.0
---

# Haptal AMR Navigation Failure Dataset

Synthetic autonomous mobile robot (AMR) navigation dataset with labeled failure modes,
generated by [Haptal](https://haptal.ai) — robotics failure intelligence.

Companion benchmark: [HaptalAI/robotics-failure-benchmark](https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark)

---

## Dataset Summary

| Split | Episodes | Rows |
|-------|----------|------|
| train | 400 | ~52,000 |
| test  | 100 | ~13,000 |

5 failure classes, 10×10 metre room, 50 Hz simulation (DT=0.1s).

---

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | string | Unique episode ID, e.g. `nominal_0001` |
| `timestep` | int | Step index within episode, starting at 0 |
| `x` | float | Robot x position in metres (room is 0–10 m) |
| `y` | float | Robot y position in metres (room is 0–10 m) |
| `theta` | float | Robot heading in radians (0–2π) |
| `velocity` | float | Linear velocity in m/s |
| `angular_velocity` | float | Rotational velocity in rad/s |
| `failure_class` | string | One of: nominal, path_deviation, velocity_spike, stuck, overcorrect |
| `failure_timestep` | float or null | Timestep at which failure begins; null for nominal |
| `synthetic` | bool | Always True — these are simulated episodes |

---

## Failure Classes

| Class | Description |
|-------|-------------|
| `nominal` | Clean successful navigation — robot follows waypoints, reaches destination |
| `path_deviation` | Robot gradually drifts laterally from its intended path mid-run |
| `velocity_spike` | Robot experiences a sudden sharp speed anomaly for 2–5 timesteps |
| `stuck` | Robot stops completely mid-path and does not recover |
| `overcorrect` | Robot overcorrects a minor deviation and oscillates |

---

## About Episode IDs

Episode IDs are sequential synthetic counters: `{failure_class}_{index:04d}`.
The index is **not** an index into any external dataset — it simply counts how many
episodes of that class were generated. `path_deviation_0012` is the 13th
path_deviation episode in the generation run.

---

## Load with `datasets` library

```python
from datasets import load_dataset

ds = load_dataset("HaptalAI/amr-navigation-failure-dataset")
train = ds["train"].to_pandas()
test  = ds["test"].to_pandas()

# Get all timesteps for one episode
ep = train[train["episode_id"] == "nominal_0001"]
print(ep[["timestep","x","y","velocity"]].head())
```

---

## Contact

- Email: aarav@haptal.ai
- Website: https://haptal.ai
"""
    path = Path(output_dir) / 'amr_dataset_README.md'
    path.write_text(readme)
    print(f"  {path}")


if __name__ == '__main__':
    print("Generating AMR navigation dataset...")
    df, counts = generate_all(seed=42)
    train_df, test_df = split_and_save(df)
    write_dataset_readme()
    print("\nDone.")
