"""
Haptal Human-in-the-Loop Robot Trajectory Reviewer
====================================================
Gradio Space — humans watch robot trajectories animate in Three.js and
make Pass / Reject decisions. Curated dataset downloadable as CSV.
"""

import base64
import csv
import datetime
import io
import json
import os
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
WEBHOOK_URL       = ""   # e.g. "https://hooks.zapier.com/hooks/catch/..."
HUGGINGFACE_TOKEN = ""   # optional — for private datasets
CTA_EMAIL         = "aarav@haptal.ai"
LEADS_FILE        = "leads.json"

DATASETS = [
    "HaptalAI/amr-navigation-failure-dataset",
    "lerobot/xarm_lift_medium_replay",
    "lerobot/xarm_push_medium_replay",
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_insertion_human",
]

# ── Caches ────────────────────────────────────────────────────────────────────
_df_cache: dict[str, pd.DataFrame] = {}


# ── Webhook helper ────────────────────────────────────────────────────────────
def _fire(payload: dict):
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception:
        pass


def _log_lead(entry: dict):
    existing = []
    if Path(LEADS_FILE).exists():
        try:
            with open(LEADS_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.append(entry)
    with open(LEADS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


# ── Dataset loading ───────────────────────────────────────────────────────────
def _load_df(dataset_name: str) -> pd.DataFrame:
    if dataset_name in _df_cache:
        return _df_cache[dataset_name]

    if dataset_name == "HaptalAI/amr-navigation-failure-dataset":
        # Try local parquet first (for development)
        local = Path("amr_train.parquet")
        if local.exists():
            df = pd.read_parquet(local)
        else:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            df = ds.to_pandas()
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()

    _df_cache[dataset_name] = df
    return df


def _get_episode_ids(dataset_name: str) -> list[str]:
    df = _load_df(dataset_name)
    if "episode_id" in df.columns:
        return sorted(df["episode_id"].unique().tolist())
    col = next((c for c in df.columns if "episode" in c.lower()), None)
    if col:
        return [str(v) for v in sorted(df[col].unique().tolist())]
    return ["0"]


def _extract_trajectory(dataset_name: str, ep_id: str) -> dict:
    """
    Returns dict with keys:
      traj: list of {x, y, theta, velocity}
      episode_id, failure_class, failure_timestep, total_timesteps
    """
    df = _load_df(dataset_name)

    if dataset_name == "HaptalAI/amr-navigation-failure-dataset":
        ep_df = df[df["episode_id"] == ep_id].sort_values("timestep").reset_index(drop=True)
        if len(ep_df) == 0:
            raise ValueError(f"Episode {ep_id} not found.")
        traj = ep_df[["x", "y", "theta", "velocity"]].rename(
            columns={"velocity": "velocity"}
        ).to_dict("records")
        fc   = ep_df["failure_class"].iloc[0]
        ft   = ep_df["failure_timestep"].iloc[0]
        ft   = None if pd.isna(ft) else int(ft)
        return {
            "traj": traj,
            "episode_id": ep_id,
            "failure_class": fc,
            "failure_timestep": ft,
            "total_timesteps": len(ep_df),
            "confidence": 0.92 if fc != "nominal" else 0.05,
        }
    else:
        # LeRobot — use episode_index
        ep_col = next((c for c in df.columns if "episode" in c.lower()), None)
        try:
            ep_idx = int(ep_id)
        except ValueError:
            ep_idx = 0
        ep_df = df[df[ep_col] == ep_idx].reset_index(drop=True)
        if len(ep_df) == 0:
            raise ValueError(f"Episode {ep_idx} not found in {dataset_name}.")

        # Extract state dims 0,1 as x,y proxy — normalise to 0-10 range
        state_col = next(
            (c for c in ep_df.columns if "observation.state" in c or "state" in c.lower()),
            None,
        )
        if state_col:
            sample = ep_df[state_col].iloc[0]
            if hasattr(sample, "__len__"):
                arr = np.vstack(ep_df[state_col].values).astype(np.float32)
            else:
                arr = ep_df[state_col].values.astype(np.float32).reshape(-1, 1)
        else:
            num_cols = ep_df.select_dtypes("number").columns[:3].tolist()
            arr = ep_df[num_cols].values.astype(np.float32)

        # Normalise x,y to 0-10
        def _norm(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn + 1e-8) * 9.0 + 0.5

        x_raw = arr[:, 0]
        y_raw = arr[:, 1] if arr.shape[1] > 1 else np.zeros(len(arr))
        x_n   = _norm(x_raw)
        y_n   = _norm(y_raw)

        # Heading from movement direction
        dx = np.diff(x_n, prepend=x_n[:1])
        dy = np.diff(y_n, prepend=y_n[:1])
        theta = np.arctan2(dy, dx)

        # Velocity proxy
        vel = np.sqrt(dx**2 + dy**2) * 10

        # Heuristic anomaly: z-score > 2.5 on any dim
        if arr.shape[1] >= 1:
            z = np.abs((arr - arr.mean(0)) / (arr.std(0) + 1e-8))
            spike_steps = np.where(z.max(1) > 2.5)[0]
            ft = int(spike_steps[0]) if len(spike_steps) > 0 else None
            fc = "velocity_spike" if ft is not None else "nominal"
        else:
            ft, fc = None, "nominal"

        traj = [
            {"x": float(x_n[i]), "y": float(y_n[i]),
             "theta": float(theta[i]), "velocity": float(vel[i])}
            for i in range(len(x_n))
        ]
        return {
            "traj": traj,
            "episode_id": str(ep_idx),
            "failure_class": fc,
            "failure_timestep": ft,
            "total_timesteps": len(traj),
            "confidence": 0.71 if ft is not None else 0.12,
        }


# ── Three.js animation HTML ───────────────────────────────────────────────────
def _make_animation_html(ep_data: dict) -> str:
    traj_json = json.dumps(ep_data["traj"])
    ft_json   = json.dumps(ep_data["failure_timestep"])

    inner = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; font-family: system-ui, sans-serif; overflow: hidden; }}
  #c {{ display: block; width: 100%; }}
  #controls {{
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; background: #1e293b;
    border-top: 1px solid #334155;
  }}
  button {{
    background: #334155; color: #f1f5f9; border: none;
    padding: 5px 12px; border-radius: 5px; cursor: pointer;
    font-size: 12px; font-weight: 600;
  }}
  button:hover {{ background: #475569; }}
  button.active {{ background: #ef4444; }}
  #step-info {{
    margin-left: auto; font-size: 12px; color: #94a3b8; font-family: monospace;
  }}
  #scrubber-wrap {{
    padding: 0 12px 6px; background: #1e293b; position: relative;
  }}
  #scrubber {{
    width: 100%; accent-color: #ef4444; cursor: pointer;
  }}
  #failure-marker {{
    position: absolute; top: 0; height: 18px; width: 2px;
    background: #ef4444; pointer-events: none;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="controls">
  <button id="btn-play" onclick="togglePlay()">▶ Play</button>
  <button onclick="setSpeed(0.5)">0.5×</button>
  <button id="btn-1x" class="active" onclick="setSpeed(1)">1×</button>
  <button onclick="setSpeed(2)">2×</button>
  <button onclick="restart()">↺</button>
  <span id="step-info">Step 0 / 0</span>
</div>
<div id="scrubber-wrap">
  <div id="failure-marker" style="display:none;"></div>
  <input id="scrubber" type="range" min="0" max="100" value="0"
    oninput="onScrub(this.value)">
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const TRAJ       = {traj_json};
const FAILURE_TS = {ft_json};
const N          = TRAJ.length;

// ── Scene setup ──────────────────────────────────────────────────────────────
const canvas   = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.shadowMap.enabled = true;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0f172a);
scene.fog        = new THREE.Fog(0x0f172a, 18, 30);

// Adjust canvas to window
function resize() {{
  const w = canvas.parentElement.clientWidth || 640;
  const h = Math.round(w * 0.55);
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}

const camera = new THREE.PerspectiveCamera(55, 1.6, 0.1, 100);
camera.position.set(5, 11, -2);
camera.lookAt(5, 0, 5);

// ── Room ─────────────────────────────────────────────────────────────────────
// Floor
const floorGeo = new THREE.PlaneGeometry(10, 10);
const floorMat = new THREE.MeshPhongMaterial({{ color: 0x0d1b2a }});
const floor    = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.set(5, 0, 5);
floor.receiveShadow = true;
scene.add(floor);

// Grid
const grid = new THREE.GridHelper(10, 20, 0x1e3a5f, 0x1e2d3d);
grid.position.set(5, 0.001, 5);
scene.add(grid);

// Walls (thin boxes)
function wall(w, h, d, x, y, z) {{
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshPhongMaterial({{ color: 0x1e293b, transparent: true, opacity: 0.6 }})
  );
  m.position.set(x, y, z);
  scene.add(m);
}}
wall(10, 0.8, 0.08,  5, 0.4, 0);    // north
wall(10, 0.8, 0.08,  5, 0.4, 10);   // south
wall(0.08, 0.8, 10,  0, 0.4, 5);    // west
wall(0.08, 0.8, 10, 10, 0.4, 5);    // east

// ── Lights ───────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(5, 12, 2);
dirLight.castShadow = true;
scene.add(dirLight);

// ── Roomba robot ─────────────────────────────────────────────────────────────
const robotGroup = new THREE.Group();

// Body disc
const bodyMat  = new THREE.MeshPhongMaterial({{ color: 0x334155 }});
const bodyMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.09, 32), bodyMat);
bodyMesh.castShadow = true;
robotGroup.add(bodyMesh);

// Top dome accent
const domeMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.18, 16, 8, 0, Math.PI*2, 0, Math.PI/2),
  new THREE.MeshPhongMaterial({{ color: 0x475569 }})
);
domeMesh.position.y = 0.045;
robotGroup.add(domeMesh);

// Side wheels
const wheelGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.04, 14);
const wheelMat = new THREE.MeshPhongMaterial({{ color: 0x0f172a }});
[-1, 1].forEach(side => {{
  const w = new THREE.Mesh(wheelGeo, wheelMat);
  w.rotation.z = Math.PI / 2;
  w.position.set(side * 0.30, -0.025, 0);
  robotGroup.add(w);
}});

// Front indicator LED
const ledMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.04, 8, 8),
  new THREE.MeshPhongMaterial({{ color: 0x22d3ee, emissive: 0x22d3ee, emissiveIntensity: 0.8 }})
);
ledMesh.position.set(0, 0.05, 0.28);
robotGroup.add(ledMesh);

scene.add(robotGroup);

// ── Path line ─────────────────────────────────────────────────────────────────
const MAX_PATH   = N;
const pathPositions = new Float32Array(MAX_PATH * 3);
const pathGeo    = new THREE.BufferGeometry();
pathGeo.setAttribute('position', new THREE.BufferAttribute(pathPositions, 3));
const pathMat    = new THREE.LineBasicMaterial({{ color: 0x22c55e, linewidth: 2 }});
const pathLine   = new THREE.Line(pathGeo, pathMat);
scene.add(pathLine);

// ── Failure marker sphere ─────────────────────────────────────────────────────
let markerMesh = null;
if (FAILURE_TS !== null && FAILURE_TS < N) {{
  const t = TRAJ[FAILURE_TS];
  markerMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 12, 12),
    new THREE.MeshPhongMaterial({{ color: 0xef4444, transparent: true, opacity: 0.8 }})
  );
  markerMesh.position.set(t.x, 0.15, t.y);
  scene.add(markerMesh);
}}

// Scrubber failure marker
const scrubEl = document.getElementById('scrubber');
const fmarker  = document.getElementById('failure-marker');
scrubEl.max = N - 1;
if (FAILURE_TS !== null) {{
  const pct = FAILURE_TS / (N - 1) * 100;
  fmarker.style.display  = 'block';
  fmarker.style.left     = `calc(${{pct}}% + 12px)`;
}}

// ── State ─────────────────────────────────────────────────────────────────────
let step    = 0;
let playing = false;
let speed   = 1;
let lastTs  = 0;
const MS_PER_STEP = 80;   // ~12.5 fps at 1×

function updateRobot(s) {{
  const t = TRAJ[Math.min(s, N-1)];
  robotGroup.position.set(t.x, 0.045, t.y);
  robotGroup.rotation.y = -t.theta;

  const isFail = FAILURE_TS !== null && s >= FAILURE_TS;
  const pulse  = isFail && Math.floor(Date.now() / 350) % 2 === 0;
  bodyMat.color.setHex(pulse ? 0xff4444 : (isFail ? 0xef4444 : 0x334155));
  ledMesh.material.color.setHex(isFail ? 0xef4444 : 0x22d3ee);
  ledMesh.material.emissive.setHex(isFail ? 0xef4444 : 0x22d3ee);

  // Build path
  for (let i = 0; i <= s && i < N; i++) {{
    const p = TRAJ[i];
    pathPositions[i*3]   = p.x;
    pathPositions[i*3+1] = 0.012;
    pathPositions[i*3+2] = p.y;
  }}
  pathGeo.setDrawRange(0, Math.min(s+1, N));
  pathGeo.attributes.position.needsUpdate = true;
  pathMat.color.setHex(isFail ? 0xef4444 : 0x22c55e);

  // UI
  document.getElementById('step-info').textContent =
    `Step ${{s}} / ${{N-1}}${{isFail ? '  ⚠ FAILURE' : ''}}`;
  scrubEl.value = s;
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (step >= N - 1) step = 0;
}}

function setSpeed(s) {{
  speed = s;
  document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}}

function restart() {{
  step = 0; playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  updateRobot(0);
}}

function onScrub(val) {{
  step = parseInt(val);
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  updateRobot(step);
}}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate(ts) {{
  requestAnimationFrame(animate);
  if (playing) {{
    if (ts - lastTs > MS_PER_STEP / speed) {{
      lastTs = ts;
      step   = Math.min(step + 1, N - 1);
      if (step >= N - 1) playing = false;
      updateRobot(step);
    }}
  }} else {{
    // Still re-render for pulse effect
    if (FAILURE_TS !== null && step >= FAILURE_TS) updateRobot(step);
  }}
  renderer.render(scene, camera);
}}

window.addEventListener('resize', resize);
resize();
updateRobot(0);
animate(0);
</script>
</body>
</html>"""

    enc = base64.b64encode(inner.encode("utf-8")).decode()
    return (
        f'<iframe src="data:text/html;base64,{enc}" '
        f'style="width:100%;height:520px;border:none;border-radius:8px;" '
        f'scrolling="no"></iframe>'
    )


# ── CSV generation ────────────────────────────────────────────────────────────
def _make_csv(records: list[dict]) -> str:
    if not records:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=records[0].keys())
    w.writeheader()
    w.writerows(records)
    return buf.getvalue()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div style="
  background:linear-gradient(135deg,#0f172a 0%,#1a0a0a 100%);
  border-bottom:2px solid #ef4444;
  padding:20px 28px 16px;
  border-radius:8px 8px 0 0;
  display:flex;align-items:center;gap:16px;
">
  <div>
    <div style="font-size:26px;font-weight:900;color:#f1f5f9;letter-spacing:-1px;">Haptal</div>
    <div style="font-size:12px;color:#94a3b8;letter-spacing:0.6px;margin-top:2px;">
      Human-in-the-Loop Failure Annotation
    </div>
  </div>
  <div style="margin-left:auto;font-size:11px;color:#475569;text-align:right;">
    Label robot failures.<br>Curate your training data.
  </div>
</div>
"""

css = """
body,.gradio-container{background:#0f172a!important;color:#f1f5f9!important}
.gr-button-primary{background:#ef4444!important;border-color:#ef4444!important}
.gr-button{border-radius:6px!important;font-weight:600!important}
.gr-form,.gr-box{background:#1e293b!important;border-color:#334155!important}
label{color:#94a3b8!important;font-size:12px!important}
.gr-input,textarea,select{background:#0f172a!important;color:#f1f5f9!important;border-color:#334155!important}
.gr-markdown h3,.gr-markdown h4{color:#f1f5f9!important}
.gr-markdown table{border-collapse:collapse;width:100%}
.gr-markdown td,.gr-markdown th{border:1px solid #334155;padding:6px 10px;font-size:12px}
.gr-markdown th{background:#1e293b;color:#94a3b8}
#pass-btn{background:#16a34a!important;border-color:#16a34a!important;font-size:18px!important;height:56px!important}
#reject-btn{background:#ef4444!important;border-color:#ef4444!important;font-size:18px!important;height:56px!important}
"""

with gr.Blocks(title="Haptal HITL Trajectory Reviewer", css=css, theme=gr.themes.Base()) as demo:

    # ── Session state ──────────────────────────────────────────────────────────
    session_id_state  = gr.State(str(uuid.uuid4()))
    start_time_state  = gr.State(time.time())
    ep_ids_state      = gr.State([])
    ep_idx_state      = gr.State(0)
    ep_data_state     = gr.State({})
    passed_state      = gr.State([])
    rejected_state    = gr.State([])

    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Left: dataset selector + viz ──────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Row():
                ds_dd = gr.Dropdown(
                    choices=DATASETS,
                    value=DATASETS[0],
                    label="Dataset",
                    scale=3,
                )
                ep_num = gr.Number(value=0, label="Episode #", precision=0, scale=1)

            with gr.Row():
                btn_prev  = gr.Button("← Prev", scale=1)
                btn_load  = gr.Button("Load Episode", variant="primary", scale=2)
                btn_next  = gr.Button("Next →", scale=1)

            anim_html = gr.HTML(
                '<div style="background:#0f172a;height:520px;border-radius:8px;'
                'display:flex;align-items:center;justify-content:center;'
                'color:#475569;font-size:14px;">Select a dataset and click Load Episode</div>'
            )
            status_msg = gr.Markdown()

        # ── Right: review panel ───────────────────────────────────────────────
        with gr.Column(scale=2):
            ep_info_md  = gr.Markdown("### Episode Info\n*Load an episode to begin.*")
            progress_md = gr.Markdown("**Progress:** 0 episodes reviewed")
            tally_md    = gr.Markdown("✅ Passed: 0 &nbsp;&nbsp; ❌ Rejected: 0")

            gr.Markdown("---")
            gr.Markdown("### Your Decision")

            reject_reason = gr.Textbox(
                label="Rejection reason (optional)",
                placeholder="e.g. false positive, mislabeled class...",
                lines=2,
            )
            with gr.Row():
                pass_btn   = gr.Button("✓ Pass",   elem_id="pass-btn",   variant="primary")
                reject_btn = gr.Button("✗ Reject", elem_id="reject-btn", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### Download Curated Dataset")
            download_section = gr.Column(visible=False)
            with download_section:
                dl_email = gr.Textbox(
                    label="Enter email to download and receive updates",
                    placeholder="you@company.com",
                )
                with gr.Row():
                    dl_pass_btn   = gr.Button("⬇ Download Clean Dataset (Passed)", variant="primary")
                    dl_reject_btn = gr.Button("⬇ Download Rejection Log")
                dl_pass_file   = gr.File(label="Clean Dataset CSV", visible=False)
                dl_reject_file = gr.File(label="Rejection Log CSV", visible=False)

    gr.HTML(
        f'<div style="text-align:center;padding:16px;color:#475569;font-size:11px;">'
        f'Built by <a href="https://haptal.ai" style="color:#ef4444;">Haptal</a> · '
        f'<a href="mailto:{CTA_EMAIL}" style="color:#ef4444;">{CTA_EMAIL}</a> · '
        f'<a href="https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark" '
        f'style="color:#ef4444;">Failure Benchmark</a></div>'
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def load_episode(dataset_name, ep_number, session_id, start_time):
        try:
            ep_number = int(ep_number)
            ep_ids    = _get_episode_ids(dataset_name)

            if dataset_name == "HaptalAI/amr-navigation-failure-dataset":
                if ep_number < 0 or ep_number >= len(ep_ids):
                    return (
                        gr.update(), f"⚠ Episode {ep_number} out of range (0–{len(ep_ids)-1})",
                        gr.update(), gr.update(), ep_ids, ep_number, {}
                    )
                ep_id = ep_ids[ep_number]
            else:
                ep_id = str(ep_number)

            ep_data = _extract_trajectory(dataset_name, ep_id)
            html    = _make_animation_html(ep_data)

            ft  = ep_data["failure_timestep"]
            fc  = ep_data["failure_class"]
            conf = ep_data["confidence"]

            info_md = (
                f"### Episode Info\n"
                f"| Field | Value |\n|---|---|\n"
                f"| **Episode ID** | `{ep_data['episode_id']}` |\n"
                f"| **Dataset** | `{dataset_name.split('/')[-1]}` |\n"
                f"| **Total timesteps** | {ep_data['total_timesteps']} |\n"
                f"| **Predicted class** | `{fc}` |\n"
                f"| **Confidence** | {conf:.0%} |\n"
                f"| **Anomaly timestep** | {'step ' + str(ft) if ft is not None else 'none'} |\n"
            )

            return html, "", info_md, ep_number, ep_ids, ep_number, ep_data

        except Exception as e:
            return (
                gr.update(), f"❌ {str(e)}",
                gr.update(), gr.update(), [], 0, {}
            )

    def on_load(dataset_name, ep_number, session_id, start_time):
        html, msg, info, ep_num_out, ep_ids, ep_idx, ep_data = load_episode(
            dataset_name, ep_number, session_id, start_time
        )
        return html, msg, info, ep_num_out, ep_ids, ep_idx, ep_data

    def on_prev(ep_idx, ep_ids, dataset_name, session_id, start_time):
        new_idx = max(0, ep_idx - 1)
        html, msg, info, ep_num_out, ep_ids_out, ep_idx_out, ep_data = load_episode(
            dataset_name, new_idx, session_id, start_time
        )
        return html, msg, info, new_idx, ep_ids_out, ep_idx_out, ep_data

    def on_next(ep_idx, ep_ids, dataset_name, session_id, start_time):
        max_idx = max(0, len(ep_ids) - 1)
        new_idx = min(max_idx, ep_idx + 1)
        html, msg, info, ep_num_out, ep_ids_out, ep_idx_out, ep_data = load_episode(
            dataset_name, new_idx, session_id, start_time
        )
        return html, msg, info, new_idx, ep_ids_out, ep_idx_out, ep_data

    def on_decision(decision, reason, ep_data, dataset_name, session_id,
                    passed, rejected, ep_ids, ep_idx, start_time):
        if not ep_data:
            return (gr.update(), gr.update(), gr.update(),
                    passed, rejected, gr.update(), gr.update(), ep_idx)

        ts  = datetime.datetime.utcnow().isoformat() + "Z"
        rec = {
            "episode_id":          ep_data.get("episode_id", ""),
            "dataset":             dataset_name,
            "human_label":         decision,
            "rejection_reason":    reason if decision == "reject" else "",
            "reviewer_session_id": session_id,
            "review_timestamp":    ts,
        }

        payload = {
            "session_id":       session_id,
            "episode_id":       rec["episode_id"],
            "dataset":          dataset_name,
            "decision":         decision,
            "rejection_reason": rec["rejection_reason"],
            "timestamp":        ts,
        }
        _fire(payload)
        _log_lead({**payload, "type": "decision"})

        if decision == "pass":
            passed = passed + [rec]
        else:
            rejected = rejected + [rec]

        total   = len(passed) + len(rejected)
        prog_md = f"**Progress:** {total} episode{'s' if total!=1 else ''} reviewed"
        tally   = f"✅ Passed: {len(passed)} &nbsp;&nbsp; ❌ Rejected: {len(rejected)}"

        # Auto-advance
        max_idx  = max(0, len(ep_ids) - 1)
        new_idx  = min(max_idx, ep_idx + 1)
        show_dl  = total >= 5

        return prog_md, tally, gr.update(visible=show_dl), passed, rejected, new_idx, new_idx

    def on_dl_pass(email, passed, rejected, dataset_name, session_id, start_time):
        if not email or "@" not in email:
            return gr.update(visible=False), gr.update(visible=False)
        ts    = datetime.datetime.utcnow().isoformat() + "Z"
        total = len(passed) + len(rejected)
        secs  = round(time.time() - start_time)
        payload = {
            "type": "download_pass", "session_id": session_id, "email": email,
            "dataset_used": dataset_name, "pass_count": len(passed),
            "reject_count": len(rejected), "total_reviewed": total,
            "time_spent_seconds": secs, "timestamp": ts,
        }
        _fire(payload)
        _log_lead(payload)

        if not passed:
            return gr.update(visible=True), gr.update(visible=False)

        csv_str  = _make_csv(passed)
        tmp_path = Path("/tmp/haptal_curated_passed.csv")
        tmp_path.write_text(csv_str)
        return gr.update(value=str(tmp_path), visible=True), gr.update(visible=False)

    def on_dl_reject(email, passed, rejected, dataset_name, session_id, start_time):
        if not email or "@" not in email:
            return gr.update(visible=False), gr.update(visible=False)
        ts    = datetime.datetime.utcnow().isoformat() + "Z"
        total = len(passed) + len(rejected)
        secs  = round(time.time() - start_time)
        payload = {
            "type": "download_reject", "session_id": session_id, "email": email,
            "dataset_used": dataset_name, "pass_count": len(passed),
            "reject_count": len(rejected), "total_reviewed": total,
            "time_spent_seconds": secs, "timestamp": ts,
        }
        _fire(payload)
        _log_lead(payload)

        if not rejected:
            return gr.update(visible=False), gr.update(visible=True)

        csv_str  = _make_csv(rejected)
        tmp_path = Path("/tmp/haptal_curated_rejected.csv")
        tmp_path.write_text(csv_str)
        return gr.update(visible=False), gr.update(value=str(tmp_path), visible=True)

    # ── Wire events ────────────────────────────────────────────────────────────

    load_outputs = [anim_html, status_msg, ep_info_md, ep_num, ep_ids_state, ep_idx_state, ep_data_state]

    btn_load.click(on_load, [ds_dd, ep_num, session_id_state, start_time_state], load_outputs)
    btn_prev.click(on_prev, [ep_idx_state, ep_ids_state, ds_dd, session_id_state, start_time_state], load_outputs)
    btn_next.click(on_next, [ep_idx_state, ep_ids_state, ds_dd, session_id_state, start_time_state], load_outputs)

    decision_outputs = [progress_md, tally_md, download_section,
                        passed_state, rejected_state, ep_idx_state, ep_num]

    pass_btn.click(
        lambda reason, ep, ds, sid, p, r, ids, idx, st: on_decision(
            "pass", reason, ep, ds, sid, p, r, ids, idx, st),
        [reject_reason, ep_data_state, ds_dd, session_id_state,
         passed_state, rejected_state, ep_ids_state, ep_idx_state, start_time_state],
        decision_outputs,
    )
    reject_btn.click(
        lambda reason, ep, ds, sid, p, r, ids, idx, st: on_decision(
            "reject", reason, ep, ds, sid, p, r, ids, idx, st),
        [reject_reason, ep_data_state, ds_dd, session_id_state,
         passed_state, rejected_state, ep_ids_state, ep_idx_state, start_time_state],
        decision_outputs,
    )

    dl_pass_btn.click(
        on_dl_pass,
        [dl_email, passed_state, rejected_state, ds_dd, session_id_state, start_time_state],
        [dl_pass_file, dl_reject_file],
    )
    dl_reject_btn.click(
        on_dl_reject,
        [dl_email, passed_state, rejected_state, ds_dd, session_id_state, start_time_state],
        [dl_pass_file, dl_reject_file],
    )

    # Session start log on first load
    def _session_start(session_id):
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        payload = {"type": "session_start", "session_id": session_id, "timestamp": ts}
        _fire(payload)
        _log_lead(payload)

    demo.load(_session_start, inputs=[session_id_state], outputs=[])


if __name__ == "__main__":
    demo.launch()
