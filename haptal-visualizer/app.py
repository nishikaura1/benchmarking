"""
Haptal Robotics 3D Trajectory Visualizer
==========================================
HuggingFace Space — Gradio app with two tabs:
  Tab 1: Quick Demo — stream episodes from 4 LeRobot datasets
  Tab 2: Analyze Your Data — upload CSV/parquet, log leads, quality summary
"""

import json
import os
import re
import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# ─── Config ───────────────────────────────────────────────────────────────────
WEBHOOK_URL = ""   # set to your webhook endpoint, e.g. "https://hooks.zapier.com/..."
LEADS_FILE  = "leads.json"
CTA_EMAIL   = "aarav@haptal.ai"

DATASETS = {
    "lerobot/xarm_lift_medium_replay":       {"max_ep": 200, "label": "xArm Lift Medium Replay"},
    "lerobot/xarm_push_medium_replay":       {"max_ep": 200, "label": "xArm Push Medium Replay"},
    "lerobot/aloha_sim_transfer_cube_human": {"max_ep":  50, "label": "ALOHA Sim Transfer Cube"},
    "lerobot/aloha_sim_insertion_human":     {"max_ep":  50, "label": "ALOHA Sim Insertion"},
}

TRAJ_KEYWORDS = ["pos", "joint", "state", "obs", "action"]

# ─── Caches ───────────────────────────────────────────────────────────────────
_dataset_cache: dict[str, pd.DataFrame] = {}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_lerobot(dataset_name: str) -> pd.DataFrame:
    """Load and cache a LeRobot dataset as a flat DataFrame."""
    if dataset_name in _dataset_cache:
        return _dataset_cache[dataset_name]

    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()
    _dataset_cache[dataset_name] = df
    return df


def _get_episode_df(df: pd.DataFrame, episode_idx: int) -> pd.DataFrame:
    """Extract rows for a single episode_index."""
    col = next((c for c in df.columns if "episode" in c.lower()), None)
    if col is None:
        raise ValueError("No episode_index column found in dataset.")
    ep_df = df[df[col] == episode_idx].reset_index(drop=True)
    return ep_df, col


def _extract_state_array(ep_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    Pull the first usable state/observation column.
    LeRobot stores arrays as objects — unpack them.
    Returns (array of shape (T, D), column_name).
    """
    preferred = ["observation.state", "observation.qpos", "state"]
    cols_to_try = preferred + [c for c in ep_df.columns if any(k in c.lower() for k in TRAJ_KEYWORDS)]

    for col in cols_to_try:
        if col not in ep_df.columns:
            continue
        sample = ep_df[col].iloc[0]
        if hasattr(sample, "__len__") and not isinstance(sample, str):
            arr = np.vstack(ep_df[col].values).astype(np.float32)
        else:
            try:
                arr = ep_df[col].values.astype(np.float32).reshape(-1, 1)
            except Exception:
                continue
        if arr.shape[0] > 1:
            return arr, col

    raise ValueError("No usable trajectory column found.")


def _make_3d_plot(arr: np.ndarray, title: str) -> go.Figure:
    """
    Build a 3D Plotly trajectory.
    Uses first 3 dimensions of state array.
    Color: blue (start) → red (end).
    """
    T = arr.shape[0]
    D = arr.shape[1]

    x = arr[:, 0]
    y = arr[:, 1] if D > 1 else np.zeros(T)
    z = arr[:, 2] if D > 2 else np.arange(T, dtype=np.float32)

    colors = np.linspace(0, 1, T)
    colorscale = [[0, "#3b82f6"], [0.5, "#a855f7"], [1, "#ef4444"]]

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        marker=dict(
            size=3,
            color=colors,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Time",
                tickvals=[0, 1],
                ticktext=["Start", "End"],
                thickness=12,
                len=0.6,
            ),
        ),
        line=dict(color="rgba(255,255,255,0.15)", width=2),
        hovertemplate="Step %{text}<br>dim0=%{x:.4f}<br>dim1=%{y:.4f}<br>dim2=%{z:.4f}",
        text=[str(i) for i in range(T)],
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=14)),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        scene=dict(
            bgcolor="#0f172a",
            xaxis=dict(title="dim 0", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            yaxis=dict(title="dim 1", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            zaxis=dict(title="dim 2 / time", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
        ),
        font=dict(color="#f1f5f9"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig


def _log_lead(email: str, filename: str, n_rows: int, columns: list[str]):
    """Append lead info to leads.json and fire webhook."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "email":     email,
        "filename":  filename,
        "row_count": n_rows,
        "columns":   columns,
    }

    # Local log
    existing = []
    if Path(LEADS_FILE).exists():
        try:
            with open(LEADS_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(LEADS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    # Webhook (fail silently)
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=entry, timeout=5)
        except Exception:
            pass


# ─── Tab 1: Quick Demo ────────────────────────────────────────────────────────

def visualize_demo(dataset_name: str, episode_num: int):
    """Load a LeRobot episode and return (fig, info_markdown)."""
    info = DATASETS.get(dataset_name, {})
    max_ep = info.get("max_ep", 200)

    if episode_num < 0 or episode_num >= max_ep:
        err = (
            f"⚠️ Episode {episode_num} is out of range for **{dataset_name}**.\n\n"
            f"Valid episode numbers: **0 – {max_ep - 1}**."
        )
        return None, err

    try:
        df = _load_lerobot(dataset_name)

        # Verify actual max from data
        col = next((c for c in df.columns if "episode" in c.lower()), None)
        if col:
            actual_max = int(df[col].max()) + 1
            if episode_num >= actual_max:
                return None, (
                    f"⚠️ Episode {episode_num} not found in **{dataset_name}**.\n\n"
                    f"This dataset contains episodes **0 – {actual_max - 1}**."
                )

        ep_df, ep_col = _get_episode_df(df, episode_num)
        if len(ep_df) == 0:
            return None, f"⚠️ Episode {episode_num} not found. Try a number between 0 and {max_ep - 1}."

        arr, state_col = _extract_state_array(ep_df)
        fig  = _make_3d_plot(arr, f"{dataset_name} — Episode {episode_num}")

        col_names = list(ep_df.columns[:8])  # show first 8
        info_md = (
            f"**Episode:** {episode_num}  \n"
            f"**Dataset:** `{dataset_name}`  \n"
            f"**Total timesteps:** {len(ep_df)}  \n"
            f"**State column used:** `{state_col}` ({arr.shape[1]} dims)  \n"
            f"**Sample columns:** `{', '.join(col_names)}`"
        )
        return fig, info_md

    except Exception as e:
        return None, f"❌ Error loading dataset: {str(e)}"


# ─── Tab 2: Analyze Your Data ─────────────────────────────────────────────────

def analyze_upload(file, email: str):
    """Parse uploaded CSV/parquet, log lead, return (fig, summary_md, cta_visible)."""

    # Email validation
    if not email or "@" not in email:
        return None, "⚠️ Please enter a valid email address before analyzing.", gr.update(visible=False)

    if file is None:
        return None, "⚠️ Please upload a file first.", gr.update(visible=False)

    # Load file
    filepath = file.name
    filename  = Path(filepath).name
    ext = Path(filepath).suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(filepath)
        else:
            return None, "❌ Unsupported file type. Please upload a CSV or Parquet file.", gr.update(visible=False)
    except Exception as e:
        return None, f"❌ Could not read file: {str(e)}", gr.update(visible=False)

    # Log lead immediately
    _log_lead(email, filename, len(df), list(df.columns))

    # Find trajectory columns
    traj_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in TRAJ_KEYWORDS)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not traj_cols:
        return (
            None,
            (
                "⚠️ We could not find trajectory columns in your file.\n\n"
                "We look for columns containing: **pos, joint, state, obs, action**.\n\n"
                "Please check your file format and try again."
            ),
            gr.update(visible=False),
        )

    # Build trajectory array (up to first 3 traj cols for 3D plot)
    arr = df[traj_cols].dropna().values.astype(np.float32)
    T, D = arr.shape

    fig = _make_3d_plot(arr[:, :3] if D >= 3 else arr, f"Uploaded: {filename}")

    # Quality summary
    # Detect episode breaks — look for episode_index column
    ep_col = next((c for c in df.columns if "episode" in c.lower()), None)
    n_episodes = int(df[ep_col].nunique()) if ep_col else 1

    # Velocity spikes: timesteps where any traj col exceeds mean ± 2σ
    means = arr.mean(axis=0)
    stds  = arr.std(axis=0) + 1e-8
    z_scores = np.abs((arr - means) / stds)
    spike_mask = (z_scores > 2.0).any(axis=1)
    n_spikes = int(spike_mask.sum())
    quality_score = round(100 * (1 - n_spikes / max(T, 1)), 1)
    quality_color = "🟢" if quality_score >= 85 else "🟡" if quality_score >= 65 else "🔴"

    summary_md = f"""
### Quality Summary

| Metric | Value |
|--------|-------|
| **Episodes detected** | {n_episodes} |
| **Total timesteps** | {T:,} |
| **Columns used** | `{", ".join(traj_cols[:6])}{"..." if len(traj_cols) > 6 else ""}` |
| **Velocity spike timesteps** | {n_spikes:,} ({round(100*n_spikes/max(T,1),1)}%) |
| **Quality score** | {quality_color} {quality_score} / 100 |

*Quality score = % of timesteps with no detected velocity anomalies (>2σ). Higher is better.*

---
*Full failure attribution — including per-timestep failure class labels, physics-grounded root cause, and training-ready annotations — is available by contacting us below.*
"""
    return fig, summary_md, gr.update(visible=True)


# ─── Build UI ─────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div style="
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border-bottom: 2px solid #ef4444;
    padding: 24px 32px 20px;
    margin-bottom: 8px;
    border-radius: 8px 8px 0 0;
">
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:28px; font-weight:900; color:#f1f5f9; letter-spacing:-1px;">Haptal</span>
        <span style="
            background:#ef4444;
            color:white;
            font-size:10px;
            font-weight:700;
            padding:3px 8px;
            border-radius:4px;
            letter-spacing:1px;
            text-transform:uppercase;
        ">Beta</span>
    </div>
    <div style="color:#94a3b8; font-size:14px; margin-top:4px; letter-spacing:0.5px;">
        Robotics Failure Intelligence
    </div>
</div>
"""

CTA_HTML = f"""
<div style="
    background: #1c0a0a;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
">
    <div>
        <div style="color:#f1f5f9; font-weight:700; font-size:14px;">Want full failure attribution on your data?</div>
        <div style="color:#94a3b8; font-size:12px; margin-top:3px;">Per-timestep failure class labels · Physics root cause · Training-ready annotations</div>
    </div>
    <a href="mailto:{CTA_EMAIL}" style="
        background:#ef4444;
        color:white;
        padding:10px 20px;
        border-radius:6px;
        font-weight:700;
        font-size:13px;
        text-decoration:none;
        white-space:nowrap;
    ">Get Full Failure Attribution Report →</a>
</div>
"""

css = """
body, .gradio-container { background: #0f172a !important; color: #f1f5f9 !important; }
.gr-button-primary { background: #ef4444 !important; border-color: #ef4444 !important; }
.gr-button { border-radius: 6px !important; font-weight: 600 !important; }
.gr-form, .gr-box { background: #1e293b !important; border-color: #334155 !important; }
label { color: #94a3b8 !important; font-size: 13px !important; }
.gr-input, .gr-dropdown select, textarea { background: #0f172a !important; color: #f1f5f9 !important; border-color: #334155 !important; }
.gr-markdown h3 { color: #f1f5f9 !important; }
.gr-markdown table { border-collapse: collapse; width: 100%; }
.gr-markdown td, .gr-markdown th { border: 1px solid #334155; padding: 8px 12px; font-size: 13px; }
.gr-markdown th { background: #1e293b; color: #94a3b8; }
.gr-tab-button { background: #1e293b !important; color: #94a3b8 !important; border-color: #334155 !important; }
.gr-tab-button.selected { color: #f1f5f9 !important; border-bottom-color: #ef4444 !important; }
"""

with gr.Blocks(title="Haptal Robotics 3D Trajectory Visualizer", css=css, theme=gr.themes.Base()) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1 ─────────────────────────────────────────────────────────────
        with gr.Tab("🚀 Quick Demo"):
            gr.Markdown(
                "Visualize 3D joint trajectories from four benchmark LeRobot datasets. "
                "Select a dataset, enter an episode number, and hit **Visualize**."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    ds_dropdown = gr.Dropdown(
                        choices=list(DATASETS.keys()),
                        value="lerobot/xarm_lift_medium_replay",
                        label="Dataset",
                        interactive=True,
                    )
                    ep_input = gr.Number(
                        value=0,
                        label="Episode number",
                        precision=0,
                        minimum=0,
                        interactive=True,
                    )
                    viz_btn = gr.Button("Visualize →", variant="primary")

                    gr.Markdown(
                        "**Episode ranges:**\n"
                        "- xarm_lift_medium_replay: 0 – 199\n"
                        "- xarm_push_medium_replay: 0 – 199\n"
                        "- aloha_sim_transfer_cube_human: 0 – 49\n"
                        "- aloha_sim_insertion_human: 0 – 49\n\n"
                        "🔵 Blue = start of episode &nbsp;&nbsp; 🔴 Red = end"
                    )

            demo_plot  = gr.Plot(label="3D Trajectory")
            demo_info  = gr.Markdown()

            viz_btn.click(
                fn=visualize_demo,
                inputs=[ds_dropdown, ep_input],
                outputs=[demo_plot, demo_info],
            )

        # ── Tab 2 ─────────────────────────────────────────────────────────────
        with gr.Tab("📂 Analyze Your Data"):
            gr.Markdown(
                "### Upload your robot trajectory dataset to get a 3D visualization and failure summary\n\n"
                "Accepted formats: **CSV** or **Parquet**. "
                "We look for columns containing: `pos`, `joint`, `state`, `obs`, or `action`."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload CSV or Parquet",
                        file_types=[".csv", ".parquet", ".pq"],
                    )
                    email_input = gr.Textbox(
                        label="Enter your email to receive your full failure attribution report",
                        placeholder="you@company.com",
                    )
                    analyze_btn = gr.Button("Analyze →", variant="primary")

            upload_plot   = gr.Plot(label="3D Trajectory")
            upload_summary = gr.Markdown()
            cta_html = gr.HTML(CTA_HTML, visible=False)

            analyze_btn.click(
                fn=analyze_upload,
                inputs=[file_upload, email_input],
                outputs=[upload_plot, upload_summary, cta_html],
            )

    gr.HTML(
        f'<div style="text-align:center; padding:20px; color:#475569; font-size:12px;">'
        f'Built by <a href="https://haptal.ai" style="color:#ef4444;">Haptal</a> · '
        f'<a href="https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark" style="color:#ef4444;">Failure Benchmark Dataset</a> · '
        f'<a href="mailto:{CTA_EMAIL}" style="color:#ef4444;">{CTA_EMAIL}</a>'
        f'</div>'
    )


if __name__ == "__main__":
    demo.launch()
