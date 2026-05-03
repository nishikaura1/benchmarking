"""
Client Demo Dashboard — Robotics Annotation Platform
Run: streamlit run client_demo.py

Five sections:
  1. How It Works      — step-by-step pipeline walkthrough
  2. Failure Detection — live model performance, ROC-AUC, confusion matrix
  3. Step Annotations  — per-timestep failure types + semantic labels + 3D view
  4. Review Queue      — human-in-the-loop low-confidence correction interface
  5. Analytics         — UMAP failure clustering + quality distribution
"""

import json, h5py, pickle, warnings, subprocess, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Haptal AI · Robot Data Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = Path("benchmark_output")

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* dark base */
  .stApp { background:#0a0f1e; }
  section[data-testid="stSidebar"] { background:#0d1424; border-right:1px solid #1e2d45; }

  /* metric cards */
  div[data-testid="metric-container"] {
    background:#0d1a2e;
    border:1px solid #1e3a5f;
    border-radius:10px;
    padding:16px 20px;
  }
  div[data-testid="metric-container"] label { color:#64b5f6 !important; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; }
  div[data-testid="metric-container"] div[data-testid="metric-delta"] { font-size:.78rem; }

  /* section header */
  .section-header {
    background:linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
    border-radius:14px;
    padding:2rem 2.4rem 1.6rem;
    margin-bottom:1.6rem;
    border:1px solid #1e3a5f;
  }
  .section-header h1 { color:#fff; margin:0; font-size:2rem; font-weight:700; }
  .section-header p  { color:#90caf9; margin:.35rem 0 0; font-size:1.05rem; }

  /* step cards */
  .step-card {
    background:#0d1a2e;
    border:1px solid #1e3a5f;
    border-radius:12px;
    padding:1.4rem 1.6rem;
    height:100%;
    transition:border-color .2s;
  }
  .step-card:hover { border-color:#3b82f6; }
  .step-num { font-size:2rem; font-weight:800; color:#3b82f6; margin-bottom:.4rem; }
  .step-title { color:#e2e8f0; font-size:1rem; font-weight:600; margin-bottom:.5rem; }
  .step-body { color:#94a3b8; font-size:.88rem; line-height:1.55; }

  /* layer pills */
  .layer-pill {
    display:inline-block;
    background:#1e3a5f;
    color:#90caf9;
    border-radius:6px;
    padding:3px 10px;
    font-size:.78rem;
    font-weight:600;
    margin:2px;
  }

  /* output file cards */
  .file-card {
    background:#0d1a2e;
    border:1px solid #1e3a5f;
    border-left:4px solid #3b82f6;
    border-radius:8px;
    padding:1rem 1.2rem;
    margin-bottom:.7rem;
  }
  .file-name { color:#60a5fa; font-family:monospace; font-size:.9rem; font-weight:700; }
  .file-desc { color:#94a3b8; font-size:.83rem; margin-top:.3rem; }

  /* perf badge */
  .perf-badge {
    background:#064e3b;
    border:1px solid #059669;
    color:#34d399;
    border-radius:8px;
    padding:6px 16px;
    font-size:.85rem;
    font-weight:700;
    display:inline-block;
    margin:3px;
  }
  .perf-badge-warn {
    background:#451a03;
    border:1px solid #d97706;
    color:#fbbf24;
    border-radius:8px;
    padding:6px 16px;
    font-size:.85rem;
    font-weight:700;
    display:inline-block;
    margin:3px;
  }

  /* divider */
  hr { border-color:#1e3a5f !important; }

  /* tabs */
  div[data-baseweb="tab-list"] { background:#0d1424; border-radius:8px; }
  button[data-baseweb="tab"] { color:#64b5f6; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1rem;'>
      <div style='font-size:1.6rem;font-weight:900;letter-spacing:-.5px;
                  background:linear-gradient(90deg,#60a5fa,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        ⚡ Haptal AI
      </div>
      <div style='color:#475569;font-size:.8rem;margin-top:.25rem;'>
        Robot Data Annotation Platform
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 How It Works", "🎯 Failure Detection", "🏷️ Step Annotations",
         "🔍 Review Queue", "📊 Analytics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='color:#475569;font-size:.78rem;line-height:1.7;'>
      <b style='color:#64b5f6;'>Models</b><br>
      IsolationForest (anomaly)<br>
      RandomForest (failure type)<br>
      RandomForest (semantic)<br><br>
      <b style='color:#64b5f6;'>Training data</b><br>
      xarm_lift · xarm_push<br>
      DROID · ALOHA (4 variants)<br>
      177 k training steps<br><br>
      <b style='color:#64b5f6;'>Validated on</b><br>
      LeRobot / HuggingFace<br>
      Real robot manipulation
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_benchmark_data():
    out = []
    for card_path in sorted(OUTPUT_DIR.glob("*_card.json")):
        if "clip" in card_path.name:
            continue
        h5_path = card_path.with_name(card_path.stem.replace("_card", "_scores") + ".h5")
        if not h5_path.exists():
            continue
        card = json.loads(card_path.read_text())
        with h5py.File(h5_path) as f:
            scores = f["anomaly_scores"][:]
            labels = f["true_labels"][:]
            preds  = f["predictions"][:]
        out.append({"card": card, "scores": scores, "labels": labels,
                    "preds": preds, "name": card.get("dataset", card_path.stem)})
    return out

@st.cache_data
def load_pipeline_report():
    p = OUTPUT_DIR / "_demo_input_report.json"
    s = OUTPUT_DIR / "_demo_input_summary.json"
    if not p.exists():
        return None, None
    return json.loads(p.read_text()), json.loads(s.read_text())

def load_review_queue():
    p = OUTPUT_DIR / "_demo_input_review_queue.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

def load_corrections():
    p = OUTPUT_DIR / "corrections.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

def save_corrections(corrections: list):
    p = OUTPUT_DIR / "corrections.json"
    p.write_text(json.dumps(corrections, indent=2))

@st.cache_data
def load_annotations():
    out = []
    for p in sorted(OUTPUT_DIR.glob("*_annotations.json")):
        out.append(json.loads(p.read_text()))
    return out

def dark_chart(fig, height=360, legend_h=True):
    extra = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if legend_h else {}
    fig.update_layout(
        height=height, margin=dict(t=36, b=36, l=10, r=10),
        plot_bgcolor="#060d1a", paper_bgcolor="#0a0f1e",
        font_color="#cbd5e1", font_size=12,
        legend=extra if legend_h else {},
        xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
    )
    return fig

FAIL_COLOR = {
    "nominal":              "#3b82f6",
    "velocity_spike":       "#ef4444",
    "position_jerk":        "#f97316",
    "stuck_joint":          "#a855f7",
    "gripper_event":        "#eab308",
    "high_anomaly":         "#ec4899",
    "self_collision":       "#dc2626",
    "overshoot":            "#f59e0b",
    "trajectory_deviation": "#06b6d4",
    "perception_failure":   "#8b5cf6",
}
FAIL_LABEL = {
    "nominal":              "Normal operation",
    "velocity_spike":       "Velocity spike",
    "position_jerk":        "Position jerk",
    "stuck_joint":          "Stuck joint",
    "gripper_event":        "Gripper event",
    "high_anomaly":         "High anomaly",
    "self_collision":       "Self collision",
    "overshoot":            "Overshoot",
    "trajectory_deviation": "Trajectory deviation",
    "perception_failure":   "Perception failure",
}
ALL_FAILURE_CLASSES = list(FAIL_COLOR.keys())
SEM_COLOR = {
    "approaching": "#3b82f6", "grasping": "#8b5cf6",
    "transporting": "#06b6d4", "placing": "#10b981",
    "returning": "#f59e0b",   "idle": "#6b7280",
    "home": "#3b82f6",        "near_object": "#8b5cf6",
    "mid_transit": "#06b6d4", "near_target": "#10b981",
    "boundary": "#ef4444",
    "no_contact": "#3b82f6",  "pre_grasp": "#f59e0b",
    "in_grasp": "#10b981",    "releasing": "#ef4444",
    "stationary": "#6b7280",  "slow_move": "#3b82f6",
    "fast_move": "#ef4444",   "decelerating": "#f59e0b",
    "rotating": "#8b5cf6",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 How It Works":

    st.markdown("""
    <div class='section-header'>
      <h1>🤖 Robotics Annotation Platform</h1>
      <p>Automatically detect failures and label every timestep of robot operation —
         from raw sensor data to fully annotated reports, in minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Value prop strip ──────────────────────────────────────────────────────
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Failure Detection", "ROC-AUC 0.943", "xarm manipulation")
    v2.metric("Step Annotation Acc.", "92 %", "failure type labeling")
    v3.metric("Semantic Labels Acc.", "94.9 %", "task phase detection")
    v4.metric("Training Steps", "177 k", "DROID + ALOHA + xarm")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Workflow steps ────────────────────────────────────────────────────────
    st.markdown("### How a client engagement works")
    st.caption("Three inputs → three annotation layers → fully labeled output")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, arr1, c2, arr2, c3, arr3, c4 = st.columns([5, 1, 5, 1, 5, 1, 5])

    with c1:
        st.markdown("""
        <div class='step-card'>
          <div class='step-num'>01</div>
          <div class='step-title'>📋 Provide SOP Reference</div>
          <div class='step-body'>
            Upload an HDF5 file of <b>nominal / correct</b> robot sessions.
            This is your Standard Operating Procedure — what good operation looks like.<br><br>
            The platform learns the statistical fingerprint of normal behavior.
            No manual labeling required.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with arr1:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;font-size:1.8rem;color:#3b82f6;padding-top:3rem;'>→</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='step-card'>
          <div class='step-num'>02</div>
          <div class='step-title'>📂 Upload Production Data</div>
          <div class='step-body'>
            Upload the HDF5 file of robot sessions you want to analyze —
            field data, test runs, or teleoperation recordings.<br><br>
            Any robot format. Joint states, actions, and rewards
            are all optional — the platform adapts to what you have.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with arr2:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;font-size:1.8rem;color:#3b82f6;padding-top:3rem;'>→</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='step-card'>
          <div class='step-num'>03</div>
          <div class='step-title'>⚙️ 3-Layer Annotation</div>
          <div class='step-body'>
            <b>Layer 1</b> — Episode anomaly score vs SOP baseline<br><br>
            <b>Layer 2</b> — Per-timestep failure type:<br>
            velocity spike · position jerk · stuck joint · gripper event<br><br>
            <b>Layer 3</b> — Semantic labels per timestep:<br>
            task phase · workspace zone · contact state · motion type
          </div>
        </div>
        """, unsafe_allow_html=True)

    with arr3:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;font-size:1.8rem;color:#3b82f6;padding-top:3rem;'>→</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='step-card'>
          <div class='step-num'>04</div>
          <div class='step-title'>📦 Labeled Output Files</div>
          <div class='step-body'>
            <code style='color:#60a5fa;'>_annotated.h5</code> — same structure as input + all annotation layers written in<br><br>
            <code style='color:#60a5fa;'>_report.json</code> — full per-episode breakdown with every timestep labeled<br><br>
            <code style='color:#60a5fa;'>_summary.json</code> — model card, aggregate stats, confusion matrix
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── The 3 annotation layers detail ───────────────────────────────────────
    st.markdown("### Three annotation layers explained")
    st.markdown("<br>", unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)

    with l1:
        st.markdown("""
        <div class='step-card' style='border-top:3px solid #3b82f6;'>
          <div style='color:#3b82f6;font-size:.75rem;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;margin-bottom:.6rem;'>Layer 1 · Episode Level</div>
          <div style='color:#e2e8f0;font-size:1.05rem;font-weight:700;margin-bottom:.8rem;'>
            Anomaly Detection
          </div>
          <div style='color:#94a3b8;font-size:.87rem;line-height:1.6;'>
            <b style='color:#cbd5e1;'>Model:</b> IsolationForest<br>
            <b style='color:#cbd5e1;'>Input:</b> SOP reference file<br>
            <b style='color:#cbd5e1;'>Output:</b> anomaly score per episode<br>
            <b style='color:#cbd5e1;'>ROC-AUC:</b> 0.943 (xarm lift)<br>
            <b style='color:#cbd5e1;'>Detection rate:</b> 92.2 %<br><br>
            Scores each session against the normal distribution learned from the SOP.
            High score = the session looks different from nominal operation.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with l2:
        st.markdown("""
        <div class='step-card' style='border-top:3px solid #8b5cf6;'>
          <div style='color:#8b5cf6;font-size:.75rem;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;margin-bottom:.6rem;'>Layer 2 · Step Level</div>
          <div style='color:#e2e8f0;font-size:1.05rem;font-weight:700;margin-bottom:.8rem;'>
            Failure Type Labeling
          </div>
          <div style='color:#94a3b8;font-size:.87rem;line-height:1.6;'>
            <b style='color:#cbd5e1;'>Model:</b> RandomForest (weak supervision)<br>
            <b style='color:#cbd5e1;'>Training:</b> 3,175 steps · xarm_lift + push<br>
            <b style='color:#cbd5e1;'>Accuracy:</b> 92 %<br>
            <b style='color:#cbd5e1;'>Classes:</b> 6 failure types<br><br>
            Labels every timestep with the type of anomaly occurring.
            Rule-based physics heuristics seed the training labels —
            no manual annotation needed.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with l3:
        st.markdown("""
        <div class='step-card' style='border-top:3px solid #06b6d4;'>
          <div style='color:#06b6d4;font-size:.75rem;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;margin-bottom:.6rem;'>Layer 3 · Step Level</div>
          <div style='color:#e2e8f0;font-size:1.05rem;font-weight:700;margin-bottom:.8rem;'>
            Semantic Annotation
          </div>
          <div style='color:#94a3b8;font-size:.87rem;line-height:1.6;'>
            <b style='color:#cbd5e1;'>Model:</b> 4 × RandomForest classifiers<br>
            <b style='color:#cbd5e1;'>Training:</b> 177 k steps · DROID + ALOHA<br>
            <b style='color:#cbd5e1;'>Task phase acc.:</b> 94.9 %<br>
            <b style='color:#cbd5e1;'>Contact state acc.:</b> 99.9 %<br><br>
            Answers "what is the robot doing right now?" —
            task phase, workspace zone, contact state, and motion type,
            simultaneously per timestep.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Output format detail ──────────────────────────────────────────────────
    st.markdown("### What your annotated HDF5 contains")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class='file-card'>
          <div class='file-name'>/episode_0000/</div>
          <div class='file-desc'>One group per session — matches your original file structure exactly</div>
        </div>
        <div class='file-card' style='border-left-color:#8b5cf6;'>
          <div class='file-name'>→ anomaly_score  (scalar)</div>
          <div class='file-desc'>How anomalous this session is vs the SOP reference. Above threshold = flagged.</div>
        </div>
        <div class='file-card' style='border-left-color:#ef4444;'>
          <div class='file-name'>→ step_failure_types  (T,)</div>
          <div class='file-desc'>Per-timestep failure class: nominal · velocity_spike · position_jerk · stuck_joint · gripper_event · high_anomaly</div>
        </div>
        <div class='file-card' style='border-left-color:#f97316;'>
          <div class='file-name'>→ step_failure_confs  (T,)</div>
          <div class='file-desc'>Model confidence for each step label (0–1). Use this to filter high-confidence events.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='file-card' style='border-left-color:#06b6d4;'>
          <div class='file-name'>→ semantic_task_phase  (T,)</div>
          <div class='file-desc'>approaching · grasping · transporting · placing · returning · idle</div>
        </div>
        <div class='file-card' style='border-left-color:#10b981;'>
          <div class='file-name'>→ semantic_workspace_zone  (T,)</div>
          <div class='file-desc'>home · near_object · mid_transit · near_target · boundary</div>
        </div>
        <div class='file-card' style='border-left-color:#a855f7;'>
          <div class='file-name'>→ semantic_contact_state  (T,)</div>
          <div class='file-desc'>no_contact · pre_grasp · in_grasp · releasing</div>
        </div>
        <div class='file-card' style='border-left-color:#eab308;'>
          <div class='file-name'>→ semantic_motion_type  (T,)</div>
          <div class='file-desc'>stationary · slow_move · fast_move · decelerating · rotating</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ground truth note ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#0d1a2e;border:1px solid #1e3a5f;border-radius:10px;padding:1.4rem 1.8rem;'>
      <div style='color:#60a5fa;font-weight:700;font-size:.95rem;margin-bottom:.6rem;'>
        💡 How we validate model performance on your data
      </div>
      <div style='color:#94a3b8;font-size:.87rem;line-height:1.65;'>
        For <b style='color:#cbd5e1;'>open-source benchmark data</b> (xarm, DROID, ALOHA),
        ground truth comes from the reward signal: episodes with reward &gt; 0.5 are nominal,
        the bottom 20 % by reward are failures. This validates ROC-AUC and detection rate.<br><br>
        For <b style='color:#cbd5e1;'>client data</b> without labels, the SOP file
        defines "normal." The platform flags statistical deviations.
        Clients review flagged sessions, confirm or correct labels,
        and those corrections feed back into model retraining —
        improving accuracy with every batch of data.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FAILURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Failure Detection":

    st.markdown("""
    <div class='section-header' style='background:linear-gradient(135deg,#1a0533 0%,#2d1b69 50%,#1e3a5f 100%);'>
      <h1>🎯 Failure Detection</h1>
      <p>Episode-level anomaly detection — validated on real robot manipulation data</p>
    </div>
    """, unsafe_allow_html=True)

    results = load_benchmark_data()
    if not results:
        st.error("No benchmark data found. Run: `python main.py --source lerobot --dataset lerobot/xarm_lift_medium_replay`")
        st.stop()

    # dataset picker
    ds_names = [r["name"] for r in results]
    sel = st.selectbox("Select dataset", ds_names,
                       format_func=lambda x: x.replace("lerobot/", ""))
    data = next(r for r in results if r["name"] == sel)
    card, scores, labels, preds = data["card"], data["scores"], data["labels"], data["preds"]
    cm   = card["confusion_matrix"]
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total  = card["total_episodes"]
    n_fail = card["failure_episodes"]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ROC-AUC",            f"{card['roc_auc']:.3f}")
    k2.metric("Detection Rate",     f"{card['detection_rate_pct']} %",
              f"{tp} of {n_fail} failures caught")
    k3.metric("False Positive Rate",f"{card['false_positive_rate_pct']} %",
              f"{fp} false alarms", delta_color="inverse")
    k4.metric("Episodes Analyzed",  f"{total:,}")
    k5.metric("Failure Rate",       f"{n_fail/total*100:.1f} %",
              f"{n_fail} failures in dataset")

    # ── Quality badge ─────────────────────────────────────────────────────────
    auc = card['roc_auc']
    if auc >= 0.9:
        badge_html = f"<span class='perf-badge'>✓ Production-ready — ROC-AUC {auc:.3f}</span>"
    elif auc >= 0.75:
        badge_html = f"<span class='perf-badge-warn'>⚠ Good — ROC-AUC {auc:.3f} · consider more training data</span>"
    else:
        badge_html = f"<span class='perf-badge-warn'>⚠ Needs improvement — ROC-AUC {auc:.3f}</span>"
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Score distribution + confusion matrix ────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Anomaly Score Distribution")
        st.caption("Nominal episodes (blue) should score low; failures (red) should score high. "
                   "The dashed line is the detection threshold.")
        thresh = np.quantile(scores, card.get("threshold_quantile",
                             card.get("confidence_threshold_quantile", 0.75)))
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores[labels==0], name="Nominal",
            marker_color="#3b82f6", opacity=0.75,
            xbins=dict(size=0.01), hovertemplate="Score %{x:.3f}<br>Count %{y}<extra>Nominal</extra>"))
        fig.add_trace(go.Histogram(
            x=scores[labels==1], name="Failure",
            marker_color="#ef4444", opacity=0.75,
            xbins=dict(size=0.01), hovertemplate="Score %{x:.3f}<br>Count %{y}<extra>Failure</extra>"))
        fig.add_vline(x=thresh, line_dash="dash", line_color="#fbbf24", line_width=2,
                      annotation_text=f"Threshold {thresh:.3f}",
                      annotation_font_color="#fbbf24",
                      annotation_position="top right")
        fig.update_layout(barmode="overlay", xaxis_title="Anomaly Score", yaxis_title="Episode count")
        st.plotly_chart(dark_chart(fig, 340), use_container_width=True)

    with right:
        st.markdown("#### Confusion Matrix")
        st.caption("How predictions compare to ground truth labels derived from reward signal.")
        z  = [[tn, fp], [fn, tp]]
        xt = ["Predicted OK", "Predicted FAIL"]
        yt = ["True OK", "True FAIL"]
        tx = [[f"TN · {tn}\n(correct)", f"FP · {fp}\n(false alarm)"],
              [f"FN · {fn}\n(missed)", f"TP · {tp}\n(caught)"]]
        fig2 = go.Figure(go.Heatmap(
            z=z, x=xt, y=yt,
            text=tx, texttemplate="%{text}",
            colorscale=[[0,"#0d1a2e"],[0.5,"#1e3a5f"],[1,"#3b82f6"]],
            showscale=False,
            hoverongaps=False,
        ))
        fig2.update_layout(
            height=340, margin=dict(t=36,b=36,l=10,r=10),
            plot_bgcolor="#060d1a", paper_bgcolor="#0a0f1e",
            font_color="#cbd5e1",
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Episode timeline ──────────────────────────────────────────────────────
    st.markdown("#### All Episodes — Anomaly Scores")
    st.caption("Every episode plotted by score. Hover to inspect individual episodes.")

    nom_i = np.where((labels==0) & (preds==0))[0]
    tp_i  = np.where((labels==1) & (preds==1))[0]
    fn_i  = np.where((labels==1) & (preds==0))[0]
    fp_i  = np.where((labels==0) & (preds==1))[0]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=nom_i, y=scores[nom_i], mode="markers",
        marker=dict(color="#3b82f6", size=5, opacity=0.55),
        name="Nominal — correct",
        hovertemplate="Ep %{x}<br>Score %{y:.4f}<extra>Nominal</extra>"))
    fig3.add_trace(go.Scatter(
        x=tp_i, y=scores[tp_i], mode="markers",
        marker=dict(color="#ef4444", size=8),
        name=f"Failure — caught ({len(tp_i)})",
        hovertemplate="Ep %{x}<br>Score %{y:.4f}<extra>Caught failure</extra>"))
    fig3.add_trace(go.Scatter(
        x=fn_i, y=scores[fn_i], mode="markers",
        marker=dict(color="#f97316", size=10, symbol="x", line=dict(width=2,color="#f97316")),
        name=f"Failure — missed ({len(fn_i)})",
        hovertemplate="Ep %{x}<br>Score %{y:.4f}<extra>Missed failure</extra>"))
    fig3.add_trace(go.Scatter(
        x=fp_i, y=scores[fp_i], mode="markers",
        marker=dict(color="#a855f7", size=8, symbol="diamond"),
        name=f"False alarm ({len(fp_i)})",
        hovertemplate="Ep %{x}<br>Score %{y:.4f}<extra>False alarm</extra>"))
    fig3.add_hline(y=thresh, line_dash="dash", line_color="#fbbf24", line_width=1.5,
                   annotation_text="Threshold", annotation_font_color="#fbbf24")
    fig3.update_layout(xaxis_title="Episode index", yaxis_title="Anomaly score")
    st.plotly_chart(dark_chart(fig3, 420), use_container_width=True)

    # ── Cross-dataset comparison ──────────────────────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("#### Cross-Dataset Performance")
        rows = []
        for r in results:
            c2 = r["card"]
            rows.append({
                "Dataset":             r["name"].replace("lerobot/",""),
                "Episodes":            c2["total_episodes"],
                "Failure rate":        f"{c2['failure_episodes']/c2['total_episodes']*100:.1f}%",
                "ROC-AUC":             c2["roc_auc"],
                "Detection rate":      f"{c2['detection_rate_pct']}%",
                "False positive rate": f"{c2['false_positive_rate_pct']}%",
                "TP": c2["confusion_matrix"]["tp"],
                "FP": c2["confusion_matrix"]["fp"],
                "FN": c2["confusion_matrix"]["fn"],
                "TN": c2["confusion_matrix"]["tn"],
            })
        df_cross = pd.DataFrame(rows)

        bar_df = df_cross[["Dataset","ROC-AUC","Detection rate"]].copy()
        bar_df["Detection rate num"] = bar_df["Detection rate"].str.replace("%","").astype(float)/100

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=bar_df["Dataset"], y=bar_df["ROC-AUC"],
            name="ROC-AUC", marker_color="#3b82f6",
            text=bar_df["ROC-AUC"].apply(lambda v: f"{v:.3f}"),
            textposition="outside"))
        fig4.add_trace(go.Bar(
            x=bar_df["Dataset"], y=bar_df["Detection rate num"],
            name="Detection rate", marker_color="#10b981",
            text=bar_df["Detection rate"],
            textposition="outside"))
        fig4.update_layout(barmode="group", yaxis_range=[0, 1.12],
                           xaxis_title="Dataset", yaxis_title="Score (0–1)")
        st.plotly_chart(dark_chart(fig4, 360), use_container_width=True)

        st.dataframe(
            df_cross[["Dataset","Episodes","Failure rate","ROC-AUC","Detection rate","False positive rate"]],
            use_container_width=True, hide_index=True)

    # ── Misclassified table ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Episode Breakdown")
    t1, t2, t3 = st.tabs(["❌ Misclassified", "🔴 Failures", "📋 All Episodes"])

    def ep_df(mask):
        idx = np.where(mask)[0]
        return pd.DataFrame({
            "Episode":       idx,
            "Anomaly Score": scores[idx].round(4),
            "True Label":    ["FAILURE" if labels[i] else "OK" for i in idx],
            "Predicted":     ["FAILURE" if preds[i]  else "OK" for i in idx],
            "Result":        ["✓ correct" if preds[i]==labels[i] else "✗ wrong" for i in idx],
        })

    with t1:
        df = ep_df(preds != labels)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
        st.caption(f"{len(df)} misclassified out of {len(labels)} total episodes")
    with t2:
        st.dataframe(ep_df(labels==1), use_container_width=True, hide_index=True, height=300)
    with t3:
        st.dataframe(ep_df(np.ones(len(labels), bool)), use_container_width=True, hide_index=True, height=300)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STEP ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏷️ Step Annotations":

    st.markdown("""
    <div class='section-header' style='background:linear-gradient(135deg,#052e16 0%,#064e3b 40%,#065f46 100%);'>
      <h1>🏷️ Step-Level Annotations</h1>
      <p>Per-timestep failure types and semantic labels on real robot trajectories</p>
    </div>
    """, unsafe_allow_html=True)

    report, summary = load_pipeline_report()

    if report is None:
        st.warning("Pipeline demo output not found. Run: `python pipeline.py --demo`")
        st.stop()

    episodes = report  # list of episode dicts

    # ── Model performance strip ───────────────────────────────────────────────
    st.markdown("##### Model accuracy — trained on open source robot datasets")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Failure Typing",       "92.0 %",  "RF · xarm_lift + push")
    a2.metric("Task Phase",           "94.9 %",  "RF · DROID + ALOHA")
    a3.metric("Workspace Zone",       "100.0 %", "RF · DROID + ALOHA")
    a4.metric("Contact State",        "99.9 %",  "RF · DROID + ALOHA")
    a5.metric("Motion Type",          "76.2 %",  "RF · DROID + ALOHA")
    a6.metric("Training Steps",       "177 k",   "4 datasets · 4 robot types")

    st.markdown("---")

    # ── Episode selector ──────────────────────────────────────────────────────
    ep_options = []
    for ep in episodes:
        fa = ep["failure_annotation"]
        label_str = ep.get("label_str", "?")
        ep_options.append(
            f"{ep['episode_id']}  ·  {label_str}  ·  dominant: {fa['dominant']}  ·  peak score {fa['peak_score']:.3f}")

    sel_idx = st.selectbox("Select episode to inspect", range(len(episodes)),
                           format_func=lambda i: ep_options[i])
    ep = episodes[sel_idx]
    fa = ep["failure_annotation"]
    sa = ep["semantic_annotation"]
    n_steps = ep["n_steps"]
    coords  = np.array(ep["coords_3d"])            # (T, 3)
    step_scores = np.array(fa["step_scores"])      # (T,)
    step_labels = fa["step_labels"]                # list[str]
    step_confs  = fa["step_confs"]                 # list[float]

    # ── Episode summary strip ─────────────────────────────────────────────────
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Timesteps",      n_steps)
    e2.metric("Episode label",  ep.get("label_str", "—"))
    e3.metric("Dominant failure", fa["dominant"].replace("_"," ").title())
    e4.metric("Peak anomaly score", f"{fa['peak_score']:.3f}", f"at step {fa['peak_step']}")
    e5.metric("Pipeline prediction", "⚠ FLAGGED" if ep.get("flagged") else "✓ NORMAL",
              delta_color="inverse" if ep.get("flagged") else "normal")

    st.markdown("---")

    # ── 3D viewer + step-level breakdown ─────────────────────────────────────
    viewer_col, side_col = st.columns([3, 2])

    with viewer_col:
        st.markdown("#### 3D Trajectory")
        color_mode = st.radio(
            "Color by", ["Failure Type", "Anomaly Score", "Task Phase", "Motion Type"],
            horizontal=True)

        fig3d = go.Figure()

        if color_mode == "Failure Type":
            for ftype, color in FAIL_COLOR.items():
                idx = [i for i, f in enumerate(step_labels) if f == ftype]
                if not idx:
                    continue
                fig3d.add_trace(go.Scatter3d(
                    x=coords[idx,0], y=coords[idx,1], z=coords[idx,2],
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.85),
                    name=FAIL_LABEL[ftype],
                    hovertemplate=(
                        "Step %{customdata[0]}<br>"
                        "Score: %{customdata[1]:.3f}<br>"
                        "Conf: %{customdata[2]:.2f}<br>"
                        f"Type: {FAIL_LABEL[ftype]}"
                        "<extra></extra>"),
                    customdata=[[i, step_scores[i], step_confs[i]] for i in idx],
                ))

        elif color_mode == "Anomaly Score":
            fig3d.add_trace(go.Scatter3d(
                x=coords[:,0], y=coords[:,1], z=coords[:,2],
                mode="markers",
                marker=dict(size=5, color=step_scores,
                            colorscale="RdBu_r", showscale=True,
                            colorbar=dict(title="Anomaly<br>Score", thickness=12,
                                          tickfont=dict(color="white"), titlefont=dict(color="white"))),
                hovertemplate=("Step %{customdata[0]}<br>Score: %{customdata[1]:.3f}"
                               "<extra></extra>"),
                customdata=[[i, step_scores[i]] for i in range(n_steps)],
                name="Trajectory",
            ))

        elif color_mode == "Task Phase":
            phases = sa["task_phase"]["step_labels"]
            for phase in set(phases):
                idx = [i for i, p in enumerate(phases) if p == phase]
                color = SEM_COLOR.get(phase, "#888888")
                fig3d.add_trace(go.Scatter3d(
                    x=coords[idx,0], y=coords[idx,1], z=coords[idx,2],
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.85),
                    name=phase.replace("_"," ").title(),
                    hovertemplate=f"Step %{{customdata}}<br>Phase: {phase}<extra></extra>",
                    customdata=idx,
                ))

        elif color_mode == "Motion Type":
            mtypes = sa["motion_type"]["step_labels"]
            for mt in set(mtypes):
                idx = [i for i, m in enumerate(mtypes) if m == mt]
                color = SEM_COLOR.get(mt, "#888888")
                fig3d.add_trace(go.Scatter3d(
                    x=coords[idx,0], y=coords[idx,1], z=coords[idx,2],
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.85),
                    name=mt.replace("_"," ").title(),
                    hovertemplate=f"Step %{{customdata}}<br>Motion: {mt}<extra></extra>",
                    customdata=idx,
                ))

        # trajectory line
        fig3d.add_trace(go.Scatter3d(
            x=coords[:,0], y=coords[:,1], z=coords[:,2],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.12)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
        # peak anomaly marker
        peak = fa["peak_step"]
        if peak < len(coords):
            fig3d.add_trace(go.Scatter3d(
                x=[coords[peak,0]], y=[coords[peak,1]], z=[coords[peak,2]],
                mode="markers+text",
                marker=dict(size=11, color="#fbbf24", symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"Peak (step {peak})"],
                textfont=dict(color="#fbbf24", size=11),
                textposition="top center",
                name="Peak anomaly",
            ))

        fig3d.update_layout(
            height=500,
            scene=dict(
                bgcolor="#060d1a",
                xaxis=dict(title="PC1", backgroundcolor="#060d1a",
                           gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                           tickfont=dict(color="#475569"), titlefont=dict(color="#64b5f6")),
                yaxis=dict(title="PC2", backgroundcolor="#060d1a",
                           gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                           tickfont=dict(color="#475569"), titlefont=dict(color="#64b5f6")),
                zaxis=dict(title="PC3", backgroundcolor="#060d1a",
                           gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                           tickfont=dict(color="#475569"), titlefont=dict(color="#64b5f6")),
            ),
            paper_bgcolor="#0a0f1e", font_color="#cbd5e1",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#1e2d45",
                        font=dict(size=11)),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with side_col:
        st.markdown("#### Semantic Labels")
        st.caption("Dominant label per layer across all timesteps")

        def sem_gauge(layer_name, layer_key, counts_dict):
            dominant = sa[layer_key]["dominant"]
            counts   = sa[layer_key]["counts"]
            total    = sum(counts.values())
            items = sorted(counts.items(), key=lambda x: -x[1])
            bars = go.Figure()
            for label, cnt in items:
                if cnt == 0:
                    continue
                pct = cnt / total * 100
                color = SEM_COLOR.get(label, "#6b7280")
                bars.add_trace(go.Bar(
                    x=[pct], y=[label.replace("_"," ").title()],
                    orientation="h",
                    marker_color=color,
                    text=[f"{cnt} steps ({pct:.0f}%)"],
                    textposition="auto",
                    name=label,
                    showlegend=False,
                    hovertemplate=f"{label}: {cnt} steps ({pct:.1f}%)<extra></extra>",
                ))
            bars.update_layout(
                height=max(100, len([c for c in counts.values() if c > 0]) * 36 + 40),
                margin=dict(t=0, b=10, l=10, r=10),
                plot_bgcolor="#060d1a", paper_bgcolor="#0a0f1e",
                font_color="#cbd5e1", font_size=11,
                xaxis=dict(gridcolor="#1e2d45", range=[0, 110],
                           title="", ticksuffix="%"),
                yaxis=dict(gridcolor="#1e2d45"),
            )
            st.markdown(f"**{layer_name}** — dominant: `{dominant.replace('_',' ')}`")
            st.plotly_chart(bars, use_container_width=True)

        sem_gauge("Task Phase",       "task_phase",     sa["task_phase"]["counts"])
        sem_gauge("Contact State",    "contact_state",  sa["contact_state"]["counts"])
        sem_gauge("Motion Type",      "motion_type",    sa["motion_type"]["counts"])
        sem_gauge("Workspace Zone",   "workspace_zone", sa["workspace_zone"]["counts"])

    # ── Step-level timeline ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step-by-Step Timeline")
    st.caption("Each point is one timestep. Hover to see the label, score, and confidence.")

    timeline_mode = st.radio("View", ["Anomaly Score", "Failure Type", "Task Phase"],
                             horizontal=True, key="timeline_mode")

    fig_t = go.Figure()

    if timeline_mode == "Anomaly Score":
        fig_t.add_trace(go.Scatter(
            x=list(range(n_steps)), y=step_scores,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=1.5),
            marker=dict(size=4, color=step_scores, colorscale="RdBu_r"),
            name="Anomaly score",
            hovertemplate="Step %{x}<br>Score %{y:.4f}<extra></extra>",
        ))
        # confidence band
        fig_t.add_trace(go.Scatter(
            x=list(range(n_steps)), y=step_confs,
            mode="lines",
            line=dict(color="rgba(16,185,129,0.4)", width=1.5, dash="dot"),
            name="Model confidence",
            hovertemplate="Step %{x}<br>Conf %{y:.2f}<extra></extra>",
        ))
        fig_t.update_layout(yaxis_title="Score / Confidence")

    elif timeline_mode == "Failure Type":
        for ftype, color in FAIL_COLOR.items():
            idx = [i for i, f in enumerate(step_labels) if f == ftype]
            if not idx:
                continue
            fig_t.add_trace(go.Scatter(
                x=idx, y=[step_scores[i] for i in idx],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.85),
                name=FAIL_LABEL[ftype],
                hovertemplate=(f"Step %{{x}}<br>Score %{{y:.4f}}<br>{FAIL_LABEL[ftype]}"
                               "<extra></extra>"),
            ))
        fig_t.add_trace(go.Scatter(
            x=list(range(n_steps)), y=step_scores,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            showlegend=False,
        ))
        fig_t.update_layout(yaxis_title="Anomaly score")

    elif timeline_mode == "Task Phase":
        phases = sa["task_phase"]["step_labels"]
        for phase in set(phases):
            idx = [i for i, p in enumerate(phases) if p == phase]
            color = SEM_COLOR.get(phase, "#888888")
            fig_t.add_trace(go.Scatter(
                x=idx, y=[step_scores[i] for i in idx],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.85),
                name=phase.replace("_"," ").title(),
                hovertemplate=f"Step %{{x}}<br>Score %{{y:.4f}}<br>Phase: {phase}<extra></extra>",
            ))
        fig_t.add_trace(go.Scatter(
            x=list(range(n_steps)), y=step_scores,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            showlegend=False,
        ))
        fig_t.update_layout(yaxis_title="Anomaly score")

    fig_t.update_layout(xaxis_title="Timestep")
    st.plotly_chart(dark_chart(fig_t, 320), use_container_width=True)

    # ── All episodes summary table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### All 100 Episodes — Annotation Summary")
    st.caption("Full pipeline output. Flagged = anomaly score above threshold.")

    rows = []
    for ep2 in episodes:
        fa2 = ep2["failure_annotation"]
        sa2 = ep2["semantic_annotation"]
        rows.append({
            "Episode":       ep2["episode_id"],
            "True Label":    ep2.get("label_str", "—"),
            "Flagged":       "⚠ YES" if ep2.get("flagged") else "✓ NO",
            "Anomaly Score": round(ep2["anomaly_score"], 4),
            "Dominant Failure": fa2["dominant"].replace("_"," ").title(),
            "Peak Step":     fa2["peak_step"],
            "Task Phase":    sa2["task_phase"]["dominant"].replace("_"," ").title(),
            "Contact State": sa2["contact_state"]["dominant"].replace("_"," ").title(),
            "Motion Type":   sa2["motion_type"]["dominant"].replace("_"," ").title(),
            "Correct":       "✓" if ep2.get("correct") else ("✗" if ep2.get("true_label") is not None else "—"),
        })

    df_all = pd.DataFrame(rows)
    st.dataframe(df_all, use_container_width=True, hide_index=True, height=380)

    # ── Aggregate failure breakdown ────────────────────────────────────────────
    st.markdown("---")
    col_pie, col_sem = st.columns(2)

    with col_pie:
        st.markdown("#### Failure Type Distribution (all episodes)")
        totals = summary.get("failure_type_totals", {})
        labels_f = [FAIL_LABEL.get(k, k) for k, v in totals.items() if v > 0]
        vals_f   = [v for v in totals.values() if v > 0]
        colors_f = [FAIL_COLOR.get(k, "#888") for k, v in totals.items() if v > 0]
        if vals_f:
            fig_pie = go.Figure(go.Pie(
                labels=labels_f, values=vals_f,
                hole=0.4,
                marker_colors=colors_f,
                textinfo="label+percent",
                textfont=dict(size=11, color="white"),
                hovertemplate="%{label}<br>%{value} steps (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                height=340, margin=dict(t=20,b=20,l=10,r=10),
                paper_bgcolor="#0a0f1e", font_color="#cbd5e1",
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_sem:
        st.markdown("#### Task Phase Distribution (all episodes)")
        all_phases = {}
        for ep2 in episodes:
            for phase, cnt in ep2["semantic_annotation"]["task_phase"]["counts"].items():
                all_phases[phase] = all_phases.get(phase, 0) + cnt
        phase_items = [(k, v) for k, v in sorted(all_phases.items(), key=lambda x: -x[1]) if v > 0]
        if phase_items:
            fig_ph = go.Figure(go.Bar(
                x=[v for k, v in phase_items],
                y=[k.replace("_"," ").title() for k, v in phase_items],
                orientation="h",
                marker_color=[SEM_COLOR.get(k, "#888") for k, v in phase_items],
                text=[f"{v:,}" for k, v in phase_items],
                textposition="outside",
                hovertemplate="%{y}: %{x} steps<extra></extra>",
            ))
            fig_ph.update_layout(
                xaxis_title="Total steps", showlegend=False,
                height=340,
            )
            st.plotly_chart(dark_chart(fig_ph, 340, legend_h=False), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REVIEW QUEUE (human-in-the-loop + active learning)
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Review Queue":

    st.markdown("""
    <div class='section-header' style='background:linear-gradient(135deg,#1c0a2e 0%,#2d1b4e 50%,#1a0a2e 100%);
         border:1px solid #7c3aed;'>
      <h1>🔍 Human-in-the-Loop Review</h1>
      <p>Active learning surfaces the <b>most informative</b> steps first — label fewer, teach the model more</p>
    </div>
    """, unsafe_allow_html=True)

    report, summary = load_pipeline_report()
    if report is None:
        st.warning("Run `python pipeline.py --demo` first to generate the review queue.")
        st.stop()

    review_queue = load_review_queue()
    corrections  = load_corrections()
    corrected_ids = {c["episode_id"] + "_" + str(c["step"]) for c in corrections}

    # ── KPI bar ───────────────────────────────────────────────────────────────
    try:
        ann_path = OUTPUT_DIR / "robot_annotator.pkl"
        with open(ann_path, "rb") as f:
            state = pickle.load(f)
        cal = state.get("calibration_report", {})
    except Exception:
        cal = {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence gate",   "60 %",  "steps below → human review")
    c2.metric("Episodes queued",   len(review_queue))
    c3.metric("Uncertain steps",
              summary.get("quality", {}).get("review_steps", "—"),
              f"{summary.get('quality', {}).get('review_rate_pct', '—')}% of total steps")
    c4.metric("Corrections saved", len(corrections),
              f"{sum(1 for c in corrections if c.get('changed'))} label changes")
    if cal:
        c5.metric("Brier score", cal.get("brier_score_calibrated", "—"),
                  f"{cal.get('improvement_pct', 0):+.1f}% vs uncalibrated")
    else:
        c5.metric("Model accuracy", "93.7 %", "Platt-scaled RF")

    # ── Pipeline explanation ──────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#1a0a2e;border:1px solid #7c3aed;border-radius:10px;
                padding:1rem 1.4rem;margin:1rem 0;'>
      <div style='display:flex;gap:2.5rem;flex-wrap:wrap;'>
        <div style='flex:1;min-width:180px;'>
          <div style='color:#a78bfa;font-weight:700;font-size:.85rem;margin-bottom:.4rem;
                      letter-spacing:.05em;text-transform:uppercase;'>① Model annotates</div>
          <div style='color:#94a3b8;font-size:.85rem;line-height:1.6;'>
            Calibrated RF labels every step. Steps below 60% confidence are flagged automatically.
          </div>
        </div>
        <div style='flex:1;min-width:180px;'>
          <div style='color:#a78bfa;font-weight:700;font-size:.85rem;margin-bottom:.4rem;
                      letter-spacing:.05em;text-transform:uppercase;'>② Active learning ranks</div>
          <div style='color:#94a3b8;font-size:.85rem;line-height:1.6;'>
            Entropy + diversity selection surfaces steps with the highest information gain first.
          </div>
        </div>
        <div style='flex:1;min-width:180px;'>
          <div style='color:#a78bfa;font-weight:700;font-size:.85rem;margin-bottom:.4rem;
                      letter-spacing:.05em;text-transform:uppercase;'>③ You correct labels</div>
          <div style='color:#94a3b8;font-size:.85rem;line-height:1.6;'>
            Use the dropdowns below. Each correction is saved with its feature vector.
          </div>
        </div>
        <div style='flex:1;min-width:180px;'>
          <div style='color:#a78bfa;font-weight:700;font-size:.85rem;margin-bottom:.4rem;
                      letter-spacing:.05em;text-transform:uppercase;'>④ Model retrains 10×</div>
          <div style='color:#94a3b8;font-size:.85rem;line-height:1.6;'>
            Human labels are weighted <b style='color:#c4b5fd;'>10× higher</b> than weak supervision — model learns fast.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if not review_queue:
        st.success("✓ Review queue is empty — model is confident on all steps!")
        st.stop()

    # ── Quality + review rate overview ────────────────────────────────────────
    quality_scores_all = [r["quality_score"] for r in report]
    q_arr = np.array(quality_scores_all)

    ql, qr = st.columns([2, 1])
    with ql:
        st.markdown("#### Episode Quality Distribution")
        st.caption("Episodes ≥ 0.65 exported as training-ready.")
        fig_q = go.Figure()
        fig_q.add_trace(go.Histogram(
            x=q_arr, xbins=dict(size=0.04),
            marker_color="#7c3aed", opacity=0.75,
            hovertemplate="Quality %{x:.2f}<br>Count %{y}<extra></extra>",
        ))
        fig_q.add_vline(x=0.65, line_dash="dash", line_color="#10b981", line_width=2,
                        annotation_text="Training threshold",
                        annotation_font_color="#10b981")
        fig_q.update_layout(xaxis_title="Quality score", yaxis_title="Episodes")
        st.plotly_chart(dark_chart(fig_q, 260), use_container_width=True)

    with qr:
        st.markdown("#### Quality Breakdown")
        q_summary = summary.get("quality", {})
        st.markdown(f"""
        <div class='step-card' style='margin-top:.5rem;'>
          <div style='color:#94a3b8;font-size:.87rem;line-height:2;'>
            <div>Mean quality score: <b style='color:#e2e8f0;'>{q_summary.get('mean_quality','—')}</b></div>
            <div>Training-ready: <b style='color:#10b981;'>{q_summary.get('episodes_above_threshold','—')}
              ({q_summary.get('pct_training_ready','—')}%)</b></div>
            <div>Review steps: <b style='color:#fbbf24;'>{q_summary.get('review_steps','—')}
              ({q_summary.get('review_rate_pct','—')}%)</b></div>
            <div style='margin-top:.8rem;font-size:.78rem;color:#64748b;'>
              Quality = 50% anomaly clean +<br>35% nominal step fraction +<br>15% model confidence
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Episode selector ──────────────────────────────────────────────────────
    # sort by active-learning episode score: most to gain from labeling first
    def _ep_al_score(ep):
        rr  = ep.get("review_rate", 0.0)
        qs  = ep.get("quality_score", 0.5)
        ans = ep.get("anomaly_score", 0.0)
        return 0.5 * rr + 0.3 * (1 - qs) + 0.2 * ans

    rq_sorted = sorted(review_queue, key=_ep_al_score, reverse=True)

    ep_labels_rq = [
        f"[#{i+1} priority]  {ep['episode_id']}  ·  {ep['n_needs_review']} uncertain steps  ·  "
        f"quality {ep['quality_score']:.2f}"
        for i, ep in enumerate(rq_sorted)
    ]

    st.markdown("#### Select Episode to Review")
    st.caption("Episodes are sorted by information gain — label the top ones for maximum model improvement.")
    sel_rq = st.selectbox("Episode", range(len(rq_sorted)),
                           format_func=lambda i: ep_labels_rq[i],
                           label_visibility="collapsed")
    ep_rq = rq_sorted[sel_rq]

    re1, re2, re3, re4 = st.columns(4)
    re1.metric("Episode",          ep_rq["episode_id"])
    re2.metric("Uncertain steps",  ep_rq["n_needs_review"],
               f"{ep_rq['review_rate']*100:.0f}% of episode")
    re3.metric("Quality score",    f"{ep_rq['quality_score']:.2f}")
    re4.metric("Dominant failure", ep_rq["dominant"].replace("_"," ").title())

    # ── Active learning ranking for this episode ──────────────────────────────
    al_ranked    = ep_rq.get("al_ranked", [])
    low_conf_steps = ep_rq["low_conf_steps"]
    all_labels   = ep_rq["step_labels"]
    all_confs    = ep_rq["step_confs"]

    # build a lookup: step → AL rank info
    al_lookup = {r["step"]: r for r in al_ranked}

    FAILURE_OPTIONS = ALL_FAILURE_CLASSES
    STRATEGY_COLOR  = {"uncertainty": "#f59e0b", "diversity": "#8b5cf6"}

    st.markdown("<br>", unsafe_allow_html=True)

    if al_ranked:
        # Show AL informativeness bar chart
        al_steps  = [r["step"]            for r in al_ranked]
        al_scores = [r["informativeness"] for r in al_ranked]
        al_strats = [r["strategy"]        for r in al_ranked]
        al_colors = [STRATEGY_COLOR.get(s, "#64748b") for s in al_strats]

        fig_al = go.Figure(go.Bar(
            x=[f"Step {s}" for s in al_steps],
            y=al_scores,
            marker_color=al_colors,
            hovertemplate="Step %{x}<br>Informativeness: %{y:.3f}<br><extra></extra>",
        ))
        # legend traces
        for strat, color in STRATEGY_COLOR.items():
            fig_al.add_trace(go.Bar(
                x=[], y=[], marker_color=color,
                name=strat.capitalize(), showlegend=True,
            ))
        fig_al.update_layout(
            title=dict(text="Active Learning Priority — Steps Ranked by Information Gain",
                       font_color="#c4b5fd"),
            xaxis_title="Step index", yaxis_title="Informativeness score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            barmode="overlay",
        )
        st.plotly_chart(dark_chart(fig_al, 230), use_container_width=True)
        st.caption("🟡 Uncertainty = model near decision boundary  ·  🟣 Diversity = covers undersampled failure region")
    else:
        st.info("Active learning ranking not available for this episode — re-run pipeline to enable.")

    st.markdown("---")

    # ── ✏️  LABEL CORRECTION INTERFACE ────────────────────────────────────────
    st.markdown(f"""
    <div style='background:#0f172a;border:2px solid #7c3aed;border-radius:10px;
                padding:1rem 1.4rem;margin-bottom:1rem;'>
      <div style='color:#a78bfa;font-weight:700;font-size:1rem;margin-bottom:.3rem;'>
        ✏️  Correct Model Predictions
      </div>
      <div style='color:#94a3b8;font-size:.87rem;'>
        <b style='color:#cbd5e1;'>{len(low_conf_steps)} uncertain steps</b> in
        <code style='color:#60a5fa;'>{ep_rq['episode_id']}</code> are shown below —
        sorted by active learning priority (highest info gain first).<br>
        The model's best guess is pre-selected. <b style='color:#c4b5fd;'>Change the dropdown if it's wrong.</b>
        Your corrections are weighted <b style='color:#fbbf24;'>10×</b> in the next retrain.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sort steps by AL priority if available, else by step index
    if al_ranked:
        ordered_steps = [r["step"] for r in al_ranked
                         if r["step"] in low_conf_steps]
        # append any remaining not in al_ranked
        ordered_steps += [s for s in low_conf_steps if s not in ordered_steps]
    else:
        ordered_steps = low_conf_steps

    new_corrections = []
    batch_size = 10
    for batch_start in range(0, len(ordered_steps), batch_size):
        batch = ordered_steps[batch_start:batch_start + batch_size]
        cols  = st.columns(min(len(batch), 5))
        for col_idx, step_i in enumerate(batch):
            col       = cols[col_idx % 5]
            cur_label = all_labels[step_i] if step_i < len(all_labels) else "nominal"
            cur_conf  = all_confs[step_i]  if step_i < len(all_confs)  else 0.0
            key_id    = f"{ep_rq['episode_id']}_step{step_i}"
            already   = key_id in corrected_ids
            al_info   = al_lookup.get(step_i, {})
            priority  = al_info.get("priority", "—")
            strategy  = al_info.get("strategy", "")
            info_score= al_info.get("informativeness", 0.0)
            strat_col = STRATEGY_COLOR.get(strategy, "#64748b")

            with col:
                conf_color = "#ef4444" if cur_conf < 0.4 else "#f97316"
                badge = (f"<span style='background:{strat_col};color:#fff;border-radius:4px;"
                         f"padding:1px 5px;font-size:.68rem;margin-left:3px;'>"
                         f"#{priority}</span>") if priority != "—" else ""
                st.markdown(
                    f"<div style='font-size:.75rem;color:{conf_color};font-weight:700;'>"
                    f"Step {step_i}{badge}"
                    f"{'  ✓' if already else ''}</div>"
                    f"<div style='font-size:.7rem;color:#64748b;'>conf {cur_conf:.2f}"
                    f"{f' · info {info_score:.2f}' if info_score else ''}</div>",
                    unsafe_allow_html=True)
                corrected = st.selectbox(
                    "Label",
                    FAILURE_OPTIONS,
                    index=FAILURE_OPTIONS.index(cur_label) if cur_label in FAILURE_OPTIONS else 0,
                    key=f"rq_{key_id}",
                    label_visibility="collapsed",
                )
                new_corrections.append({
                    "episode_id":      ep_rq["episode_id"],
                    "step":            step_i,
                    "original_label":  cur_label,
                    "corrected_label": corrected,
                    "original_conf":   cur_conf,
                    "informativeness": info_score,
                    "al_strategy":     strategy,
                    "changed":         corrected != cur_label,
                })

    st.markdown("<br>", unsafe_allow_html=True)
    save_col, retrain_col, _ = st.columns([2, 2, 4])

    with save_col:
        if st.button("💾 Save corrections", type="primary", use_container_width=True):
            existing = {f"{c['episode_id']}_{c['step']}": c for c in corrections}
            for nc in new_corrections:
                k = f"{nc['episode_id']}_{nc['step']}"
                existing[k] = {**nc, "saved_at": datetime.now().isoformat()}
            merged = list(existing.values())
            save_corrections(merged)
            changed = [c for c in new_corrections if c["changed"]]
            st.success(f"✓ Saved {len(merged)} corrections ({len(changed)} label changes). "
                       f"These are weighted 10× in next retrain.")
            st.cache_data.clear()

    with retrain_col:
        if st.button("🔄 Retrain with human labels (10×)", use_container_width=True):
            corr_path = OUTPUT_DIR / "corrections.json"
            if not corr_path.exists() or len(corrections) == 0:
                st.warning("Save corrections first.")
            else:
                with st.spinner("Retraining — human labels weighted 10× over weak supervision..."):
                    result = subprocess.run(
                        [sys.executable, "annotation_model.py", "--train"],
                        capture_output=True, text=True, cwd=str(Path(__file__).parent)
                    )
                if result.returncode == 0:
                    acc_line = [l for l in result.stdout.splitlines() if "Overall accuracy" in l]
                    acc_str  = acc_line[-1].strip() if acc_line else ""
                    human_line = [l for l in result.stdout.splitlines() if "human-corrected" in l.lower()]
                    human_str  = human_line[0].strip() if human_line else ""
                    st.success(f"✓ Model retrained.  {acc_str}")
                    if human_str:
                        st.info(f"★ {human_str}")
                    st.cache_data.clear()
                else:
                    st.error(f"Retraining failed:\n{result.stderr[-500:]}")

    # ── Saved corrections log ─────────────────────────────────────────────────
    if corrections:
        st.markdown("---")
        st.markdown("#### All Saved Corrections")
        df_corr = pd.DataFrame([{
            "Episode":          c["episode_id"],
            "Step":             c["step"],
            "Model predicted":  c["original_label"],
            "Conf":             f"{c['original_conf']:.2f}",
            "Corrected to":     c["corrected_label"],
            "AL strategy":      c.get("al_strategy", "—"),
            "Changed":          "✓ yes" if c.get("changed") else "— same",
            "Saved at":         c.get("saved_at", "—")[:19],
        } for c in corrections])
        changed_count = df_corr["Changed"].str.contains("yes").sum()
        st.caption(f"{len(corrections)} corrections · {changed_count} label changes · "
                   f"All injected at 10× weight in next retraining cycle.")
        st.dataframe(df_corr, use_container_width=True, hide_index=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYTICS (UMAP + quality + failure clusters)
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":

    st.markdown("""
    <div class='section-header' style='background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
         border:1px solid #4c1d95;'>
      <h1>📊 Failure Analytics</h1>
      <p>Failure clustering, quality trends, and root cause patterns across all episodes</p>
    </div>
    """, unsafe_allow_html=True)

    report, summary = load_pipeline_report()
    if report is None:
        st.warning("Run `python pipeline.py --demo` first.")
        st.stop()

    episodes = report

    # ── Quality vs anomaly scatter ────────────────────────────────────────────
    qa1, qa2, qa3, qa4 = st.columns(4)
    quality_arr  = np.array([r["quality_score"]  for r in episodes])
    anomaly_arr  = np.array([r["anomaly_score"]   for r in episodes])
    flagged_arr  = np.array([r.get("flagged", False) for r in episodes])
    qa1.metric("Mean quality",       f"{quality_arr.mean():.3f}")
    qa2.metric("Episodes flagged",   int(flagged_arr.sum()),
               f"{flagged_arr.mean()*100:.0f}% of batch")
    qa3.metric("High-quality (≥0.65)", int((quality_arr >= 0.65).sum()))
    qa4.metric("Low-quality (<0.40)", int((quality_arr < 0.40).sum()))

    st.markdown("---")

    # ── UMAP failure cluster visualization ───────────────────────────────────
    st.markdown("#### Failure Clustering — Episode Embedding Space")
    st.caption(
        "Each point is an episode. Position is determined by anomaly score + failure type "
        "composition. Clusters reveal distinct failure modes the model has separated.")

    try:
        from umap import UMAP as UMAPReducer
        umap_available = True
    except ImportError:
        umap_available = False

    # Build episode feature vectors: anomaly_score + failure type fractions
    FTYPES = ["nominal", "velocity_spike", "position_jerk", "stuck_joint",
              "gripper_event", "high_anomaly", "self_collision", "overshoot",
              "trajectory_deviation", "perception_failure"]
    rows = []
    for r in episodes:
        fa     = r["failure_annotation"]
        n_s    = max(r["n_steps"], 1)
        counts = fa.get("counts", {})
        # failure type fractions + anomaly score + quality + review rate
        row = [r["anomaly_score"], r["quality_score"],
               fa.get("review_rate", 0.0), fa["peak_score"]]
        for ft in FTYPES:
            row.append(counts.get(ft, 0) / n_s)
        rows.append(row)
    X_ep = np.array(rows, dtype=np.float32)

    dominant_labels = [r["failure_annotation"]["dominant"] for r in episodes]
    label_strs      = [r.get("label_str", "UNKNOWN") for r in episodes]
    quality_ep      = [r["quality_score"] for r in episodes]
    flagged_ep      = [r.get("flagged", False) for r in episodes]

    if umap_available and len(X_ep) >= 10:
        with st.spinner("Computing UMAP embedding..."):
            reducer = UMAPReducer(n_components=2, random_state=42,
                                  n_neighbors=min(15, len(X_ep)-1), min_dist=0.1)
            emb = reducer.fit_transform(X_ep)
        x_emb, y_emb = emb[:, 0], emb[:, 1]
        dim_label = ("UMAP dim 1", "UMAP dim 2")
    else:
        # Fallback: PCA on the feature matrix
        from sklearn.decomposition import PCA
        pca   = PCA(n_components=2)
        emb   = pca.fit_transform(X_ep)
        x_emb, y_emb = emb[:, 0], emb[:, 1]
        var   = pca.explained_variance_ratio_
        dim_label = (f"PC1 ({var[0]*100:.0f}% var)", f"PC2 ({var[1]*100:.0f}% var)")
        if not umap_available:
            st.info("Install `umap-learn` for richer embeddings: `pip install umap-learn`")

    cluster_col, cluster_side = st.columns([3, 1])

    with cluster_col:
        cluster_mode = st.radio("Color by", ["Dominant Failure", "True Label",
                                              "Quality Score", "Flagged"],
                                horizontal=True, key="cluster_mode")
        fig_umap = go.Figure()

        if cluster_mode == "Dominant Failure":
            for ftype in FTYPES:
                idx = [i for i, d in enumerate(dominant_labels) if d == ftype]
                if not idx:
                    continue
                color = FAIL_COLOR.get(ftype, "#888")
                fig_umap.add_trace(go.Scatter(
                    x=x_emb[idx], y=y_emb[idx], mode="markers",
                    marker=dict(size=9, color=color, opacity=0.8,
                                line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
                    name=FAIL_LABEL.get(ftype, ftype),
                    hovertemplate=(
                        f"Episode %{{customdata[0]}}<br>"
                        f"Quality: %{{customdata[1]:.3f}}<br>"
                        f"Anomaly: %{{customdata[2]:.3f}}<br>"
                        f"Dominant: {ftype}<extra></extra>"
                    ),
                    customdata=[[episodes[i]["episode_id"],
                                 quality_ep[i], anomaly_arr[i]] for i in idx],
                ))

        elif cluster_mode == "True Label":
            for lbl, color in [("OK", "#3b82f6"), ("FAILURE", "#ef4444"), ("UNKNOWN", "#6b7280")]:
                idx = [i for i, l in enumerate(label_strs) if l == lbl]
                if not idx:
                    continue
                fig_umap.add_trace(go.Scatter(
                    x=x_emb[idx], y=y_emb[idx], mode="markers",
                    marker=dict(size=9, color=color, opacity=0.8),
                    name=lbl,
                    hovertemplate=f"Episode %{{customdata}}<br>Label: {lbl}<extra></extra>",
                    customdata=[episodes[i]["episode_id"] for i in idx],
                ))

        elif cluster_mode == "Quality Score":
            fig_umap.add_trace(go.Scatter(
                x=x_emb, y=y_emb, mode="markers",
                marker=dict(size=9, color=quality_ep, colorscale="RdYlGn",
                            showscale=True, cmin=0, cmax=1,
                            colorbar=dict(title="Quality", thickness=12,
                                          tickfont=dict(color="white"),
                                          titlefont=dict(color="white")),
                            opacity=0.85),
                hovertemplate=(
                    "Episode %{customdata[0]}<br>"
                    "Quality: %{customdata[1]:.3f}<br>"
                    "Anomaly: %{customdata[2]:.3f}<extra></extra>"
                ),
                customdata=[[episodes[i]["episode_id"], quality_ep[i],
                             float(anomaly_arr[i])] for i in range(len(episodes))],
                name="Quality",
            ))

        elif cluster_mode == "Flagged":
            for flagged, color, name in [(True, "#ef4444", "Flagged anomalous"),
                                          (False, "#3b82f6", "Normal")]:
                idx = [i for i, fl in enumerate(flagged_ep) if fl == flagged]
                if not idx:
                    continue
                sym = "diamond" if flagged else "circle"
                fig_umap.add_trace(go.Scatter(
                    x=x_emb[idx], y=y_emb[idx], mode="markers",
                    marker=dict(size=9 if flagged else 7, color=color,
                                opacity=0.85, symbol=sym),
                    name=name,
                    hovertemplate=f"Episode %{{customdata}}<br>{name}<extra></extra>",
                    customdata=[episodes[i]["episode_id"] for i in idx],
                ))

        fig_umap.update_layout(
            xaxis_title=dim_label[0], yaxis_title=dim_label[1],
        )
        st.plotly_chart(dark_chart(fig_umap, 480), use_container_width=True)

    with cluster_side:
        st.markdown("**Failure cluster sizes**")
        from collections import Counter
        dom_counts = Counter(dominant_labels)
        for ftype in FTYPES:
            cnt = dom_counts.get(ftype, 0)
            if cnt == 0:
                continue
            pct = cnt / len(episodes) * 100
            color = FAIL_COLOR.get(ftype, "#888")
            st.markdown(
                f"<div style='margin:.3rem 0;'>"
                f"<span style='color:{color};font-size:1.1rem;'>■</span> "
                f"<span style='color:#e2e8f0;font-size:.88rem;'>"
                f"<b>{ftype.replace('_',' ').title()}</b></span><br>"
                f"<span style='color:#64748b;font-size:.82rem;margin-left:1.4rem;'>"
                f"{cnt} episodes ({pct:.0f}%)</span></div>",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Interpretation**")
        st.markdown("""
        <div style='color:#94a3b8;font-size:.82rem;line-height:1.6;'>
          Episodes that cluster together share similar anomaly patterns.
          Tight clusters = consistent failure mode
          (e.g. always failing at approach).
          Scattered points = diverse failure causes.
        </div>
        """, unsafe_allow_html=True)

    # ── Anomaly score vs quality scatter ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Anomaly Score vs Quality Score")
    st.caption("High-quality training data lives in the bottom-right: low anomaly, high quality. "
               "Top-left = anomalous failures to fix or exclude.")

    fig_aq = go.Figure()
    for ftype in FTYPES:
        idx = [i for i, d in enumerate(dominant_labels) if d == ftype]
        if not idx:
            continue
        color = FAIL_COLOR.get(ftype, "#888")
        fig_aq.add_trace(go.Scatter(
            x=[float(anomaly_arr[i]) for i in idx],
            y=[quality_ep[i] for i in idx],
            mode="markers",
            marker=dict(size=8, color=color, opacity=0.75,
                        symbol=["diamond" if flagged_ep[i] else "circle" for i in idx]),
            name=FAIL_LABEL.get(ftype, ftype),
            hovertemplate=(
                "Episode %{customdata[0]}<br>"
                "Anomaly: %{x:.3f}<br>"
                "Quality: %{y:.3f}<extra></extra>"
            ),
            customdata=[[episodes[i]["episode_id"]] for i in idx],
        ))
    # training threshold line
    fig_aq.add_hline(y=0.65, line_dash="dash", line_color="#10b981", line_width=1.5,
                     annotation_text="Training threshold", annotation_font_color="#10b981")
    fig_aq.update_layout(xaxis_title="Anomaly score", yaxis_title="Quality score",
                          xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]))
    st.plotly_chart(dark_chart(fig_aq, 420), use_container_width=True)

    # ── Failure type × task phase heatmap ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Failure Type × Task Phase Co-occurrence")
    st.caption("Where in the task does each failure type tend to occur? "
               "Dark cells = rare combination, bright = common co-occurrence.")

    PHASES = ["approaching", "grasping", "transporting", "placing", "returning", "idle"]
    heat = np.zeros((len(FTYPES), len(PHASES)), dtype=int)
    for ep_r in episodes:
        dom_f  = ep_r["failure_annotation"]["dominant"]
        dom_p  = ep_r["semantic_annotation"].get("task_phase", {}).get("dominant", "idle")
        fi = FTYPES.index(dom_f)  if dom_f  in FTYPES  else 0
        pi = PHASES.index(dom_p)  if dom_p  in PHASES  else -1
        if pi >= 0:
            heat[fi, pi] += 1

    fig_heat = go.Figure(go.Heatmap(
        z=heat,
        x=[p.replace("_"," ").title() for p in PHASES],
        y=[ft.replace("_"," ").title() for ft in FTYPES],
        colorscale=[[0, "#060d1a"], [0.3, "#1e3a5f"], [0.7, "#3b82f6"], [1, "#93c5fd"]],
        text=heat.astype(str),
        texttemplate="%{text}",
        textfont=dict(color="white", size=12),
        hovertemplate="Failure: %{y}<br>Phase: %{x}<br>Episodes: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Episodes", thickness=12,
                      tickfont=dict(color="white"), titlefont=dict(color="white")),
    ))
    fig_heat.update_layout(
        xaxis_title="Task Phase", yaxis_title="Dominant Failure Type",
        height=380, margin=dict(t=30, b=50, l=10, r=10),
        plot_bgcolor="#060d1a", paper_bgcolor="#0a0f1e", font_color="#cbd5e1",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── RF vs BiLSTM comparison (shows once lstm_annotator.pt exists) ─────────
    comp_path = OUTPUT_DIR / "rf_vs_lstm_comparison.json"
    lstm_path = OUTPUT_DIR / "lstm_annotator.pt"

    st.markdown("---")
    st.markdown("#### RF vs BiLSTM Model Comparison")

    if lstm_path.exists():
        if not comp_path.exists():
            if st.button("▶ Run RF vs BiLSTM comparison", type="primary"):
                with st.spinner("Running comparison on held-out val episodes..."):
                    result = subprocess.run(
                        [sys.executable, "lstm_annotator.py", "--compare"],
                        capture_output=True, text=True, cwd=str(Path(__file__).parent)
                    )
                if result.returncode == 0:
                    st.success("Comparison complete!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(result.stderr[-500:])
        else:
            comp = json.loads(comp_path.read_text())
            rf_acc   = comp["rf"]["accuracy"]
            lstm_acc = comp["lstm"]["accuracy"]
            winner   = comp["winner"].upper()

            cw1, cw2, cw3 = st.columns(3)
            cw1.metric("RF accuracy",     f"{rf_acc:.3f}")
            cw2.metric("BiLSTM accuracy", f"{lstm_acc:.3f}",
                       f"{(lstm_acc - rf_acc)*100:+.1f}pp vs RF")
            cw3.metric("Winner", winner,
                       "temporal context wins" if winner == "LSTM" else "tabular wins")

            # per-class comparison bar chart — all 10 classes
            rf_f1s, lstm_f1s = [], []
            for cls in ALL_FAILURE_CLASSES:
                rf_f1s.append(comp["rf"]["per_class"].get(cls, {}).get("f1-score", 0))
                lstm_f1s.append(comp["lstm"]["per_class"].get(cls, {}).get("f1-score", 0))

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=[c.replace("_"," ").title() for c in ALL_FAILURE_CLASSES],
                y=rf_f1s, name="RF (sliding window)",
                marker_color="#3b82f6",
                text=[f"{v:.2f}" for v in rf_f1s], textposition="outside",
            ))
            fig_cmp.add_trace(go.Bar(
                x=[c.replace("_"," ").title() for c in ALL_FAILURE_CLASSES],
                y=lstm_f1s, name="FullContext MLP (episode context)",
                marker_color="#8b5cf6",
                text=[f"{v:.2f}" for v in lstm_f1s], textposition="outside",
            ))
            fig_cmp.update_layout(barmode="group", yaxis_range=[0, 1.15],
                                   xaxis_title="Failure class", yaxis_title="F1 score")
            st.plotly_chart(dark_chart(fig_cmp, 380), use_container_width=True)
    else:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #1e3a5f;border-radius:8px;
                    padding:1rem 1.4rem;color:#94a3b8;font-size:.87rem;'>
          FullContext MLP not yet compared. Run:
          <code style='color:#60a5fa;'>python lstm_annotator.py --compare</code>
          to benchmark RF vs full-episode-context MLP across all 10 failure classes.
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS-EPISODE PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(90deg,#0f2027,#203a43);border-radius:10px;
                padding:1rem 1.4rem;margin-bottom:1rem;border:1px solid #1e3a5f;'>
      <div style='color:#60a5fa;font-weight:700;font-size:1.05rem;'>
        🔬 Cross-Episode Pattern Detection
      </div>
      <div style='color:#94a3b8;font-size:.87rem;margin-top:.3rem;'>
        Are failures correlated with episode position? Workspace region?
        These patterns tell you <b style='color:#cbd5e1;'>exactly what to fix in the policy.</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cp1, cp2 = st.columns(2)

    with cp1:
        # Failure rate vs episode position (normalised 0→1)
        st.markdown("##### Failure Rate by Episode Position")
        st.caption("Where in the episode do failures concentrate?")
        BINS = 10
        bin_counts   = np.zeros(BINS)
        bin_failures = np.zeros(BINS)

        for r in episodes:
            fa   = r["failure_annotation"]
            labs = fa["step_labels"]
            T    = max(len(labs), 1)
            for i, lbl in enumerate(labs):
                b = min(int(i / T * BINS), BINS - 1)
                bin_counts[b]   += 1
                if lbl != "nominal":
                    bin_failures[b] += 1

        fail_rate_by_pos = np.where(bin_counts > 0, bin_failures / bin_counts, 0)
        bin_labels = [f"{int(i*100/BINS)}–{int((i+1)*100/BINS)}%" for i in range(BINS)]

        fig_pos = go.Figure(go.Bar(
            x=bin_labels, y=fail_rate_by_pos,
            marker_color=[
                "#ef4444" if v > fail_rate_by_pos.mean() * 1.3 else "#3b82f6"
                for v in fail_rate_by_pos
            ],
            hovertemplate="Position: %{x}<br>Failure rate: %{y:.1%}<extra></extra>",
        ))
        fig_pos.update_layout(xaxis_title="Episode position", yaxis_title="Failure rate",
                              yaxis_tickformat=".0%")
        st.plotly_chart(dark_chart(fig_pos, 280), use_container_width=True)
        peak_bin = bin_labels[int(np.argmax(fail_rate_by_pos))]
        st.caption(f"🔴 Failures peak at **{peak_bin}** of episodes — "
                   f"focus policy improvement here")

    with cp2:
        # Per-failure-class stacked breakdown
        st.markdown("##### Failure Type Breakdown by Episode Phase")
        st.caption("Which failures happen early vs late in episodes?")
        # Split into early (0–33%), mid (33–66%), late (66–100%)
        phase_names = ["Early (0–33%)", "Mid (33–66%)", "Late (66–100%)"]
        ftype_show  = [f for f in ALL_FAILURE_CLASSES if f != "nominal"]
        phase_data  = {ft: [0, 0, 0] for ft in ftype_show}

        for r in episodes:
            fa   = r["failure_annotation"]
            labs = fa["step_labels"]
            T    = max(len(labs), 1)
            for i, lbl in enumerate(labs):
                if lbl in ftype_show:
                    ph = min(int(i / T * 3), 2)
                    phase_data[lbl][ph] += 1

        fig_phase = go.Figure()
        for ft in ftype_show:
            vals = phase_data[ft]
            if sum(vals) == 0:
                continue
            fig_phase.add_trace(go.Bar(
                name=FAIL_LABEL.get(ft, ft),
                x=phase_names, y=vals,
                marker_color=FAIL_COLOR.get(ft, "#888"),
            ))
        fig_phase.update_layout(barmode="stack", xaxis_title="Episode phase",
                                yaxis_title="Step count")
        st.plotly_chart(dark_chart(fig_phase, 280), use_container_width=True)

    # Failure correlation with quality/anomaly
    st.markdown("##### Failure Mix Correlation with Episode Quality")
    st.caption("Each point = one episode. Color = dominant failure. Size = anomaly score.")
    fig_corr = go.Figure()
    for ftype in ALL_FAILURE_CLASSES:
        idx = [i for i, r in enumerate(episodes)
               if r["failure_annotation"]["dominant"] == ftype]
        if not idx:
            continue
        fig_corr.add_trace(go.Scatter(
            x=[quality_arr[i] for i in idx],
            y=[anomaly_arr[i] for i in idx],
            mode="markers",
            name=FAIL_LABEL.get(ftype, ftype),
            marker=dict(
                color=FAIL_COLOR.get(ftype, "#888"),
                size=[max(8, episodes[i]["failure_annotation"].get("review_rate", 0)*40) for i in idx],
                opacity=0.75,
                line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            ),
            hovertemplate="Episode %{customdata}<br>Quality: %{x:.3f}<br>Anomaly: %{y:.3f}<extra></extra>",
            customdata=[episodes[i]["episode_id"] for i in idx],
        ))
    fig_corr.update_layout(xaxis_title="Quality score", yaxis_title="Anomaly score",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(dark_chart(fig_corr, 340), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TREND MONITORING
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(90deg,#0a1628,#1a2a4a);border-radius:10px;
                padding:1rem 1.4rem;margin-bottom:1rem;border:1px solid #1e3a5f;'>
      <div style='color:#60a5fa;font-weight:700;font-size:1.05rem;'>
        📈 Trend Monitoring
      </div>
      <div style='color:#94a3b8;font-size:.87rem;margin-top:.3rem;'>
        Failure rate over time — see if policy updates are working or new failure modes are emerging.
        <b style='color:#cbd5e1;'>Episode index is used as a proxy for time</b> in this batch;
        in production this becomes a real timestamp.
      </div>
    </div>
    """, unsafe_allow_html=True)

    ROLL = max(5, len(episodes) // 10)   # rolling window size

    ep_indices    = list(range(len(episodes)))
    fail_rates    = [
        1 - r["failure_annotation"]["counts"].get("nominal", 0) /
        max(r["n_steps"], 1)
        for r in episodes
    ]
    quality_trend = [r["quality_score"]  for r in episodes]
    anomaly_trend = [r["anomaly_score"]  for r in episodes]

    # rolling average
    def rolling_avg(arr, w):
        out = []
        for i in range(len(arr)):
            s = max(0, i - w + 1)
            out.append(np.mean(arr[s:i+1]))
        return out

    roll_fail    = rolling_avg(fail_rates,    ROLL)
    roll_quality = rolling_avg(quality_trend, ROLL)
    roll_anomaly = rolling_avg(anomaly_trend, ROLL)

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("##### Rolling Failure Rate")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=ep_indices, y=fail_rates, mode="markers",
            marker=dict(size=4, color="#ef444460"), name="Per episode", showlegend=True,
        ))
        fig_trend.add_trace(go.Scatter(
            x=ep_indices, y=roll_fail, mode="lines",
            line=dict(color="#ef4444", width=2.5), name=f"Rolling avg ({ROLL} eps)",
        ))
        fig_trend.update_layout(xaxis_title="Episode index", yaxis_title="Failure rate",
                                yaxis_tickformat=".0%")
        st.plotly_chart(dark_chart(fig_trend, 270), use_container_width=True)

    with t2:
        st.markdown("##### Rolling Quality & Anomaly Score")
        fig_qa_trend = go.Figure()
        fig_qa_trend.add_trace(go.Scatter(
            x=ep_indices, y=roll_quality, mode="lines",
            line=dict(color="#10b981", width=2.5), name="Quality (rolling)",
        ))
        fig_qa_trend.add_trace(go.Scatter(
            x=ep_indices, y=roll_anomaly, mode="lines",
            line=dict(color="#ef4444", width=2.5, dash="dot"), name="Anomaly (rolling)",
        ))
        fig_qa_trend.add_hline(y=0.65, line_dash="dash", line_color="#10b98160",
                               annotation_text="Training threshold",
                               annotation_font_color="#10b981")
        fig_qa_trend.update_layout(xaxis_title="Episode index", yaxis_title="Score")
        st.plotly_chart(dark_chart(fig_qa_trend, 270), use_container_width=True)

    # Failure type trend stacked area
    st.markdown("##### Failure Type Mix Over Time")
    st.caption(f"Rolling {ROLL}-episode window. Rising lines = emerging failure mode.")
    fig_stack = go.Figure()
    for ft in [f for f in ALL_FAILURE_CLASSES if f != "nominal"]:
        ft_series = [
            r["failure_annotation"]["counts"].get(ft, 0) / max(r["n_steps"], 1)
            for r in episodes
        ]
        roll_ft = rolling_avg(ft_series, ROLL)
        if max(roll_ft) < 0.005:
            continue
        fig_stack.add_trace(go.Scatter(
            x=ep_indices, y=roll_ft, mode="lines",
            name=FAIL_LABEL.get(ft, ft),
            line=dict(color=FAIL_COLOR.get(ft, "#888"), width=2),
            stackgroup="one",
            hovertemplate=f"{ft}: %{{y:.1%}}<extra></extra>",
        ))
    fig_stack.update_layout(xaxis_title="Episode index", yaxis_title="Fraction of steps",
                            yaxis_tickformat=".0%",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(dark_chart(fig_stack, 300), use_container_width=True)
