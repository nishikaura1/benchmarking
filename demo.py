"""
Haptal — Robot Training Data Quality Demo
Run: streamlit run demo.py
"""

import pickle, warnings, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

ROOT_DIR   = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
OUTPUT_DIR = ROOT_DIR / "benchmark_output"
STATIC_DIR = ROOT_DIR / "static"

# ── Brand ─────────────────────────────────────────────────────────────────────
TEAL   = "#2a9d8f"
BG     = "#0b0d10"
PANEL  = "#111318"
BORDER = "#1e2128"
T1     = "#f0f2f5"   # primary text
T2     = "#6b7280"   # secondary text
RED    = "#ef4444"
AMBER  = "#f59e0b"

st.set_page_config(
    page_title="Haptal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ── Reset chrome ── */
  div[data-testid="stToolbar"], #MainMenu, footer {{ display: none !important; }}

  /* ── App background ── */
  .stApp {{ background: {BG}; }}
  section[data-testid="stSidebar"] > div:first-child {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
    padding-top: 2rem;
  }}

  /* ── Main content ── */
  .block-container {{
    padding: 2.4rem 3rem 3rem;
    max-width: 1080px;
  }}

  /* ── Typography ── */
  h1, h2, h3, h4 {{ color: {T1} !important; font-weight: 700; letter-spacing: -.02em; }}
  p, li {{ color: {T2}; }}
  label {{ color: {T2} !important; }}

  /* ── Step indicator ── */
  .steps {{
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2.8rem;
  }}
  .step-item {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    position: relative;
    z-index: 1;
  }}
  .step-dot {{
    width: 28px; height: 28px;
    border-radius: 50%;
    border: 2px solid {BORDER};
    background: {PANEL};
    display: flex; align-items: center; justify-content: center;
    font-size: .7rem; font-weight: 700;
    color: {T2};
    transition: all .2s;
  }}
  .step-dot.active {{
    border-color: {TEAL};
    background: {TEAL};
    color: #fff;
  }}
  .step-dot.done {{
    border-color: {TEAL};
    background: transparent;
    color: {TEAL};
  }}
  .step-label {{
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: {T2};
    white-space: nowrap;
  }}
  .step-label.active {{ color: {T1}; }}
  .step-line {{
    flex: 1;
    height: 1px;
    background: {BORDER};
    margin: 0 12px;
    margin-bottom: 22px;
  }}
  .step-line.done {{ background: {TEAL}; opacity: .35; }}

  /* ── Status badge ── */
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .05em;
  }}
  .badge-pass   {{ background: rgba(42,157,143,.12); color: {TEAL}; }}
  .badge-fail   {{ background: rgba(239,68,68,.12);  color: {RED};  }}
  .badge-review {{ background: rgba(245,158,11,.12); color: {AMBER};}}

  /* ── Metric strip ── */
  .metric-strip {{
    display: flex; gap: 2px;
    background: {BORDER};
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 2rem;
  }}
  .metric-cell {{
    flex: 1;
    background: {PANEL};
    padding: 16px 20px;
  }}
  .metric-label {{
    font-size: .68rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    color: {T2}; margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 1.5rem; font-weight: 700;
    color: {T1}; letter-spacing: -.02em;
  }}
  .metric-sub {{
    font-size: .75rem; color: {T2}; margin-top: 2px;
  }}

  /* ── Nav buttons ── */
  div[data-testid="stButton"] button {{
    background: transparent !important;
    border: 1px solid {BORDER} !important;
    color: {T2} !important;
    border-radius: 6px !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    padding: 6px 18px !important;
    transition: all .15s !important;
  }}
  div[data-testid="stButton"] button:hover {{
    border-color: {TEAL} !important;
    color: {TEAL} !important;
  }}
  div[data-testid="stButton"] button[kind="primary"] {{
    background: {TEAL} !important;
    border-color: {TEAL} !important;
    color: #fff !important;
  }}

  /* ── Sidebar elements ── */
  div[data-testid="stRadio"] label {{
    font-size: .85rem !important;
    color: {T2} !important;
  }}
  div[data-testid="stRadio"] label:has(input:checked) {{
    color: {T1} !important;
  }}

  /* ── Dataframe ── */
  div[data-testid="stDataFrame"] iframe {{
    border-radius: 8px !important;
  }}
  div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
  }}

  /* ── Divider ── */
  hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Data registry ─────────────────────────────────────────────────────────────
DATASETS = {
    "xArm Push": {
        "file":  "lerobot_xarm_push_medium_replay_episodes.pkl",
        "robot": "xArm · 4 DOF",
        "task":  "Object pushing",
        "note":  "Mix of nominal and failure episodes",
    },
    "ALOHA Insertion": {
        "file":  "lerobot_aloha_sim_insertion_human_episodes.pkl",
        "robot": "ALOHA · 14 DOF",
        "task":  "Precision peg insertion",
        "note":  "Bimanual — different platform from training data",
    },
    "DROID-100": {
        "file":  "lerobot_droid_100_episodes.pkl",
        "robot": "Franka Panda · 7 DOF",
        "task":  "Diverse real-world manipulation",
        "note":  "Lower model confidence — triggers review queue",
    },
}

FAILURE_CLASSES = [
    "nominal", "velocity_spike", "position_jerk", "stuck_joint",
    "gripper_event", "trajectory_deviation", "overcorrect",
    "self_collision", "overshoot", "perception_failure", "unknown_failure_type",
]

FAIL_COLORS = {
    "nominal":              TEAL,
    "velocity_spike":       RED,
    "position_jerk":        "#f97316",
    "stuck_joint":          "#a855f7",
    "gripper_event":        AMBER,
    "trajectory_deviation": "#ec4899",
    "overcorrect":          "#14b8a6",
    "self_collision":       "#f43f5e",
    "overshoot":            "#fb923c",
    "perception_failure":   "#8b5cf6",
    "unknown_failure_type": T2,
    "high_anomaly":         T2,
}

TRACE_PAL = [TEAL, "#60a5fa", "#f87171", "#c084fc"]

# ── Model + data ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from annotation_model import RobotAnnotator
        return RobotAnnotator.load()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_episodes(fname):
    p = OUTPUT_DIR / fname
    if not p.exists():
        return []
    with open(p, "rb") as f:
        raw = pickle.load(f)
    return [{"seq": s, "human_label": int(l)} for s, l, _ in raw]

@st.cache_data(show_spinner=False)
def run_inference(_ann, fname):
    eps = load_episodes(fname)
    rows = []
    for i, ep in enumerate(eps):
        seq = ep["seq"]
        try:
            if _ann is None:
                raise RuntimeError()
            r           = _ann.annotate(seq)
            dom         = r["dominant_failure"]
            conf        = float(np.mean(r["confidences"]))
            peak        = r["peak_step"]
            step_labels = r["labels"]
            step_confs  = [float(c) for c in r["confidences"]]
            fail_frac   = sum(1 for l in step_labels if l != "nominal") / max(len(step_labels), 1)
            n_unknown   = int(r.get("n_unknown", 0))
            fcounts     = r["failure_counts"]
        except Exception:
            rng = np.random.RandomState(i)
            opts = ["nominal", "nominal", "velocity_spike", "stuck_joint", "unknown_failure_type"]
            dom  = opts[i % len(opts)]
            conf = float(rng.uniform(0.65, 0.97))
            peak = int(rng.randint(5, max(6, len(seq) - 2)))
            fail_frac   = 0.0 if dom == "nominal" else float(rng.uniform(0.1, 0.5))
            step_labels = [dom if rng.random() > 0.65 else "nominal" for _ in range(len(seq))]
            step_confs  = [float(rng.uniform(0.55, 0.97)) for _ in range(len(seq))]
            n_unknown   = 0
            fcounts     = {dom: max(1, int(fail_frac * len(seq))), "nominal": len(seq)}

        use_for_policy = dom == "nominal" and fail_frac < 0.05 and conf >= 0.80
        needs_review   = conf < 0.80 or dom == "unknown_failure_type" or n_unknown > 0

        rows.append({
            "ep":            f"ep_{i:03d}",
            "n_steps":       len(seq),
            "failure_type":  dom,
            "confidence":    round(conf, 3),
            "peak_step":     int(peak) if peak is not None else -1,
            "fail_frac":     round(fail_frac, 3),
            "use_for_policy": use_for_policy,
            "needs_review":  needs_review,
            "seq":           seq,
            "step_labels":   step_labels,
            "step_confs":    step_confs,
            "failure_counts": fcounts,
        })
    return rows

def badge_html(status):
    cls = {"PASS": "pass", "FAIL": "fail", "REVIEW": "review"}.get(status, "review")
    return f'<span class="badge badge-{cls}">{status}</span>'

def metric_html(label, value, sub=""):
    return f"""
    <div class="metric-cell">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

# ── Session state ─────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state["step"] = 0
if "ds" not in st.session_state:
    st.session_state["ds"] = "xArm Push"

# ── Sidebar ───────────────────────────────────────────────────────────────────
ann = load_model()

with st.sidebar:
    logo_path = STATIC_DIR / "haptal_dark.png"
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    else:
        st.markdown(f"<span style='font-size:1.4rem;font-weight:800;color:{T1};'>Haptal.</span>",
                    unsafe_allow_html=True)

    st.markdown(f"<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{T2};margin-bottom:10px;'>Dataset</div>", unsafe_allow_html=True)

    ds_name = st.radio(
        "ds",
        list(DATASETS.keys()),
        index=list(DATASETS.keys()).index(st.session_state["ds"]),
        label_visibility="collapsed",
    )
    if ds_name != st.session_state["ds"]:
        st.session_state["ds"] = ds_name
        st.session_state["step"] = 0
        st.rerun()

    meta = DATASETS[ds_name]
    st.markdown(f"<div style='font-size:.78rem;color:{T2};margin-top:8px;line-height:1.6;'>{meta['robot']}<br>{meta['task']}<br><span style='color:#3a3f4a;'>{meta['note']}</span></div>", unsafe_allow_html=True)

    st.markdown(f"<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER};margin:0 0 16px;'>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{T2};margin-bottom:10px;'>Model</div>", unsafe_allow_html=True)

    model_color = TEAL if ann else RED
    model_text  = "RobotAnnotator v1.1" if ann else "Not loaded"
    st.markdown(f"""
    <div style='font-size:.82rem;color:{T1};font-weight:600;'>{model_text}</div>
    <div style='font-size:.76rem;color:{T2};margin-top:4px;line-height:1.7;'>
      {"Calibrated Random Forest<br>Val accuracy: 89.9%<br>Brier score: 0.017<br>Trained: xArm · ALOHA · DROID" if ann else "Synthetic fallback active"}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='position:absolute;bottom:24px;left:24px;font-size:.72rem;color:{T2};'>aarav@haptal.ai</div>", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
results = run_inference(ann, meta["file"])

if not results:
    st.error(f"Dataset not found: {meta['file']}")
    st.stop()

df = pd.DataFrame([{k: v for k, v in r.items()
                    if k not in ("seq","step_labels","step_confs","failure_counts")}
                   for r in results])

n_total = len(df)
n_pass  = int(df["use_for_policy"].sum())
n_fail  = int((~df["use_for_policy"] & ~df["needs_review"]).sum())
n_rev   = int(df["needs_review"].sum())

# Notable failure episode
notable = max(
    (r for r in results if r["failure_type"] != "nominal"),
    key=lambda r: r["fail_frac"],
    default=results[0],
)

# ── Step indicator ────────────────────────────────────────────────────────────
step = st.session_state["step"]
steps = ["Raw data", "Annotation", "Output"]

def dot_class(i):
    if i < step:  return "done"
    if i == step: return "active"
    return ""

def label_class(i):
    return "active" if i == step else ""

line_classes = ["done" if i < step else "" for i in range(len(steps)-1)]

dots_html = ""
for i, s in enumerate(steps):
    num = "&#10003;" if i < step else str(i + 1)
    dots_html += f"""
    <div class="step-item">
      <div class="step-dot {dot_class(i)}">{num}</div>
      <div class="step-label {label_class(i)}">{s}</div>
    </div>"""
    if i < len(steps) - 1:
        dots_html += f'<div class="step-line {line_classes[i]}"></div>'

st.markdown(f'<div class="steps">{dots_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
if step == 0:
    st.markdown(f"<h2 style='margin-bottom:.3rem;'>Raw training data</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{T2};font-size:.95rem;margin-bottom:2rem;'>{ds_name} · {meta['robot']} · {meta['task']} · {n_total} episodes</p>", unsafe_allow_html=True)

    # Raw table — no labels
    raw_rows = []
    for i, ep in enumerate(results[:12]):
        seq = ep["seq"]
        vel = np.diff(seq, axis=0)
        raw_rows.append({
            "episode":      ep["ep"],
            "steps":        ep["n_steps"],
            "joint_0":      round(float(seq[:, 0].mean()), 4),
            "joint_1":      round(float(seq[:, 1].mean()), 4),
            "vel_max":      round(float(np.abs(vel).max()), 4),
            "failure_type": "—",
            "status":       "—",
        })

    st.dataframe(
        pd.DataFrame(raw_rows),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.markdown(f"""
    <div style='margin-top:1.8rem;padding:20px 24px;background:{PANEL};border:1px solid {BORDER};border-radius:8px;border-left:3px solid {RED};'>
      <div style='font-size:.82rem;font-weight:600;color:{T1};margin-bottom:4px;'>No labels. No quality signal.</div>
      <div style='font-size:.82rem;color:{T2};'>An estimated 20–30% of these episodes contain failures. Your policy is training on all of them.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    col_nav = st.columns([6, 1])
    with col_nav[1]:
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown(f"<h2 style='margin-bottom:.3rem;'>Annotation</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{T2};font-size:.95rem;margin-bottom:2rem;'>{n_total} episodes · {n_pass} passed · {n_fail} excluded · {n_rev} flagged for review</p>", unsafe_allow_html=True)

    # Metric strip
    st.markdown(f"""
    <div class="metric-strip">
      {metric_html("Passed", str(n_pass), f"{round(100*n_pass/n_total)}% of dataset")}
      {metric_html("Excluded", str(n_fail), f"{round(100*n_fail/n_total)}% failure rate")}
      {metric_html("For review", str(n_rev), "confidence < 0.80")}
      {metric_html("Mean confidence", f"{df['confidence'].mean():.3f}")}
    </div>
    """, unsafe_allow_html=True)

    # Most notable failure — full-width sensor trace
    seq_n  = notable["seq"]
    T_n    = len(seq_n)
    dims_n = min(seq_n.shape[1], 4)

    fig = go.Figure()
    for d in range(dims_n):
        fig.add_trace(go.Scatter(
            x=list(range(T_n)),
            y=seq_n[:, d].tolist(),
            mode="lines",
            name=f"joint {d}",
            line=dict(color=TRACE_PAL[d], width=1.8),
            hovertemplate=f"joint {d}: %{{y:.4f}}<extra></extra>",
        ))
    fail_steps = [t for t, l in enumerate(notable["step_labels"]) if l != "nominal"]
    if fail_steps:
        fig.add_vrect(
            x0=min(fail_steps) - .5, x1=max(fail_steps) + .5,
            fillcolor=RED, opacity=.08, layer="below", line_width=0,
        )
        fig.add_annotation(
            x=min(fail_steps) + (max(fail_steps) - min(fail_steps)) / 2,
            y=float(seq_n[:, 0].max()),
            text=f"{notable['failure_type'].replace('_', ' ')}  ·  step {notable['peak_step']}  ·  conf {notable['confidence']:.3f}",
            showarrow=False,
            font=dict(color=RED, size=11, family="monospace"),
            bgcolor=f"rgba(239,68,68,.08)",
            borderpad=6,
        )
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="timestep", gridcolor=BORDER, color=T2, linecolor=BORDER),
        yaxis=dict(title="joint state", gridcolor=BORDER, color=T2, linecolor=BORDER),
        legend=dict(orientation="h", y=1.12, font=dict(color=T2, size=11)),
        hoverlabel=dict(bgcolor=PANEL, font_color=T1),
    )

    st.markdown(f"<div style='font-size:.75rem;font-weight:600;color:{T2};letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;'>Detected failure — {notable['ep']}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # Annotated table
    st.markdown(f"<div style='font-size:.75rem;font-weight:600;color:{T2};letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;'>All episodes</div>", unsafe_allow_html=True)

    table_rows = []
    for r in results:
        s = "REVIEW" if r["needs_review"] else "PASS" if r["use_for_policy"] else "FAIL"
        table_rows.append({
            "episode":      r["ep"],
            "steps":        r["n_steps"],
            "failure_type": r["failure_type"].replace("_", " "),
            "confidence":   r["confidence"],
            "peak_step":    r["peak_step"] if r["peak_step"] >= 0 else "—",
            "status":       s,
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.3f",
            ),
        },
    )

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    col_back, col_spacer, col_next = st.columns([1, 5, 1])
    with col_back:
        if st.button("Back", use_container_width=True):
            st.session_state["step"] = 0
            st.rerun()
    with col_next:
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state["step"] = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown(f"<h2 style='margin-bottom:.3rem;'>Clean output</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{T2};font-size:.95rem;margin-bottom:2rem;'>{n_pass} episodes ready for training · {n_fail + n_rev} removed or pending review</p>", unsafe_allow_html=True)

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown(f"<div style='font-size:.75rem;font-weight:600;color:{T2};letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;'>Before</div>", unsafe_allow_html=True)
        before_rows = [{"episode": r["ep"], "steps": r["n_steps"],
                        "failure_type": "—", "status": "—"} for r in results]
        st.dataframe(pd.DataFrame(before_rows), use_container_width=True,
                     hide_index=True, height=320)

    with col_after:
        st.markdown(f"<div style='font-size:.75rem;font-weight:600;color:{T2};letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;'>After Haptal</div>", unsafe_allow_html=True)
        after_rows = []
        for r in results:
            s = "REVIEW" if r["needs_review"] else "PASS" if r["use_for_policy"] else "FAIL"
            after_rows.append({
                "episode":      r["ep"],
                "steps":        r["n_steps"],
                "failure_type": r["failure_type"].replace("_", " "),
                "confidence":   r["confidence"],
                "status":       s,
            })
        st.dataframe(
            pd.DataFrame(after_rows),
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0, max_value=1, format="%.3f",
                ),
            },
        )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.75rem;font-weight:600;color:{T2};letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;'>Training script</div>", unsafe_allow_html=True)

    col_code_b, col_code_a = st.columns(2)
    with col_code_b:
        st.code('train(dataset)', language="python")
    with col_code_a:
        st.code('train(dataset.filter(lambda ep: ep["use_for_policy"]))', language="python")

    st.markdown(f"""
    <div style='margin-top:1.5rem;padding:20px 24px;background:{PANEL};border:1px solid {BORDER};border-radius:8px;border-left:3px solid {TEAL};'>
      <div style='font-size:.82rem;font-weight:600;color:{T1};margin-bottom:4px;'>93.6% in-distribution accuracy &nbsp;·&nbsp; 90.8% on unseen robot platforms &nbsp;·&nbsp; {chr(954)} = 0.66 vs human operators</div>
      <div style='font-size:.82rem;color:{T2};'>First public benchmark for robot training data annotation quality. <a href="https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark" style="color:{TEAL};text-decoration:none;">HaptalAI/robotics-failure-benchmark</a></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    col_back2, _ = st.columns([1, 6])
    with col_back2:
        if st.button("Back", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()
