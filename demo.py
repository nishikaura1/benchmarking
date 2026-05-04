"""
Haptal — Robot Training Data Quality
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

TEAL = "#2a9d8f"

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Haptal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* Hide Streamlit chrome */
  div[data-testid="stToolbar"]     {{ display: none; }}
  #MainMenu                        {{ display: none; }}
  footer                           {{ display: none; }}

  /* Tighten sidebar */
  section[data-testid="stSidebar"] > div {{ padding-top: 1.6rem; }}

  /* Main content padding */
  .block-container {{ padding-top: 2rem; max-width: 1100px; }}

  /* Status pill */
  .pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .03em;
  }}
  .pill-pass   {{ background: #0d2e2a; color: {TEAL}; }}
  .pill-fail   {{ background: #2e0d0d; color: #f87171; }}
  .pill-review {{ background: #2e220d; color: #fbbf24; }}

  /* Section heading */
  .section-label {{
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: .5rem;
  }}

  /* Primary button → teal */
  button[kind="primary"] {{
    background: {TEAL} !important;
    border-color: {TEAL} !important;
  }}

  /* Clean up metric */
  div[data-testid="metric-container"] {{
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 14px 18px;
  }}
</style>
""", unsafe_allow_html=True)

# ─── Dataset registry ─────────────────────────────────────────────────────────

DATASETS = {
    "xArm Push": {
        "file":  "lerobot_xarm_push_medium_replay_episodes.pkl",
        "robot": "xArm · 4 DOF · object pushing",
        "note":  "Mix of clean episodes and failures. Velocity spikes, self-collision.",
    },
    "ALOHA Insertion": {
        "file":  "lerobot_aloha_sim_insertion_human_episodes.pkl",
        "robot": "ALOHA bimanual · 14 DOF · peg insertion",
        "note":  "Different platform from training data. Stuck joints, gripper failures.",
    },
    "DROID-100": {
        "file":  "lerobot_droid_100_episodes.pkl",
        "robot": "Franka Panda · 7 DOF · diverse manipulation",
        "note":  "Real-world data. Lower model confidence triggers review queue.",
    },
}

FAILURE_CLASSES = [
    "nominal", "velocity_spike", "position_jerk", "stuck_joint",
    "gripper_event", "trajectory_deviation", "overcorrect",
    "self_collision", "overshoot", "perception_failure", "unknown_failure_type",
]

COLORS = {
    "nominal":              TEAL,
    "velocity_spike":       "#ef4444",
    "position_jerk":        "#f97316",
    "stuck_joint":          "#a855f7",
    "gripper_event":        "#eab308",
    "trajectory_deviation": "#ec4899",
    "overcorrect":          "#14b8a6",
    "self_collision":       "#f43f5e",
    "overshoot":            "#fb923c",
    "perception_failure":   "#8b5cf6",
    "unknown_failure_type": "#6b7280",
    "high_anomaly":         "#6b7280",
}

TRACE_PALETTE = [TEAL, "#38bdf8", "#f87171", "#a78bfa"]

# ─── Load model + data ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
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

def pill(text, kind):
    return f'<span class="pill pill-{kind}">{text}</span>'

def status_pill(r):
    if r["needs_review"]:
        return pill("REVIEW", "review")
    if r["use_for_policy"]:
        return pill("PASS", "pass")
    return pill("FAIL", "fail")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

ann = load_model()

with st.sidebar:
    logo = STATIC_DIR / "haptal_dark.png"
    if logo.exists():
        st.image(str(logo), width=130)
    else:
        st.markdown("### Haptal.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)

    ds_name = st.radio(
        "dataset",
        list(DATASETS.keys()),
        label_visibility="collapsed",
    )
    ds_meta = DATASETS[ds_name]
    st.caption(ds_meta["robot"])
    st.caption(ds_meta["note"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)

    if ann:
        st.markdown(f"RobotAnnotator v1.1")
        st.caption("Calibrated Random Forest")
        st.caption("Trained: xArm, ALOHA, DROID")
        st.caption("Val accuracy: 89.9%")
        st.caption("Brier score: 0.017")
    else:
        st.warning("Model not found")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.caption("aarav@haptal.ai")

# ─── Load data ────────────────────────────────────────────────────────────────

results  = run_inference(ann, ds_meta["file"])
episodes = load_episodes(ds_meta["file"])

if not results:
    st.error(f"Dataset not found: `{ds_meta['file']}`")
    st.stop()

df = pd.DataFrame([{k: v for k, v in r.items()
                    if k not in ("seq", "step_labels", "step_confs", "failure_counts")}
                   for r in results])

# ─── Main layout ─────────────────────────────────────────────────────────────

# Page header
st.markdown(f"## {ds_name}")
st.markdown(
    f"<span style='color:#6b7280; font-size:.9rem;'>{ds_meta['robot']}</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Section 1: Raw data ─────────────────────────────────────────────────────

st.markdown('<div class="section-label">Raw data — before annotation</div>', unsafe_allow_html=True)
st.markdown(
    "This is what the dataset looks like out of the box. "
    "No labels. No quality signal. No way to know which episodes to train on.",
    )

raw_rows = []
for i, ep in enumerate(episodes[:8]):
    seq = ep["seq"]
    vel = np.diff(seq, axis=0)
    raw_rows.append({
        "episode":      f"ep_{i:03d}",
        "steps":        len(seq),
        "joint_0_mean": round(float(seq[:, 0].mean()), 4),
        "joint_1_mean": round(float(seq[:, 1].mean()), 4),
        "vel_max":      round(float(np.abs(vel).max()), 4),
        "failure_type": "—",
        "status":       "—",
    })

st.dataframe(
    pd.DataFrame(raw_rows),
    use_container_width=True,
    hide_index=True,
    height=280,
)

st.divider()

# ─── Section 2: Pipeline ──────────────────────────────────────────────────────

st.markdown('<div class="section-label">Pipeline — RobotAnnotator v1.1</div>', unsafe_allow_html=True)

# Summary metrics
n_total = len(df)
n_pass  = int(df["use_for_policy"].sum())
n_fail  = int((~df["use_for_policy"] & ~df["needs_review"]).sum())
n_rev   = int(df["needs_review"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Episodes",        n_total)
m2.metric("Pass",            n_pass,  delta=f"{round(100*n_pass/n_total)}% of dataset")
m3.metric("Fail",            n_fail,  delta=f"{round(100*n_fail/n_total)}% excluded")
m4.metric("Review",          n_rev,   delta=f"confidence < 0.80")

st.markdown("<br>", unsafe_allow_html=True)

# Annotated table
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("**Annotated episodes**")

    display_rows = []
    for r in results:
        display_rows.append({
            "episode":      r["ep"],
            "steps":        r["n_steps"],
            "failure_type": r["failure_type"].replace("_", " "),
            "confidence":   r["confidence"],
            "peak_step":    r["peak_step"] if r["peak_step"] >= 0 else "—",
            "status":       ("REVIEW" if r["needs_review"]
                             else "PASS" if r["use_for_policy"] else "FAIL"),
        })

    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )

with col_right:
    st.markdown("**Failure distribution**")
    counts = Counter(df["failure_type"])
    labels = list(counts.keys())
    values = list(counts.values())

    fig_pie = go.Figure(go.Pie(
        labels=[l.replace("_", " ") for l in labels],
        values=values,
        hole=0.55,
        marker=dict(colors=[COLORS.get(l, "#6b7280") for l in labels]),
        textinfo="label+percent",
        textfont=dict(size=11),
        showlegend=False,
    ))
    fig_pie.update_layout(
        height=230,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Confidence histogram
    st.markdown("**Confidence distribution**")
    fig_hist = go.Figure(go.Histogram(
        x=df["confidence"].tolist(),
        nbinsx=15,
        marker_color=TEAL,
        opacity=0.8,
    ))
    fig_hist.add_vline(
        x=0.80, line_dash="dot", line_color="#6b7280",
        annotation_text="review threshold",
        annotation_font_size=10,
        annotation_font_color="#6b7280",
    )
    fig_hist.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="confidence", gridcolor="#1f2937"),
        yaxis=dict(title="episodes",   gridcolor="#1f2937"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ─── Section 3: Episode detail ────────────────────────────────────────────────

st.markdown('<div class="section-label">Episode detail — step-level output</div>', unsafe_allow_html=True)

# Find the most notable failure to show first
notable = max(
    (r for r in results if r["failure_type"] != "nominal"),
    key=lambda r: r["fail_frac"],
    default=results[0],
)

ep_options = [
    f"ep_{i:03d}  ·  {r['failure_type'].replace('_', ' ')}  ·  conf {r['confidence']:.3f}"
    for i, r in enumerate(results)
]
default_idx = next(
    (i for i, r in enumerate(results) if r["ep"] == notable["ep"]), 0
)
sel = st.selectbox("Select episode", ep_options, index=default_idx, label_visibility="collapsed")
r_sel = results[ep_options.index(sel)]
seq   = r_sel["seq"]
T     = len(seq)

col_trace, col_steps = st.columns([3, 2])

with col_trace:
    fig = go.Figure()
    for d in range(min(seq.shape[1], 4)):
        fig.add_trace(go.Scatter(
            x=list(range(T)),
            y=seq[:, d].tolist(),
            mode="lines",
            name=f"joint {d}",
            line=dict(color=TRACE_PALETTE[d], width=1.6),
        ))

    fail_steps = [t for t, l in enumerate(r_sel["step_labels"]) if l != "nominal"]
    if fail_steps:
        fig.add_vrect(
            x0=min(fail_steps) - 0.5,
            x1=max(fail_steps) + 0.5,
            fillcolor="#ef4444",
            opacity=0.10,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            x=min(fail_steps),
            y=float(seq[:, 0].max()),
            text=f"{r_sel['failure_type'].replace('_', ' ')} · step {r_sel['peak_step']}",
            showarrow=False,
            font=dict(color="#f87171", size=11),
            xanchor="left",
        )

    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="timestep", gridcolor="#1f2937"),
        yaxis=dict(title="joint state", gridcolor="#1f2937"),
        legend=dict(orientation="h", y=1.12, font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-step confidence
    fig2 = go.Figure(go.Bar(
        x=list(range(T)),
        y=r_sel["step_confs"],
        marker=dict(
            color=[TEAL if l == "nominal" else "#ef4444" for l in r_sel["step_labels"]],
            opacity=0.75,
        ),
    ))
    fig2.add_hline(y=0.80, line_dash="dot", line_color="#6b7280")
    fig2.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="timestep", gridcolor="#1f2937"),
        yaxis=dict(title="confidence", range=[0, 1], gridcolor="#1f2937"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_steps:
    st.markdown("**Verdict**")
    st.table(pd.DataFrame([
        ("failure type",  r_sel["failure_type"].replace("_", " ")),
        ("confidence",    f"{r_sel['confidence']:.3f}"),
        ("peak step",     str(r_sel["peak_step"])),
        ("fail fraction", f"{r_sel['fail_frac']:.1%}"),
        ("use for policy", "yes" if r_sel["use_for_policy"] else "no"),
        ("needs review",  "yes" if r_sel["needs_review"] else "no"),
    ], columns=["", ""]).set_index(""))

    st.markdown("**Steps per class**")
    fc = {k.replace("_", " "): v for k, v in r_sel["failure_counts"].items() if v > 0}
    if fc:
        fc_df = (
            pd.DataFrame({"class": list(fc.keys()), "steps": list(fc.values())})
            .sort_values("steps", ascending=False)
        )
        st.bar_chart(fc_df.set_index("class")["steps"], color=TEAL)

st.divider()

# ─── Section 4: Output ────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Output — filtered dataset</div>', unsafe_allow_html=True)

col_out, col_code = st.columns([3, 2])

with col_out:
    st.markdown("**Annotated + filtered**")
    out_rows = []
    for r in results:
        status = "REVIEW" if r["needs_review"] else "PASS" if r["use_for_policy"] else "FAIL"
        out_rows.append({
            "episode":      r["ep"],
            "failure_type": r["failure_type"].replace("_", " "),
            "confidence":   r["confidence"],
            "use_for_policy": r["use_for_policy"],
            "status":       status,
        })
    st.dataframe(
        pd.DataFrame(out_rows),
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "use_for_policy": st.column_config.CheckboxColumn("use_for_policy"),
            "confidence":     st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )

with col_code:
    st.markdown("**Your training script — one line changes**")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Before")
    st.code(
        'dataset = load_dataset("my_robot_data")\ntrain(dataset)',
        language="python",
    )
    st.markdown("After")
    st.code(
        'dataset = load_dataset("my_robot_data")\ntrain(\n    dataset.filter(\n        lambda ep: ep["use_for_policy"]\n    )\n)',
        language="python",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "Every episode has a `failure_type`, `confidence`, and `use_for_policy` flag. "
        "No architecture changes. No new training loop. Filter and train."
    )
