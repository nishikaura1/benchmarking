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
T1     = "#f0f2f5"
T2     = "#6b7280"
RED    = "#ef4444"
AMBER  = "#f59e0b"
TRACE  = [TEAL, "#60a5fa", "#f87171", "#c084fc"]

st.set_page_config(
    page_title="Haptal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  div[data-testid="stToolbar"], #MainMenu, footer {{ display:none!important; }}
  .stApp {{ background:{BG}; }}
  section[data-testid="stSidebar"] > div:first-child {{
    background:{PANEL}; border-right:1px solid {BORDER}; padding-top:2rem;
  }}
  .block-container {{ padding:2.4rem 3rem 3rem; max-width:1080px; }}
  h1,h2,h3,h4 {{ color:{T1}!important; font-weight:700; letter-spacing:-.02em; }}
  p,li {{ color:{T2}; }}
  label {{ color:{T2}!important; }}

  .steps {{ display:flex; align-items:center; margin-bottom:2.8rem; }}
  .step-item {{ display:flex; flex-direction:column; align-items:center; gap:6px; }}
  .step-dot {{
    width:28px; height:28px; border-radius:50%;
    border:2px solid {BORDER}; background:{PANEL};
    display:flex; align-items:center; justify-content:center;
    font-size:.7rem; font-weight:700; color:{T2};
  }}
  .step-dot.active {{ border-color:{TEAL}; background:{TEAL}; color:#fff; }}
  .step-dot.done   {{ border-color:{TEAL}; background:transparent; color:{TEAL}; }}
  .step-label {{ font-size:.7rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:{T2}; white-space:nowrap; }}
  .step-label.active {{ color:{T1}; }}
  .step-line {{ flex:1; height:1px; background:{BORDER}; margin:0 12px; margin-bottom:22px; }}
  .step-line.done {{ background:{TEAL}; opacity:.35; }}

  .metric-strip {{ display:flex; gap:2px; background:{BORDER}; border-radius:8px; overflow:hidden; margin-bottom:2rem; }}
  .metric-cell {{ flex:1; background:{PANEL}; padding:16px 20px; }}
  .metric-lbl {{ font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{T2}; margin-bottom:4px; }}
  .metric-val {{ font-size:1.5rem; font-weight:700; color:{T1}; letter-spacing:-.02em; }}
  .metric-sub {{ font-size:.75rem; color:{T2}; margin-top:2px; }}

  .callout {{ padding:20px 24px; background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; margin-top:1.5rem; }}
  .callout-red   {{ border-left:3px solid {RED};  }}
  .callout-teal  {{ border-left:3px solid {TEAL}; }}

  div[data-testid="stButton"] button {{
    background:transparent!important; border:1px solid {BORDER}!important;
    color:{T2}!important; border-radius:6px!important;
    font-size:.82rem!important; font-weight:600!important;
  }}
  div[data-testid="stButton"] button:hover {{ border-color:{TEAL}!important; color:{TEAL}!important; }}
  div[data-testid="stButton"] button[kind="primary"] {{
    background:{TEAL}!important; border-color:{TEAL}!important; color:#fff!important;
  }}
  div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:8px; overflow:hidden; }}
  hr {{ border-color:{BORDER}!important; }}
  .sec-label {{ font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{T2}; margin-bottom:10px; }}
</style>
""", unsafe_allow_html=True)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = {
    "xArm Push": {
        "file":    "lerobot_xarm_push_medium_replay_episodes.pkl",
        "robot":   "xArm · 4 DOF",
        "task":    "Object pushing",
        "note":    "Mix of nominal and failure episodes",
        "dof":     4, "steps": 25,
        "failures": ["velocity_spike", "self_collision", "nominal"],
    },
    "ALOHA Insertion": {
        "file":    "lerobot_aloha_sim_insertion_human_episodes.pkl",
        "robot":   "ALOHA bimanual · 14 DOF",
        "task":    "Precision peg insertion",
        "note":    "Different platform from training data",
        "dof":     14, "steps": 80,
        "failures": ["stuck_joint", "gripper_event", "nominal"],
    },
    "DROID-100": {
        "file":    "lerobot_droid_100_episodes.pkl",
        "robot":   "Franka Panda · 7 DOF",
        "task":    "Diverse real-world manipulation",
        "note":    "Lower confidence — triggers review queue",
        "dof":     7, "steps": 60,
        "failures": ["trajectory_deviation", "unknown_failure_type", "nominal"],
    },
}

FAILURE_CLASSES = [
    "nominal","velocity_spike","position_jerk","stuck_joint",
    "gripper_event","trajectory_deviation","overcorrect",
    "self_collision","overshoot","perception_failure","unknown_failure_type",
]
FAIL_COLORS = {
    "nominal": TEAL, "velocity_spike": RED, "position_jerk": "#f97316",
    "stuck_joint": "#a855f7", "gripper_event": AMBER,
    "trajectory_deviation": "#ec4899", "overcorrect": "#14b8a6",
    "self_collision": "#f43f5e", "overshoot": "#fb923c",
    "perception_failure": "#8b5cf6", "unknown_failure_type": T2,
    "high_anomaly": T2,
}

# ── Synthetic data generation (used when real pkl not available) ───────────────

def _make_episode(ftype, dof, steps, rng):
    """Generate a realistic-looking robot trajectory with optional injected failure."""
    t    = np.linspace(0, 2 * np.pi, steps)
    seq  = np.column_stack([
        0.3 * np.sin(t * (j * 0.7 + 0.5) + rng.uniform(0, np.pi))
        + rng.randn(steps) * 0.018
        for j in range(dof)
    ]).astype(np.float32)

    if ftype == "velocity_spike":
        at  = int(steps * rng.uniform(0.3, 0.7))
        dim = rng.randint(0, dof)
        seq[at, dim] += rng.uniform(1.8, 3.2)

    elif ftype == "stuck_joint":
        at  = int(steps * rng.uniform(0.35, 0.55))
        dim = rng.randint(0, dof)
        seq[at:, dim] = seq[at, dim] + rng.randn(steps - at) * 0.004

    elif ftype == "self_collision":
        at = int(steps * 0.4)
        seq[at:, 0] += np.linspace(0, 0.6, steps - at)
        seq[at:, min(1, dof-1)] -= np.linspace(0, 0.6, steps - at)

    elif ftype == "gripper_event":
        at = int(steps * rng.uniform(0.4, 0.6))
        seq[at, -1] += rng.uniform(1.5, 2.5)

    elif ftype == "trajectory_deviation":
        at    = int(steps * 0.4)
        drift = rng.uniform(0.4, 0.9)
        seq[at:, 0] += np.linspace(0, drift, steps - at)

    elif ftype == "unknown_failure_type":
        at = int(steps * rng.uniform(0.4, 0.6))
        seq[at:at+4] += rng.randn(4, dof) * 0.25

    return seq

@st.cache_data(show_spinner=False)
def load_episodes(fname, dof, steps, failure_types, n=20):
    """Load real pkl if available, otherwise generate synthetic."""
    p = OUTPUT_DIR / fname
    if p.exists():
        with open(p, "rb") as f:
            raw = pickle.load(f)
        return [{"seq": s, "human_label": int(l)} for s, l, _ in raw]

    # Synthetic fallback — deterministic, realistic
    rng  = np.random.RandomState(abs(hash(fname)) % (2**31))
    pool = (["nominal"] * (n // 2)
            + [ft for ft in failure_types if ft != "nominal"] * (n // 3)
            + ["nominal"] * n)[:n]
    rng.shuffle(pool)
    return [{"seq": _make_episode(ft, dof, steps, rng), "human_label": 0 if ft=="nominal" else 1}
            for ft in pool]

# ── Inline annotator (used when full model pkl not available) ──────────────────

class _QuickAnnotator:
    """Threshold-based physics annotator — deterministic, no pkl needed."""
    def annotate(self, seq):
        seq  = np.array(seq, dtype=np.float32)
        T, D = seq.shape
        if T < 3:
            return {"labels":["nominal"]*T, "confidences":[0.95]*T,
                    "dominant_failure":"nominal", "peak_step":-1,
                    "failure_counts":{"nominal":T}, "n_unknown":0}

        vel      = np.diff(seq, axis=0)            # (T-1, D)
        vel_full = np.vstack([vel[:1], vel])        # pad to (T, D)
        mean_vel = np.mean(np.abs(vel)) + 1e-9
        vel_thr  = mean_vel * 2.8

        labels = []
        confs  = []
        for t in range(T):
            vm = float(np.max(np.abs(vel_full[t])))
            if t > 0 and vm > vel_thr * 2.2:
                labels.append("velocity_spike");      confs.append(round(min(0.97, 0.88 + vm / vel_thr * 0.02), 3))
            elif t > 3 and vm < mean_vel * 0.06 and float(np.max(np.abs(vel_full[t-2]))) > mean_vel * 0.5:
                labels.append("stuck_joint");         confs.append(0.89)
            elif t > 1 and D > 1 and float(np.dot(vel_full[t], vel_full[t-1])) < -vel_thr * 0.8:
                labels.append("self_collision");      confs.append(0.86)
            elif t > 3:
                drift = float(np.max(np.abs(seq[t] - seq[max(0,t-4)]))) / (4 * mean_vel + 1e-9)
                if drift > 3.5:
                    labels.append("trajectory_deviation"); confs.append(0.84)
                else:
                    labels.append("nominal");             confs.append(round(min(0.97, 0.91 + np.random.uniform(0,.04)), 3))
            else:
                labels.append("nominal");             confs.append(0.94)

        counts    = dict(Counter(labels))
        dom       = Counter(labels).most_common(1)[0][0]
        fail_steps = [i for i,l in enumerate(labels) if l != "nominal"]
        if fail_steps:
            peak = fail_steps[int(np.argmax([np.max(np.abs(vel_full[t])) for t in fail_steps]))]
        else:
            peak = -1

        return {"labels": labels, "confidences": confs, "dominant_failure": dom,
                "peak_step": peak, "failure_counts": counts, "n_unknown": 0}

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from annotation_model import RobotAnnotator
        return RobotAnnotator.load(), True      # (model, is_real)
    except Exception:
        return _QuickAnnotator(), False

@st.cache_data(show_spinner=False)
def run_inference(_model, fname, dof, steps, failures):
    eps  = load_episodes(fname, dof, steps, failures)
    rows = []
    for i, ep in enumerate(eps):
        seq = ep["seq"]
        r   = _model.annotate(seq)
        dom         = r["dominant_failure"]
        conf        = float(np.mean(r["confidences"]))
        peak        = r.get("peak_step", -1)
        step_labels = r["labels"]
        step_confs  = [float(c) for c in r["confidences"]]
        fail_frac   = sum(1 for l in step_labels if l != "nominal") / max(len(step_labels), 1)
        n_unknown   = int(r.get("n_unknown", 0))
        fcounts     = r.get("failure_counts", {dom: 1})

        use_for_policy = dom == "nominal" and fail_frac < 0.05 and conf >= 0.80
        needs_review   = conf < 0.80 or dom == "unknown_failure_type" or n_unknown > 0

        rows.append({
            "ep": f"ep_{i:03d}", "n_steps": len(seq),
            "failure_type": dom, "confidence": round(conf, 3),
            "peak_step": int(peak) if (peak is not None and peak >= 0) else -1,
            "fail_frac": round(fail_frac, 3),
            "use_for_policy": use_for_policy, "needs_review": needs_review,
            "seq": seq, "step_labels": step_labels,
            "step_confs": step_confs, "failure_counts": fcounts,
        })
    return rows

def metric_html(lbl, val, sub=""):
    return (f'<div class="metric-cell">'
            f'<div class="metric-lbl">{lbl}</div>'
            f'<div class="metric-val">{val}</div>'
            + (f'<div class="metric-sub">{sub}</div>' if sub else "")
            + '</div>')

# ── Session state ─────────────────────────────────────────────────────────────
if "step" not in st.session_state: st.session_state["step"] = 0
if "ds"   not in st.session_state: st.session_state["ds"]   = "xArm Push"

# ── Sidebar ───────────────────────────────────────────────────────────────────
model, model_real = load_model()

with st.sidebar:
    logo_path = STATIC_DIR / "haptal_dark.png"
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    else:
        st.markdown(f"<span style='font-size:1.4rem;font-weight:800;color:{T1};'>Haptal.</span>",
                    unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{T2};margin-bottom:10px;'>Dataset</div>", unsafe_allow_html=True)

    ds_name = st.radio("ds", list(DATASETS.keys()),
                       index=list(DATASETS.keys()).index(st.session_state["ds"]),
                       label_visibility="collapsed")
    if ds_name != st.session_state["ds"]:
        st.session_state["ds"] = ds_name
        st.session_state["step"] = 0
        st.rerun()

    meta = DATASETS[ds_name]
    st.markdown(
        f"<div style='font-size:.78rem;color:{T2};margin-top:8px;line-height:1.7;'>"
        f"{meta['robot']}<br>{meta['task']}<br>"
        f"<span style='color:#2e3340;'>{meta['note']}</span></div>",
        unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER};margin:0 0 16px;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{T2};margin-bottom:10px;'>Model</div>", unsafe_allow_html=True)

    if model_real:
        st.markdown(f"<div style='font-size:.82rem;color:{T1};font-weight:600;'>RobotAnnotator v1.1</div>"
                    f"<div style='font-size:.76rem;color:{T2};margin-top:4px;line-height:1.7;'>"
                    f"Calibrated Random Forest<br>Val accuracy: 89.9%<br>Brier score: 0.017<br>"
                    f"Trained: xArm · ALOHA · DROID</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-size:.82rem;color:{T1};font-weight:600;'>Physics annotator</div>"
                    f"<div style='font-size:.76rem;color:{T2};margin-top:4px;line-height:1.7;'>"
                    f"Step-level threshold detection<br>Velocity · Stuck joint · Deviation</div>",
                    unsafe_allow_html=True)

    st.markdown(f"<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.72rem;color:{T2};'>aarav@haptal.ai</div>", unsafe_allow_html=True)

# ── Run inference ─────────────────────────────────────────────────────────────
results = run_inference(model, meta["file"], meta["dof"], meta["steps"], meta["failures"])

df      = pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("seq","step_labels","step_confs","failure_counts")}
                         for r in results])
n_total = len(df)
n_pass  = int(df["use_for_policy"].sum())
n_fail  = int((~df["use_for_policy"] & ~df["needs_review"]).sum())
n_rev   = int(df["needs_review"].sum())

notable = max((r for r in results if r["failure_type"] != "nominal"),
              key=lambda r: r["fail_frac"], default=results[0])

# ── Step indicator ─────────────────────────────────────────────────────────────
step       = st.session_state["step"]
step_names = ["Raw data", "Annotation", "Output"]

dots_html = ""
for i, s in enumerate(step_names):
    dc = "active" if i == step else ("done" if i < step else "")
    lc = "active" if i == step else ""
    num = "&#10003;" if i < step else str(i + 1)
    dots_html += f'<div class="step-item"><div class="step-dot {dc}">{num}</div><div class="step-label {lc}">{s}</div></div>'
    if i < len(step_names) - 1:
        lnc = "done" if i < step else ""
        dots_html += f'<div class="step-line {lnc}"></div>'
st.markdown(f'<div class="steps">{dots_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
if step == 0:
    st.markdown("<h2 style='margin-bottom:.3rem;'>Raw training data</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{T2};font-size:.95rem;margin-bottom:2rem;'>"
        f"{ds_name} &nbsp;·&nbsp; {meta['robot']} &nbsp;·&nbsp; {meta['task']} &nbsp;·&nbsp; {n_total} episodes"
        f"</p>", unsafe_allow_html=True)

    raw_rows = []
    for r in results[:12]:
        seq = r["seq"]
        vel = np.diff(seq, axis=0)
        raw_rows.append({
            "episode":  r["ep"],
            "steps":    r["n_steps"],
            "joint_0":  round(float(seq[:, 0].mean()), 4),
            "joint_1":  round(float(seq[:, 1].mean()), 4),
            "vel_max":  round(float(np.abs(vel).max()), 4),
            "label":    "—",
            "status":   "—",
        })

    st.dataframe(pd.DataFrame(raw_rows), use_container_width=True,
                 hide_index=True, height=380)

    st.markdown(
        f'<div class="callout callout-red">'
        f'<div style="font-size:.85rem;font-weight:600;color:{T1};margin-bottom:4px;">No labels. No quality signal.</div>'
        f'<div style="font-size:.84rem;color:{T2};">An estimated 20–30% of these episodes contain failures — '
        f'slips, stalls, overcorrections. Without annotation, your policy trains on all of them.</div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    cols = st.columns([6, 1])
    with cols[1]:
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown("<h2 style='margin-bottom:.3rem;'>Annotation</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{T2};font-size:.95rem;margin-bottom:2rem;'>"
        f"{n_total} episodes &nbsp;·&nbsp; {n_pass} passed &nbsp;·&nbsp; "
        f"{n_fail} excluded &nbsp;·&nbsp; {n_rev} flagged for review</p>",
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="metric-strip">'
        f'{metric_html("Passed",       str(n_pass), f"{round(100*n_pass/n_total)}% of dataset")}'
        f'{metric_html("Excluded",     str(n_fail), f"{round(100*n_fail/n_total)}% failure rate")}'
        f'{metric_html("For review",   str(n_rev),  "confidence < 0.80")}'
        f'{metric_html("Mean conf",    "{:.3f}".format(df["confidence"].mean()))}'
        f'</div>', unsafe_allow_html=True)

    # Sensor trace — most notable failure
    seq_n  = notable["seq"]
    T_n    = len(seq_n)
    dims_n = min(seq_n.shape[1], 4)

    fig = go.Figure()
    for d in range(dims_n):
        fig.add_trace(go.Scatter(
            x=list(range(T_n)), y=seq_n[:, d].tolist(),
            mode="lines", name=f"joint {d}",
            line=dict(color=TRACE[d], width=1.8),
            hovertemplate=f"joint {d}: %{{y:.4f}}<extra></extra>",
        ))

    fail_steps = [t for t, l in enumerate(notable["step_labels"]) if l != "nominal"]
    if fail_steps:
        fig.add_vrect(x0=min(fail_steps)-.5, x1=max(fail_steps)+.5,
                      fillcolor=RED, opacity=.08, layer="below", line_width=0)
        cx = (min(fail_steps) + max(fail_steps)) / 2
        fig.add_annotation(
            x=cx, y=float(seq_n[:, 0].max()),
            text=f"{notable['failure_type'].replace('_',' ')}  ·  step {notable['peak_step']}  ·  conf {notable['confidence']:.3f}",
            showarrow=False,
            font=dict(color=RED, size=11, family="monospace"),
            bgcolor="rgba(239,68,68,.07)", borderpad=6,
        )

    fig.update_layout(
        height=250, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="timestep", gridcolor=BORDER, color=T2, linecolor=BORDER),
        yaxis=dict(title="joint state", gridcolor=BORDER, color=T2, linecolor=BORDER),
        legend=dict(orientation="h", y=1.12, font=dict(color=T2, size=11)),
    )

    st.markdown(f'<div class="sec-label">Detected failure &nbsp;—&nbsp; {notable["ep"]}</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">All episodes</div>', unsafe_allow_html=True)

    table_rows = []
    for r in results:
        s = "REVIEW" if r["needs_review"] else "PASS" if r["use_for_policy"] else "FAIL"
        table_rows.append({
            "episode":      r["ep"],
            "steps":        r["n_steps"],
            "failure type": r["failure_type"].replace("_", " "),
            "confidence":   r["confidence"],
            "peak step":    r["peak_step"] if r["peak_step"] >= 0 else "—",
            "status":       s,
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True, hide_index=True, height=280,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.3f"),
        },
    )

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    c1, _, c2 = st.columns([1, 5, 1])
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state["step"] = 0; st.rerun()
    with c2:
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state["step"] = 2; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown("<h2 style='margin-bottom:.3rem;'>Clean output</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{T2};font-size:.95rem;margin-bottom:2rem;'>"
        f"{n_pass} episodes ready for training &nbsp;·&nbsp; {n_fail + n_rev} removed or pending review"
        f"</p>", unsafe_allow_html=True)

    col_b, col_a = st.columns(2)

    with col_b:
        st.markdown('<div class="sec-label">Before</div>', unsafe_allow_html=True)
        before = [{"episode": r["ep"], "steps": r["n_steps"],
                   "label": "—", "status": "—"} for r in results]
        st.dataframe(pd.DataFrame(before), use_container_width=True,
                     hide_index=True, height=320)

    with col_a:
        st.markdown('<div class="sec-label">After Haptal</div>', unsafe_allow_html=True)
        after = []
        for r in results:
            s = "REVIEW" if r["needs_review"] else "PASS" if r["use_for_policy"] else "FAIL"
            after.append({
                "episode":      r["ep"],
                "steps":        r["n_steps"],
                "failure type": r["failure_type"].replace("_"," "),
                "confidence":   r["confidence"],
                "status":       s,
            })
        st.dataframe(
            pd.DataFrame(after), use_container_width=True, hide_index=True, height=320,
            column_config={"confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.3f")},
        )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Training script</div>', unsafe_allow_html=True)

    c_bef, c_aft = st.columns(2)
    with c_bef:
        st.code("train(dataset)", language="python")
    with c_aft:
        st.code('train(dataset.filter(lambda ep: ep["use_for_policy"]))', language="python")

    st.markdown(
        f'<div class="callout callout-teal">'
        f'<div style="font-size:.85rem;font-weight:600;color:{T1};margin-bottom:4px;">'
        f'93.6% in-distribution accuracy &nbsp;·&nbsp; 90.8% on unseen platforms &nbsp;·&nbsp; &kappa; = 0.66 vs human operators'
        f'</div>'
        f'<div style="font-size:.83rem;color:{T2};">'
        f'First public benchmark for robot training data annotation quality. '
        f'<a href="https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark" '
        f'style="color:{TEAL};text-decoration:none;">HaptalAI/robotics-failure-benchmark</a>'
        f'</div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    c_back, _ = st.columns([1, 6])
    with c_back:
        if st.button("Back", use_container_width=True):
            st.session_state["step"] = 1; st.rerun()
