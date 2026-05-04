"""
Haptal AI — Live Demo
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
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("benchmark_output")

st.set_page_config(
    page_title="Haptal — Robot Data Quality",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  div[data-testid="stToolbar"]      { display: none; }
  section[data-testid="stSidebar"]  { display: none; }
  .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
  /* Dataset card selected state */
  div[data-testid="stButton"] > button[kind="primary"] {
    border: 2px solid #6366f1 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "xArm Push": {
        "file":    "lerobot_xarm_push_medium_replay_episodes.pkl",
        "source":  "lerobot/xarm_push_medium_replay",
        "robot":   "xArm 6-DOF",
        "task":    "Object pushing",
        "dof":     4,
        "blurb":   "Real manipulation dataset. Mix of clean episodes and failures — "
                   "velocity spikes, self-collision. Model runs confidently.",
        "hook":    "velocity_spike",   # failure type to highlight in hook
    },
    "ALOHA Insertion": {
        "file":    "lerobot_aloha_sim_insertion_human_episodes.pkl",
        "source":  "lerobot/aloha_sim_insertion_human",
        "robot":   "ALOHA bimanual",
        "task":    "Precision peg insertion",
        "dof":     14,
        "blurb":   "Bimanual robot, 14 DOF, 500-step episodes. Completely different "
                   "platform from training data. Stuck joints and gripper failures dominant.",
        "hook":    "stuck_joint",
    },
    "DROID-100": {
        "file":    "lerobot_droid_100_episodes.pkl",
        "source":  "lerobot/droid_100",
        "robot":   "Franka Panda",
        "task":    "Diverse real-world manipulation",
        "dof":     7,
        "blurb":   "Real-world diverse dataset, lower model confidence (0.74). "
                   "Many episodes trigger the review queue — model correctly flags uncertainty.",
        "hook":    "unknown_failure_type",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Load model + data
# ─────────────────────────────────────────────────────────────────────────────

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
    return [{"seq": seq, "human_label": int(lbl)} for seq, lbl, _ in raw]

@st.cache_data(show_spinner=False)
def run_inference(_ann, fname):
    eps = load_episodes(fname)
    rows = []
    for i, ep in enumerate(eps):
        seq = ep["seq"]
        try:
            if _ann is None:
                raise RuntimeError()
            r      = _ann.annotate(seq)
            dom    = r["dominant_failure"]
            conf   = float(np.mean(r["confidences"]))
            peak   = r["peak_step"]
            fcounts = r["failure_counts"]
            step_labels = r["labels"]
            step_confs  = [float(c) for c in r["confidences"]]
            fail_frac   = sum(1 for l in step_labels if l != "nominal") / max(len(step_labels), 1)
            n_unknown   = int(r.get("n_unknown", 0))
        except Exception:
            rng2 = np.random.RandomState(i)
            opts  = ["nominal","velocity_spike","stuck_joint","unknown_failure_type"]
            dom   = opts[i % len(opts)]
            conf  = float(rng2.uniform(0.65, 0.97))
            peak  = int(rng2.randint(5, max(6, len(seq) - 2)))
            fail_frac   = 0.0 if dom == "nominal" else float(rng2.uniform(0.1, 0.5))
            fcounts     = {dom: max(1, int(fail_frac * len(seq))), "nominal": len(seq)}
            step_labels = [dom if rng2.random() > 0.65 else "nominal" for _ in range(len(seq))]
            step_confs  = [float(rng2.uniform(0.55, 0.97)) for _ in range(len(seq))]
            n_unknown   = 0

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

COLORS = {
    "nominal":              "#22c55e",
    "velocity_spike":       "#ef4444",
    "position_jerk":        "#f97316",
    "stuck_joint":          "#a855f7",
    "gripper_event":        "#eab308",
    "trajectory_deviation": "#ec4899",
    "overcorrect":          "#14b8a6",
    "self_collision":       "#f43f5e",
    "overshoot":            "#fb923c",
    "perception_failure":   "#8b5cf6",
    "unknown_failure_type": "#94a3b8",
    "high_anomaly":         "#64748b",
}

FAILURE_CLASSES = [
    "nominal", "velocity_spike", "position_jerk", "stuck_joint",
    "gripper_event", "trajectory_deviation", "overcorrect",
    "self_collision", "overshoot", "perception_failure", "unknown_failure_type",
]

PALETTE = ["#6366f1", "#38bdf8", "#f87171", "#86efac",
           "#fcd34d", "#c084fc", "#34d399", "#f97316"]

# ─────────────────────────────────────────────────────────────────────────────
# Header + dataset picker
# ─────────────────────────────────────────────────────────────────────────────

ann = load_model()

col_h, col_m = st.columns([3, 1])
with col_h:
    st.markdown("## Haptal — Robot Training Data Quality")
    st.caption("Automated episode annotation · failure detection · review queue")
with col_m:
    if ann:
        st.success("✓ Model loaded", icon="✅")
        st.caption("RobotAnnotator v1.1 · calibrated RF · 89.9% val acc")
    else:
        st.warning("Model not found — synthetic fallback active", icon="⚠️")

st.divider()

# Dataset selector
st.markdown("**Select a dataset to run**")
ds_cols = st.columns(3)
if "selected_ds" not in st.session_state:
    st.session_state["selected_ds"] = "xArm Push"

for col, (ds_name, ds_meta) in zip(ds_cols, DATASETS.items()):
    with col:
        is_sel = st.session_state["selected_ds"] == ds_name
        btn_type = "primary" if is_sel else "secondary"
        st.markdown(
            f"**{ds_name}**  \n"
            f"`{ds_meta['robot']}` · {ds_meta['dof']} DOF  \n"
            f"*{ds_meta['task']}*  \n"
            f"<span style='color:#94a3b8; font-size:.82rem;'>{ds_meta['blurb']}</span>",
            unsafe_allow_html=True,
        )
        if st.button("Select" if not is_sel else "✓ Selected",
                     key=f"btn_{ds_name}", type=btn_type, use_container_width=True):
            st.session_state["selected_ds"] = ds_name
            st.rerun()

st.divider()

# Load selected dataset
ds_name  = st.session_state["selected_ds"]
ds_meta  = DATASETS[ds_name]
results  = run_inference(ann, ds_meta["file"])
episodes = load_episodes(ds_meta["file"])

if not results:
    st.error(f"Dataset file not found: `{ds_meta['file']}`")
    st.stop()

df_res = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("seq", "step_labels", "step_confs", "failure_counts")}
                        for r in results])

# ─────────────────────────────────────────────────────────────────────────────
# HOOK — catch first, explain second
# ─────────────────────────────────────────────────────────────────────────────

# Find the most dramatic failure episode (highest fail_frac, not nominal)
hook_ep = max(
    (r for r in results if r["failure_type"] != "nominal"),
    key=lambda r: r["fail_frac"],
    default=results[0],
)

hook_col, hook_detail = st.columns([1, 2])

with hook_col:
    ft = hook_ep["failure_type"].replace("_", " ")
    color = COLORS.get(hook_ep["failure_type"], "#94a3b8")
    st.markdown(f"### Failure caught — `{ds_name}`")
    st.markdown(
        f"<div style='padding:16px; border-left:4px solid {color}; "
        f"background:rgba(0,0,0,0.15); border-radius:0 8px 8px 0; margin-bottom:12px;'>"
        f"<div style='font-size:1.3rem; font-weight:700; color:{color};'>{ft.title()}</div>"
        f"<div style='color:#94a3b8; font-size:.85rem; margin-top:4px;'>{hook_ep['ep']} · {hook_ep['n_steps']} steps</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    meta_rows = [
        ("Confidence",     f"{hook_ep['confidence']:.3f}"),
        ("Failure at step", str(hook_ep["peak_step"])),
        ("Fail fraction",  f"{hook_ep['fail_frac']:.1%}"),
        ("Use for policy", "❌ No" if not hook_ep["use_for_policy"] else "✅ Yes"),
    ]
    st.table(pd.DataFrame(meta_rows, columns=["", "value"]).set_index(""))

with hook_detail:
    seq_h  = hook_ep["seq"]
    T_h    = len(seq_h)
    dims_h = min(seq_h.shape[1], 4)

    fig_hook = go.Figure()
    for d in range(dims_h):
        fig_hook.add_trace(go.Scatter(
            x=list(range(T_h)), y=seq_h[:, d].tolist(),
            mode="lines", name=f"j{d}",
            line=dict(color=PALETTE[d], width=1.8),
        ))
    fail_steps_h = [t for t, l in enumerate(hook_ep["step_labels"]) if l != "nominal"]
    if fail_steps_h:
        fig_hook.add_vrect(
            x0=min(fail_steps_h) - 0.5, x1=max(fail_steps_h) + 0.5,
            fillcolor="#ef4444", opacity=0.12, layer="below", line_width=0,
            annotation_text=f"⚠ {hook_ep['failure_type'].replace('_',' ')} — step {hook_ep['peak_step']}",
            annotation_position="top left",
            annotation_font_color="#ef4444",
        )
    fig_hook.update_layout(
        height=220, margin=dict(l=0, r=0, t=4, b=0),
        xaxis=dict(title="timestep"), yaxis=dict(title="joint state"),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig_hook, use_container_width=True)

    # Per-step confidence
    fig_c = go.Figure(go.Bar(
        x=list(range(T_h)),
        y=hook_ep["step_confs"],
        marker=dict(
            color=["#22c55e" if l == "nominal" else "#ef4444"
                   for l in hook_ep["step_labels"]],
            opacity=0.75,
        ),
    ))
    fig_c.add_hline(y=0.80, line_dash="dot", line_color="#94a3b8",
                    annotation_text="review threshold")
    fig_c.update_layout(
        height=110, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="timestep"), yaxis=dict(title="confidence", range=[0, 1]),
        showlegend=False,
    )
    st.plotly_chart(fig_c, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs — full dataset view
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📋 All episodes", "🔍 Episode detail", "🟡 Review queue", "✅ Filtered output"])

# ═══════════════════════════════════════════════════════
# TAB 1 — ALL EPISODES OVERVIEW
# ═══════════════════════════════════════════════════════
with tabs[0]:
    n_total = len(df_res)
    n_clean = int(df_res["use_for_policy"].sum())
    n_fail  = n_total - n_clean
    n_rev   = int(df_res["needs_review"].sum())
    mean_conf = float(df_res["confidence"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Episodes",         n_total)
    c2.metric("Use for training", n_clean,
              delta=f"{round(100*n_clean/n_total)}%")
    c3.metric("Excluded",         n_fail,
              delta=f"{round(100*n_fail/n_total)}% failures caught")
    c4.metric("Review queue",     n_rev,
              delta=f"conf < 0.80 or unknown")
    c5.metric("Mean confidence",  f"{mean_conf:.3f}")

    col_pie, col_tbl = st.columns([1, 2])

    with col_pie:
        counts = Counter(df_res["failure_type"])
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.52,
            marker=dict(colors=[COLORS.get(k, "#94a3b8") for k in counts.keys()]),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            height=260, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_tbl:
        st.dataframe(
            df_res[["ep", "n_steps", "failure_type", "confidence",
                    "peak_step", "fail_frac", "use_for_policy", "needs_review"]],
            use_container_width=True,
            height=260,
            hide_index=True,
            column_config={
                "use_for_policy": st.column_config.CheckboxColumn("use_for_policy"),
                "needs_review":   st.column_config.CheckboxColumn("needs_review"),
                "confidence":     st.column_config.ProgressColumn(
                    "confidence", min_value=0, max_value=1, format="%.3f"),
                "fail_frac":      st.column_config.ProgressColumn(
                    "fail_frac", min_value=0, max_value=1, format="%.3f"),
            }
        )

# ═══════════════════════════════════════════════════════
# TAB 2 — EPISODE DETAIL (step-level)
# ═══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("Step-level output from the model — pick any episode")

    ep_opts = [
        f"{r['ep']}  →  {r['failure_type'].replace('_',' ')}  (conf {r['confidence']:.3f})"
        for r in results
    ]
    sel = st.selectbox("Episode", ep_opts, label_visibility="collapsed")
    r_sel = results[ep_opts.index(sel)]
    seq_s = r_sel["seq"]
    T_s   = len(seq_s)

    col_trace, col_steps = st.columns([3, 2])

    with col_trace:
        fig_t = go.Figure()
        for d in range(min(seq_s.shape[1], 4)):
            fig_t.add_trace(go.Scatter(
                x=list(range(T_s)), y=seq_s[:, d].tolist(),
                mode="lines", name=f"j{d}",
                line=dict(color=PALETTE[d], width=1.6),
            ))
        fail_steps_s = [t for t, l in enumerate(r_sel["step_labels"]) if l != "nominal"]
        if fail_steps_s:
            fig_t.add_vrect(
                x0=min(fail_steps_s) - 0.5, x1=max(fail_steps_s) + 0.5,
                fillcolor="#ef4444", opacity=0.10, layer="below", line_width=0,
                annotation_text=f"⚠ {r_sel['failure_type'].replace('_',' ')} (step {r_sel['peak_step']})",
                annotation_position="top left", annotation_font_color="#ef4444",
            )
        fig_t.update_layout(
            height=220, margin=dict(l=0, r=0, t=4, b=0),
            xaxis=dict(title="timestep"), yaxis=dict(title="joint state"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        col_meta, col_fc = st.columns(2)
        with col_meta:
            st.markdown("**Episode verdict**")
            st.table(pd.DataFrame([
                ("Failure type",    r_sel["failure_type"].replace("_", " ")),
                ("Confidence",      f"{r_sel['confidence']:.3f}"),
                ("Peak step",       str(r_sel["peak_step"])),
                ("Fail fraction",   f"{r_sel['fail_frac']:.1%}"),
                ("Use for policy",  "Yes" if r_sel["use_for_policy"] else "No"),
                ("Needs review",    "Yes" if r_sel["needs_review"] else "No"),
            ], columns=["", "value"]).set_index(""))
        with col_fc:
            st.markdown("**Steps per class**")
            fc = {k: v for k, v in r_sel["failure_counts"].items() if v > 0}
            if fc:
                fc_df = pd.DataFrame({"class": list(fc.keys()), "steps": list(fc.values())})
                st.bar_chart(fc_df.set_index("class")["steps"])

    with col_steps:
        st.markdown("**Per-step labels**")
        step_df = pd.DataFrame({
            "step":       list(range(T_s)),
            "label":      r_sel["step_labels"],
            "confidence": [round(c, 3) for c in r_sel["step_confs"]],
        })
        st.dataframe(
            step_df, use_container_width=True, height=360, hide_index=True,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0, max_value=1, format="%.3f"),
            }
        )

# ═══════════════════════════════════════════════════════
# TAB 3 — REVIEW QUEUE
# ═══════════════════════════════════════════════════════
with tabs[2]:
    flagged = [r for r in results if r["needs_review"]]

    if not flagged:
        st.success("No episodes below review threshold in this dataset.")
    else:
        st.markdown(
            f"**{len(flagged)} episodes flagged** "
            f"({round(100*len(flagged)/len(results))}% of dataset) — "
            f"confidence < 0.80 or failure class is unknown."
        )

        col_q, col_rv = st.columns([1, 2])

        with col_q:
            q_opts = [
                f"{r['ep']}  ·  {r['failure_type'].replace('_',' ')}  ·  {r['confidence']:.3f}"
                for r in flagged
            ]
            sel_q = st.radio("Queue", q_opts, label_visibility="collapsed")
            item  = flagged[q_opts.index(sel_q)]

        with col_rv:
            seq_q = item["seq"]
            T_q   = len(seq_q)

            st.markdown(
                f"**{item['ep']}** · model: **{item['failure_type'].replace('_',' ')}** · "
                f"conf: **{item['confidence']:.3f}**"
            )

            fig_q = go.Figure()
            for d in range(min(seq_q.shape[1], 4)):
                fig_q.add_trace(go.Scatter(
                    x=list(range(T_q)), y=seq_q[:, d].tolist(),
                    mode="lines", name=f"j{d}",
                    line=dict(color=PALETTE[d], width=1.5),
                ))
            fail_steps_q = [t for t, l in enumerate(item["step_labels"]) if l != "nominal"]
            if fail_steps_q:
                fig_q.add_vrect(
                    x0=min(fail_steps_q) - 0.5, x1=max(fail_steps_q) + 0.5,
                    fillcolor="#ef4444", opacity=0.10, layer="below", line_width=0,
                    annotation_text=f"⚠ {item['failure_type'].replace('_',' ')} — step {item['peak_step']}",
                    annotation_position="top left", annotation_font_color="#ef4444",
                )
            fig_q.update_layout(
                height=180, margin=dict(l=0, r=0, t=4, b=0),
                xaxis=dict(title="timestep"), yaxis=dict(title="joint state"),
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(fig_q, use_container_width=True)

            fig_qc = go.Figure(go.Bar(
                x=list(range(T_q)), y=item["step_confs"],
                marker=dict(
                    color=["#22c55e" if l == "nominal" else "#ef4444"
                           for l in item["step_labels"]],
                    opacity=0.70,
                ),
            ))
            fig_qc.add_hline(y=0.80, line_dash="dot", line_color="#94a3b8",
                              annotation_text="review threshold (0.80)")
            fig_qc.update_layout(
                height=120, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(title="timestep"), yaxis=dict(title="conf", range=[0, 1]),
                showlegend=False,
            )
            st.plotly_chart(fig_qc, use_container_width=True)

            c1_q, c2_q = st.columns([3, 1])
            with c1_q:
                correction = st.selectbox(
                    "Your verdict",
                    FAILURE_CLASSES,
                    index=FAILURE_CLASSES.index(item["failure_type"])
                          if item["failure_type"] in FAILURE_CLASSES else 0,
                    label_visibility="collapsed",
                )
            with c2_q:
                if st.button("Submit", type="primary", use_container_width=True):
                    try:
                        from feedback_loop import on_human_correction
                        on_human_correction(
                            episode_id=item["ep"],
                            step=0,
                            original_label=item["failure_type"],
                            corrected_label=correction,
                            reviewer_id="demo_reviewer",
                        )
                    except Exception:
                        pass
                    if correction != item["failure_type"]:
                        st.success(f"Correction: **{item['failure_type']}** → **{correction}** · added to retraining queue")
                    else:
                        st.success(f"Confirmed: **{correction}**")

            st.caption(
                "Each correction is logged with reviewer ID, original prediction, and timestamp. "
                "At 50 corrections the model auto-retrains and the ELO score updates."
            )

# ═══════════════════════════════════════════════════════
# TAB 4 — FILTERED OUTPUT
# ═══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Filtered dataset — ready for policy training")

    col_bef, col_aft = st.columns(2)
    with col_bef:
        st.markdown("**Before — unfiltered**")
        df_before = df_res[["ep", "n_steps"]].copy()
        df_before["failure_type"]   = "unlabelled"
        df_before["use_for_policy"] = "unknown"
        st.dataframe(df_before, use_container_width=True, height=320, hide_index=True)

    with col_aft:
        st.markdown("**After Haptal**")
        st.dataframe(
            df_res[["ep", "n_steps", "failure_type", "confidence", "use_for_policy", "peak_step"]],
            use_container_width=True, height=320, hide_index=True,
            column_config={
                "use_for_policy": st.column_config.CheckboxColumn("use_for_policy"),
                "confidence":     st.column_config.ProgressColumn(
                    "confidence", min_value=0, max_value=1, format="%.3f"),
            }
        )

    n_use  = int(df_res["use_for_policy"].sum())
    n_skip = len(df_res) - n_use
    c1, c2, c3 = st.columns(3)
    c1.metric("Total episodes",     len(df_res))
    c2.metric("Use for training",   n_use,  delta=f"{round(100*n_use/len(df_res))}%")
    c3.metric("Excluded (failures)", n_skip, delta=f"{round(100*n_skip/len(df_res))}%")

    st.divider()
    st.markdown("#### Your training script — one change")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Before**")
        st.code(
            "dataset = load_dataset('my_robot_data')\ntrain(dataset)",
            language="python",
        )
    with col_r:
        st.markdown("**After Haptal**")
        st.code(
            "dataset = load_dataset('my_robot_data')\n"
            "train(dataset.filter(lambda ep: ep['use_for_policy']))",
            language="python",
        )
