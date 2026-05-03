"""
Robotics Anomaly Detection — Client Benchmark Dashboard
Run: streamlit run dashboard.py

Two pages:
  1. Benchmark Results  — episode-level pass/fail, metrics, score distributions
  2. 3D Annotations     — step-level trajectory viewer with failure type labels
"""

import json, h5py, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Robotics Anomaly Detection", page_icon="🤖", layout="wide")

PAGES = ["📊 Benchmark Results", "🧭 3D Annotations"]
page  = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#6b7280'>Model: IsolationForest v0.1<br>Data: LeRobot / HuggingFace</small>",
    unsafe_allow_html=True)

OUTPUT_DIR = Path("benchmark_output")

FAILURE_COLORS = {
    "nominal":            "#3b82f6",
    "velocity_spike":     "#ef4444",
    "position_jerk":      "#f97316",
    "stuck_joint":        "#a855f7",
    "gripper_event":      "#eab308",
    "workspace_boundary": "#ec4899",
}
FAILURE_LABELS = {
    "nominal":            "Normal operation",
    "velocity_spike":     "Velocity spike — collision / slip",
    "position_jerk":      "Position jerk — abrupt direction change",
    "stuck_joint":        "Stuck joint — stall or grasp failure",
    "gripper_event":      "Gripper event — unexpected state change",
    "workspace_boundary": "Workspace boundary violation",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_benchmark_results():
    results = []
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
        results.append({"card": card, "scores": scores, "labels": labels,
                         "preds": preds, "name": card.get("dataset", card_path.stem)})
    return results

@st.cache_data
def load_annotations():
    anns = []
    for p in sorted(OUTPUT_DIR.glob("*_annotations.json")):
        data = json.loads(p.read_text())
        anns.append(data)
    return anns


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BENCHMARK RESULTS
# ════════════════════════════════════════════════════════════════════════════

if page == "📊 Benchmark Results":

    results = load_benchmark_results()
    if not results:
        st.error("No benchmark results found. Run: python main.py --source lerobot --dataset lerobot/xarm_lift_medium_replay")
        st.stop()

    st.markdown("""
    <div style='background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
                padding:2rem 2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;'>
      <h1 style='color:white;margin:0;font-size:2rem;'>🤖 Robotics Anomaly Detection</h1>
      <p style='color:#aad4f5;margin:0.3rem 0 0;font-size:1.05rem;'>
        Failure detection benchmarks on real robot manipulation data
      </p>
    </div>""", unsafe_allow_html=True)

    dataset_names = [r["name"] for r in results]
    selected      = st.selectbox("Select dataset", dataset_names)
    data          = next(r for r in results if r["name"] == selected)
    card, scores, labels, preds = data["card"], data["scores"], data["labels"], data["preds"]

    st.markdown("---")
    cm = card["confusion_matrix"]
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total, failures = card["total_episodes"], card["failure_episodes"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{card['roc_auc']:.3f}")
    c2.metric("Detection Rate",      f"{card['detection_rate_pct']}%",   f"{tp} of {failures} caught")
    c3.metric("False Positive Rate", f"{card['false_positive_rate_pct']}%", f"{fp} false alarms", delta_color="inverse")
    c4.metric("Total Episodes", f"{total:,}")
    c5.metric("Failure Rate",  f"{failures/total*100:.1f}%")

    st.markdown("---")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Anomaly Score Distribution")
        st.caption("Higher score = more anomalous. Episodes above the threshold are flagged as failures.")
        thresh = np.quantile(scores, card.get("confidence_threshold_quantile",
                                               card.get("threshold_quantile", 0.75)))
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores[labels==0], name="Nominal",
                                    marker_color="#3b82f6", opacity=0.7, xbins=dict(size=0.01)))
        fig.add_trace(go.Histogram(x=scores[labels==1], name="Failure",
                                    marker_color="#ef4444", opacity=0.7, xbins=dict(size=0.01)))
        fig.add_vline(x=thresh, line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold {thresh:.3f}", annotation_position="top right")
        fig.update_layout(barmode="overlay", xaxis_title="Anomaly Score", yaxis_title="Count",
                           height=340, margin=dict(t=40,b=40),
                           plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Confusion Matrix")
        st.caption("Predicted vs. ground truth")
        fig2 = go.Figure(go.Heatmap(
            z=[[tn, fp],[fn, tp]],
            x=["Predicted OK","Predicted FAIL"], y=["True OK","True FAIL"],
            text=[[f"TN\n{tn}",f"FP\n{fp}"],[f"FN\n{fn}",f"TP\n{tp}"]],
            texttemplate="%{text}", colorscale=[[0,"#1e3a5f"],[1,"#3b82f6"]], showscale=False))
        fig2.update_layout(height=340, margin=dict(t=40,b=40),
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Anomaly Scores — All Episodes")
    st.caption("Blue = nominal · Red = caught failure · Orange X = missed failure · Purple = false alarm")
    fig3 = go.Figure()
    nom_i  = np.where(labels==0)[0]
    tp_i   = np.where((labels==1)&(preds==1))[0]
    fn_i   = np.where((labels==1)&(preds==0))[0]
    fp_i   = np.where((labels==0)&(preds==1))[0]
    fig3.add_trace(go.Scatter(x=nom_i, y=scores[nom_i], mode="markers",
                               marker=dict(color="#3b82f6",size=5,opacity=0.6), name="Nominal"))
    fig3.add_trace(go.Scatter(x=tp_i,  y=scores[tp_i],  mode="markers",
                               marker=dict(color="#ef4444",size=7), name="Failure — caught ✓"))
    fig3.add_trace(go.Scatter(x=fn_i,  y=scores[fn_i],  mode="markers",
                               marker=dict(color="#f97316",size=9,symbol="x"), name="Failure — missed ✗"))
    fig3.add_trace(go.Scatter(x=fp_i,  y=scores[fp_i],  mode="markers",
                               marker=dict(color="#a855f7",size=7,symbol="diamond"), name="False alarm"))
    fig3.add_hline(y=thresh, line_dash="dash", line_color="orange")
    fig3.update_layout(xaxis_title="Episode Index", yaxis_title="Anomaly Score", height=380,
                        margin=dict(t=20,b=40), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font_color="white", legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Per-Episode Breakdown")
    tab1, tab2, tab3 = st.tabs(["❌ Misclassified", "🔴 All Failures", "📋 All Episodes"])

    def build_df(mask):
        idx = np.where(mask)[0]
        return pd.DataFrame({
            "Episode":       idx,
            "Anomaly Score": scores[idx].round(4),
            "True Label":    ["FAILURE" if labels[i] else "OK" for i in idx],
            "Predicted":     ["FAILURE" if preds[i]  else "OK" for i in idx],
            "Correct":       ["✓" if preds[i]==labels[i] else "✗" for i in idx],
        })

    with tab1:
        df = build_df(preds != labels)
        st.dataframe(df, use_container_width=True, height=320)
        st.caption(f"{(preds!=labels).sum()} misclassified of {len(labels)}")
    with tab2:
        st.dataframe(build_df(labels==1), use_container_width=True, height=320)
    with tab3:
        st.dataframe(build_df(np.ones(len(labels), dtype=bool)), use_container_width=True, height=320)

    if len(results) > 1:
        st.markdown("---")
        st.subheader("Cross-Dataset Comparison")
        summary = [{"Dataset": r["card"].get("dataset", r["name"]),
                    "Episodes": r["card"]["total_episodes"],
                    "Failure Rate": f"{r['card']['failure_episodes']/r['card']['total_episodes']*100:.1f}%",
                    "ROC-AUC": r["card"]["roc_auc"],
                    "Detection Rate": f"{r['card']['detection_rate_pct']}%",
                    "False Positive Rate": f"{r['card']['false_positive_rate_pct']}%"} for r in results]
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        fig4 = px.bar(pd.DataFrame(summary), x="Dataset", y="ROC-AUC",
                       color="ROC-AUC", color_continuous_scale="Blues",
                       text="ROC-AUC", title="ROC-AUC by Dataset")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="white", coloraxis_showscale=False, yaxis_range=[0,1.1])
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("<p style='color:#6b7280;font-size:0.85rem;text-align:center;'>"
                "Model: IsolationForest v0.1 · Data: LeRobot / HuggingFace · Built with Streamlit + Plotly"
                "</p>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — 3D ANNOTATIONS
# ════════════════════════════════════════════════════════════════════════════

elif page == "🧭 3D Annotations":

    st.markdown("""
    <div style='background:linear-gradient(90deg,#1a0533,#2d1b69,#1e3a5f);
                padding:2rem 2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;'>
      <h1 style='color:white;margin:0;font-size:2rem;'>🧭 3D Trajectory Annotations</h1>
      <p style='color:#c4b5fd;margin:0.3rem 0 0;font-size:1.05rem;'>
        Step-level failure labeling on real robot trajectories
      </p>
    </div>""", unsafe_allow_html=True)

    ann_data = load_annotations()
    if not ann_data:
        st.warning("No annotation data found.")
        st.code("python annotate.py --dataset lerobot/xarm_lift_medium_replay --max-episodes 100")
        st.stop()

    # dataset selector
    ds_names = [a["dataset"] for a in ann_data]
    sel_ds   = st.selectbox("Select dataset", ds_names)
    ann      = next(a for a in ann_data if a["dataset"] == sel_ds)
    episodes = ann["annotations"]

    st.markdown("---")

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    total_steps  = sum(e["n_steps"] for e in episodes)
    all_types    = [ft for e in episodes for ft in e["failure_types"]]
    type_counts  = {k: all_types.count(k) for k in FAILURE_COLORS}
    anomalous    = sum(1 for t in all_types if t != "nominal")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Episodes annotated", len(episodes))
    k2.metric("Total timesteps",    f"{total_steps:,}")
    k3.metric("Anomalous steps",    f"{anomalous:,}", f"{anomalous/total_steps*100:.1f}% of all steps")
    k4.metric("Failure episodes",   ann["n_failures"])

    st.markdown("---")

    # ── Failure type breakdown ────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Failure Type Distribution")
        st.caption("Across all annotated episodes")
        df_types = pd.DataFrame([
            {"Type": FAILURE_LABELS[k], "Steps": v, "key": k}
            for k, v in type_counts.items() if v > 0
        ])
        fig_bar = px.bar(df_types, x="Steps", y="Type", orientation="h",
                          color="Type",
                          color_discrete_map={FAILURE_LABELS[k]: FAILURE_COLORS[k]
                                              for k in FAILURE_COLORS},
                          text="Steps")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=340, showlegend=False,
                               plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font_color="white", margin=dict(t=20,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.subheader("Dominant Failure by Episode")
        st.caption("Most frequent failure type per episode")
        dom = pd.DataFrame([
            {"Episode": e["episode_id"], "Dominant Failure": FAILURE_LABELS[e["dominant_failure"]],
             "True Label": e["label_str"], "Peak Score": round(e["peak_score"], 4),
             "Peak Step": e["peak_step"]}
            for e in episodes
        ])
        fail_counts = dom["Dominant Failure"].value_counts().reset_index()
        fail_counts.columns = ["Failure Type", "Count"]
        fig_pie = px.pie(fail_counts, values="Count", names="Failure Type",
                          color="Failure Type",
                          color_discrete_map={FAILURE_LABELS[k]: FAILURE_COLORS[k]
                                              for k in FAILURE_COLORS},
                          hole=0.4)
        fig_pie.update_layout(height=340, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font_color="white", margin=dict(t=20,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Episode 3D trajectory viewer ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("3D Trajectory Viewer")
    st.caption("Select an episode to see its full trajectory colored by failure type. "
               "Each point is one timestep. Hover for anomaly score.")

    ep_options = {f"Episode {e['episode_id']} — {e['label_str']} (peak={e['peak_score']:.3f})": e
                  for e in episodes}
    sel_ep_key = st.selectbox("Select episode", list(ep_options.keys()))
    ep         = ep_options[sel_ep_key]

    coords  = np.array(ep["coords_3d"])
    scores_ = np.array(ep["anomaly_scores"])
    ftypes  = ep["failure_types"]
    T       = len(coords)

    view_col, info_col = st.columns([3, 1])

    with view_col:
        # color by failure type
        mode = st.radio("Color by", ["Failure Type", "Anomaly Score"], horizontal=True)

        fig3d = go.Figure()
        if mode == "Failure Type":
            for ftype, color in FAILURE_COLORS.items():
                idx = [i for i, f in enumerate(ftypes) if f == ftype]
                if not idx:
                    continue
                fig3d.add_trace(go.Scatter3d(
                    x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.8),
                    name=FAILURE_LABELS[ftype],
                    text=[f"Step {i}<br>Score: {scores_[i]:.4f}<br>{FAILURE_LABELS[ftypes[i]]}"
                          for i in idx],
                    hoverinfo="text",
                ))
        else:
            fig3d.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers",
                marker=dict(size=4, color=scores_, colorscale="RdBu_r",
                             showscale=True, colorbar=dict(title="Anomaly Score")),
                text=[f"Step {i}<br>Score: {scores_[i]:.4f}<br>{FAILURE_LABELS[ftypes[i]]}"
                      for i in range(T)],
                hoverinfo="text",
                name="Trajectory",
            ))

        # draw trajectory line
        fig3d.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

        # mark peak anomaly step
        peak = ep["peak_step"]
        fig3d.add_trace(go.Scatter3d(
            x=[coords[peak, 0]], y=[coords[peak, 1]], z=[coords[peak, 2]],
            mode="markers+text",
            marker=dict(size=10, color="white", symbol="diamond"),
            text=[f"Peak anomaly (step {peak})"],
            textposition="top center",
            name="Peak anomaly",
        ))

        fig3d.update_layout(
            height=520,
            scene=dict(
                bgcolor="#0e1117",
                xaxis=dict(title="PC1", backgroundcolor="#0e1117",
                            gridcolor="#1f2937", zerolinecolor="#374151"),
                yaxis=dict(title="PC2", backgroundcolor="#0e1117",
                            gridcolor="#1f2937", zerolinecolor="#374151"),
                zaxis=dict(title="PC3", backgroundcolor="#0e1117",
                            gridcolor="#1f2937", zerolinecolor="#374151"),
            ),
            paper_bgcolor="#0e1117",
            font_color="white",
            margin=dict(t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with info_col:
        st.markdown("**Episode summary**")
        st.markdown(f"- **Label**: `{ep['label_str']}`")
        st.markdown(f"- **Steps**: {ep['n_steps']}")
        st.markdown(f"- **Peak score**: `{ep['peak_score']:.4f}`")
        st.markdown(f"- **Peak at step**: `{ep['peak_step']}`")
        st.markdown(f"- **Dominant failure**: `{FAILURE_LABELS[ep['dominant_failure']]}`")
        st.markdown("---")
        st.markdown("**Step counts by type**")
        for k, v in ep["failure_counts"].items():
            if v > 0:
                pct = v / ep["n_steps"] * 100
                color = FAILURE_COLORS[k]
                st.markdown(
                    f"<span style='color:{color}'>■</span> **{k.replace('_',' ')}**: {v} ({pct:.0f}%)",
                    unsafe_allow_html=True)

    # ── Step-level anomaly score timeline ─────────────────────────────────────
    st.subheader("Anomaly Score Over Time")
    st.caption("Step-by-step anomaly score. Colored segments show failure type.")

    fig_time = go.Figure()
    for ftype, color in FAILURE_COLORS.items():
        idx = [i for i, f in enumerate(ftypes) if f == ftype]
        if not idx:
            continue
        fig_time.add_trace(go.Scatter(
            x=idx, y=[scores_[i] for i in idx],
            mode="markers", marker=dict(color=color, size=6, opacity=0.8),
            name=FAILURE_LABELS[ftype],
        ))
    fig_time.add_trace(go.Scatter(
        x=list(range(T)), y=scores_.tolist(),
        mode="lines", line=dict(color="rgba(255,255,255,0.2)", width=1),
        showlegend=False,
    ))
    fig_time.update_layout(
        xaxis_title="Timestep", yaxis_title="Anomaly Score",
        height=300, margin=dict(t=10, b=40),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # ── Episode table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("All Episodes — Annotation Summary")
    st.dataframe(dom, use_container_width=True, height=340)
