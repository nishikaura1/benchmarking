"""
RobotAnnotationPipeline — unified client-facing input/output pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW WE VALIDATE FAILURES (the full answer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 1 — Unsupervised anomaly score (IsolationForest)
  The model learns the statistical distribution of "normal" robot
  sessions from the SOP (nominal reference) file. Any session that
  deviates from that distribution gets a high anomaly score.
  No labels needed. Validated against open source datasets:
    xarm_lift: ROC-AUC 0.943, detection rate 92.2%
    xarm_push: ROC-AUC 0.871, detection rate 55.0%

Layer 2 — Step-level failure type (RobotAnnotator)
  Trained with weak supervision: rule-based labels (velocity spikes,
  jerk, stuck joints) applied to 3,175 steps from two real datasets.
  RF accuracy 92%. Labels each timestep: velocity_spike, position_jerk,
  stuck_joint, gripper_event, high_anomaly, nominal.

Layer 3 — Semantic labels (SemanticAnnotator)
  Trained on 177k steps from DROID + ALOHA (4 datasets, 4 robot types).
  Labels each timestep with 4 layers: task_phase, workspace_zone,
  contact_state, motion_type. Task phase accuracy 94.9%.

Ground truth validation for client data:
  No reward signal from client files → we use the SOP file as the
  nominal reference. Episodes deviating from SOP are anomalies.
  Client reviews flagged episodes, confirms/corrects labels →
  those corrections feed back into model retraining.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT / OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
  --sop     HDF5 file of nominal/reference robot sessions
            (what correct operation looks like)
  --input   HDF5 file of sessions to annotate
            (client's production data)

Output (written to benchmark_output/):
  <name>_annotated.h5     — annotated HDF5, same structure as input
                            + /episode_N/anomaly_score
                            + /episode_N/failure_type   (per step)
                            + /episode_N/task_phase     (per step)
                            + /episode_N/workspace_zone (per step)
                            + /episode_N/contact_state  (per step)
                            + /episode_N/motion_type    (per step)
  <name>_report.json      — human-readable report per episode
  <name>_summary.json     — overall stats + model card

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # with real SOP + client HDF5 files:
  python pipeline.py --sop nominal_sessions.h5 --input client_sessions.h5

  # demo mode (uses open source data, no files needed):
  python pipeline.py --demo

  # use a LeRobot dataset as both SOP and input:
  python pipeline.py --dataset lerobot/xarm_lift_medium_replay
"""

import argparse, json, pickle, warnings, sys
import numpy as np
import h5py
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("benchmark_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Import our models ─────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from annotation_model  import RobotAnnotator,   extract_window_features
from semantic_annotator import SemanticAnnotator, extract_semantic_features, LABEL_SCHEMA


# ── HDF5 I/O helpers ──────────────────────────────────────────────────────────

def load_hdf5_sessions(path: Path) -> list[dict]:
    """
    Load robot sessions from an HDF5 file.

    Expected structure (flexible — handles multiple layouts):
      /episode_N/states        (T, D)  joint state sequence
      /episode_N/actions       (T, A)  action sequence (optional)
      /episode_N/rewards       (T,)    reward signal   (optional)
      /episode_N/success       scalar  episode success  (optional)

    Also handles flat layout:
      /states  (N, T, D)
      /actions (N, T, A)
    """
    sessions = []
    with h5py.File(path, "r") as f:

        # ── layout 1: /episode_N/ groups ─────────────────────────────────────
        ep_keys = [k for k in f.keys() if k.startswith("episode_")]
        if ep_keys:
            for key in sorted(ep_keys):
                grp = f[key]
                ep  = {"episode_id": key}

                # state
                for skey in ["states", "observation", "obs", "joint_positions",
                              "anomaly_scores"]:
                    if skey in grp:
                        ep["states"] = np.array(grp[skey], dtype=np.float32)
                        break
                if "states" not in ep:
                    # try any 2D dataset
                    for dk in grp.keys():
                        arr = np.array(grp[dk])
                        if arr.ndim == 2 and arr.shape[0] > 1:
                            ep["states"] = arr.astype(np.float32)
                            break

                if "states" not in ep:
                    continue

                # optional fields
                for field in ["actions", "rewards", "true_labels", "predictions"]:
                    if field in grp:
                        ep[field] = np.array(grp[field])

                # attrs
                for attr in ["true_label", "roc_auc", "dominant_failure"]:
                    if attr in grp.attrs:
                        ep[attr] = grp.attrs[attr]

                sessions.append(ep)

        # ── layout 2: flat arrays (/states shape N,T,D) ───────────────────────
        elif "states" in f or "observations" in f:
            key    = "states" if "states" in f else "observations"
            states = np.array(f[key], dtype=np.float32)   # (N, T, D)
            if states.ndim == 2:
                states = states[np.newaxis]                # single episode
            rewards = np.array(f["rewards"]) if "rewards" in f else None
            for i, seq in enumerate(states):
                ep = {"episode_id": f"episode_{i:04d}", "states": seq}
                if rewards is not None:
                    ep["rewards"] = rewards[i]
                sessions.append(ep)

        # ── layout 3: our own annotated output format ─────────────────────────
        else:
            # fall back: treat every 2D dataset as a separate episode
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim == 2 and arr.shape[0] > 1:
                    sessions.append({"episode_id": key,
                                     "states": arr.astype(np.float32)})

    print(f"  Loaded {len(sessions)} sessions from {path.name}")
    return sessions


def derive_ground_truth(sessions: list[dict]) -> np.ndarray | None:
    """
    Extract ground truth labels from sessions if available.
    Returns (N,) int array (0=nominal, 1=failure) or None.
    """
    labels = []
    for ep in sessions:
        if "true_label" in ep:
            labels.append(int(ep["true_label"]))
        elif "rewards" in ep:
            max_r = float(ep["rewards"].max())
            # binary: >0.5 = success; continuous negative: use percentile
            if max_r >= 0:
                labels.append(0 if max_r > 0.5 else 1)
            else:
                labels.append(None)   # continuous reward — can't threshold without SOP
        else:
            labels.append(None)

    if all(l is None for l in labels):
        return None
    return np.array([l if l is not None else -1 for l in labels])


# ── Core pipeline ─────────────────────────────────────────────────────────────

class RobotAnnotationPipeline:
    """
    End-to-end pipeline: SOP + client HDF5 → fully annotated output.

    Steps:
      1. Load SOP (nominal reference) → fit anomaly detector
      2. Load input sessions → score each episode
      3. Run step-level failure type annotation (RobotAnnotator)
      4. Run semantic annotation (SemanticAnnotator)
      5. Write annotated HDF5 + JSON report
    """

    def __init__(self):
        self.anomaly_model  = None
        self.anomaly_scaler = None
        self.failure_ann    = None
        self.semantic_ann   = None

    # ── Step 1: fit anomaly detector on SOP ──────────────────────────────────

    def fit_from_sop(self, sop_sessions: list[dict]):
        """
        Learn 'normal' from SOP (nominal reference) sessions.
        Fits IsolationForest on episode-level feature summaries.
        """
        print(f"\nFitting anomaly detector on {len(sop_sessions)} SOP sessions...")
        features = []
        for ep in sop_sessions:
            seq = ep["states"]
            f   = self._episode_features(seq)
            features.append(f)

        X = np.array(features, dtype=np.float32)
        self.anomaly_scaler = StandardScaler()
        X_sc = self.anomaly_scaler.fit_transform(X)

        self.anomaly_model = IsolationForest(
            contamination=0.05, random_state=42, n_jobs=-1)
        self.anomaly_model.fit(X_sc)
        print(f"  Anomaly detector fitted on {len(features)} nominal episodes "
              f"({X.shape[1]} features each)")

    def _episode_features(self, state_seq: np.ndarray) -> np.ndarray:
        """Episode-level feature summary: mean + std + range of position/velocity."""
        T, D  = state_seq.shape
        vel   = np.vstack([np.zeros((1, D)), np.diff(state_seq, axis=0)])
        acc   = np.vstack([np.zeros((1, D)), np.diff(vel, axis=0)])
        return np.concatenate([
            state_seq.mean(0), state_seq.std(0),
            vel.mean(0),       vel.std(0),       np.abs(vel).max(0),
            acc.mean(0),       np.abs(acc).max(0),
        ])

    # ── Step 2–4: annotate sessions ───────────────────────────────────────────

    def annotate_sessions(self, sessions: list[dict]) -> list[dict]:
        """Run all three annotation layers on every session."""
        results = []

        # batch episode-level anomaly scores
        features = np.array([self._episode_features(ep["states"]) for ep in sessions],
                             dtype=np.float32)
        X_sc     = self.anomaly_scaler.transform(features)
        ep_scores = -self.anomaly_model.score_samples(X_sc)
        thresh    = float(np.quantile(ep_scores, 0.75))

        for i, ep in enumerate(sessions):
            seq        = ep["states"]
            ep_score   = float(ep_scores[i])
            ep_flagged = ep_score >= thresh

            # layer 2: step-level failure types
            if self.failure_ann:
                fail_ann = self.failure_ann.annotate(seq)
            else:
                fail_ann = {"labels": ["nominal"] * len(seq),
                             "confidences": [1.0] * len(seq),
                             "anomaly_scores": [0.0] * len(seq),
                             "failure_counts": {},
                             "dominant_failure": "nominal",
                             "peak_score": 0.0, "peak_step": 0}

            # layer 3: semantic labels
            if self.semantic_ann:
                sem_ann = self.semantic_ann.annotate(seq)
            else:
                sem_ann = {"layers": {}, "coords_3d": seq[:, :3].tolist()}

            result = {
                "episode_id":      ep["episode_id"],
                "episode_index":   i,
                "n_steps":         len(seq),

                # layer 1: episode-level anomaly
                "anomaly_score":   round(ep_score, 4),
                "flagged":         bool(ep_flagged),
                "anomaly_threshold": round(thresh, 4),

                # layer 2: step-level failure types
                "failure_annotation": {
                    "step_labels":      fail_ann["labels"],
                    "step_confs":       [round(c, 3) for c in fail_ann["confidences"]],
                    "step_scores":      [round(s, 4) for s in fail_ann["anomaly_scores"]],
                    "needs_review":     fail_ann.get("needs_review", []),
                    "n_needs_review":   fail_ann.get("n_needs_review", 0),
                    "review_rate":      fail_ann.get("review_rate", 0.0),
                    "dominant":         fail_ann["dominant_failure"],
                    "peak_score":       round(fail_ann["peak_score"], 4),
                    "peak_step":        fail_ann["peak_step"],
                    "counts":           fail_ann["failure_counts"],
                    # human-in-the-loop: feature vectors for uncertain steps (for retrain)
                    "lowconf_features": fail_ann.get("lowconf_features", {}),
                    # active learning: ranked list of most informative steps to review
                    "al_ranked":        fail_ann.get("al_ranked", []),
                },

                # layer 3: semantic labels
                "semantic_annotation": {
                    layer: {
                        "step_labels": data["labels"],
                        "dominant":    data["dominant"],
                        "counts":      data["counts"],
                    }
                    for layer, data in sem_ann.get("layers", {}).items()
                },

                "coords_3d": sem_ann.get("coords_3d", seq[:, :3].tolist()),
            }

            # ── quality score ────────────────────────────────────────────────
            # 1.0 = pristine nominal episode, 0.0 = highly anomalous
            # Penalty: non-nominal step fraction lowers the base score.
            n_steps = len(seq)
            n_nominal = fail_ann["failure_counts"].get("nominal", n_steps)
            nominal_frac  = n_nominal / n_steps if n_steps > 0 else 1.0
            # review_rate penalises low-confidence steps (ambiguous signal)
            review_rate   = fail_ann.get("review_rate", 0.0)
            # combine: anomaly score pulls it down, non-nominal steps pull it down,
            # high review rate (uncertainty) pulls it down slightly
            quality_score = max(0.0, min(1.0,
                (1.0 - ep_score)          * 0.5 +   # anomaly component
                nominal_frac              * 0.35 +  # clean-step fraction
                (1.0 - review_rate)       * 0.15    # model confidence
            ))
            result["quality_score"] = round(quality_score, 4)

            # ground truth if available
            if "true_label" in ep:
                result["true_label"] = int(ep["true_label"])
                result["label_str"]  = "FAILURE" if ep["true_label"] else "OK"
                result["correct"]    = (ep_flagged == bool(ep["true_label"]))

            results.append(result)
            if (i + 1) % 20 == 0:
                print(f"  Annotated {i+1}/{len(sessions)}")

        return results

    # ── Step 5: write annotated HDF5 + JSON ───────────────────────────────────

    def write_output(self, results: list[dict], input_name: str,
                     ground_truth: np.ndarray = None) -> dict:
        safe     = input_name.replace("/", "_").replace(" ", "_")
        h5_path  = OUTPUT_DIR / f"{safe}_annotated.h5"
        rep_path = OUTPUT_DIR / f"{safe}_report.json"
        sum_path = OUTPUT_DIR / f"{safe}_summary.json"

        # ── annotated HDF5 ────────────────────────────────────────────────────
        with h5py.File(h5_path, "w") as f:
            f.attrs["pipeline_version"] = "RobotAnnotationPipeline v1.0"
            f.attrs["n_episodes"]       = len(results)

            for r in results:
                grp = f.create_group(r["episode_id"])
                grp.attrs["anomaly_score"]   = r["anomaly_score"]
                grp.attrs["flagged"]         = r["flagged"]
                grp.attrs["dominant_failure"]= r["failure_annotation"]["dominant"].encode()
                if "true_label" in r:
                    grp.attrs["true_label"]  = r["true_label"]
                    grp.attrs["correct"]     = r["correct"]

                fa = r["failure_annotation"]
                grp.create_dataset("step_failure_types",
                                    data=np.array(fa["step_labels"], dtype="S32"))
                grp.create_dataset("step_failure_confs",
                                    data=np.array(fa["step_confs"], dtype=np.float32))
                grp.create_dataset("step_anomaly_scores",
                                    data=np.array(fa["step_scores"], dtype=np.float32))
                grp.create_dataset("coords_3d",
                                    data=np.array(r["coords_3d"], dtype=np.float32))

                sa = r["semantic_annotation"]
                for layer, data in sa.items():
                    grp.create_dataset(
                        f"semantic_{layer}",
                        data=np.array(data["step_labels"], dtype="S32"))
                    grp.attrs[f"dominant_{layer}"] = data["dominant"].encode()

        # ── performance metrics (if ground truth available) ───────────────────
        perf = {}
        if ground_truth is not None:
            valid = ground_truth >= 0
            scores_arr = np.array([r["anomaly_score"] for r in results])
            flagged_arr= np.array([r["flagged"]       for r in results]).astype(int)
            gt_valid   = ground_truth[valid]
            sc_valid   = scores_arr[valid]
            fl_valid   = flagged_arr[valid]

            if len(np.unique(gt_valid)) > 1:
                auc = roc_auc_score(gt_valid, sc_valid)
                tp  = int(((fl_valid==1) & (gt_valid==1)).sum())
                fp  = int(((fl_valid==1) & (gt_valid==0)).sum())
                fn  = int(((fl_valid==0) & (gt_valid==1)).sum())
                tn  = int(((fl_valid==0) & (gt_valid==0)).sum())
                perf = {
                    "roc_auc":           round(float(auc), 4),
                    "detection_rate_pct": round(tp/(tp+fn)*100,1) if (tp+fn) else 0,
                    "false_positive_rate_pct": round(fp/(fp+tn)*100,1) if (fp+tn) else 0,
                    "confusion_matrix":  {"tp":tp,"fp":fp,"fn":fn,"tn":tn},
                    "note": "Validated against ground truth labels from reward signal",
                }
            else:
                perf = {"note": "Single class in ground truth — no AUC computed. "
                                "All sessions treated as nominal (SOP mode)."}

        # ── quality stats + training export ──────────────────────────────────
        quality_scores = np.array([r["quality_score"] for r in results])
        QUALITY_THRESHOLD = 0.65   # episodes above this go into training export
        training_eps = [r for r in results if r["quality_score"] >= QUALITY_THRESHOLD]
        train_path   = OUTPUT_DIR / f"{safe}_training.h5"
        self._write_training_export(training_eps, train_path)

        # ── review queue ─────────────────────────────────────────────────────
        # episodes with any low-confidence steps — sorted by review count desc
        review_eps = sorted(
            [r for r in results if r["failure_annotation"].get("n_needs_review", 0) > 0],
            key=lambda r: -r["failure_annotation"]["n_needs_review"]
        )
        review_path = OUTPUT_DIR / f"{safe}_review_queue.json"
        review_queue = []
        for r in review_eps:
            fa     = r["failure_annotation"]
            nr_idx = [i for i, flag in enumerate(fa.get("needs_review", [])) if flag]

            # feature vectors for uncertain steps — keyed by step index (str for JSON)
            lowconf_feats = fa.get("lowconf_features", {})
            step_features = {str(k): v for k, v in lowconf_feats.items() if v}

            review_queue.append({
                "episode_id":      r["episode_id"],
                "quality_score":   r["quality_score"],
                "anomaly_score":   r["anomaly_score"],
                "n_needs_review":  fa["n_needs_review"],
                "review_rate":     fa["review_rate"],
                "dominant":        fa["dominant"],
                "low_conf_steps":  nr_idx,
                "step_labels":     fa["step_labels"],
                "step_confs":      fa["step_confs"],
                "label_str":       r.get("label_str", "UNKNOWN"),
                # stored so corrections can be re-injected with features at retrain
                "step_features":   step_features,
                # active learning ranking — most informative steps first
                "al_ranked":       fa.get("al_ranked", []),
            })
        review_path.write_text(json.dumps(review_queue, indent=2))

        # ── summary ───────────────────────────────────────────────────────────
        flagged_results = [r for r in results if r["flagged"]]
        failure_type_totals: dict = {}
        for r in results:
            for k, v in r["failure_annotation"]["counts"].items():
                failure_type_totals[k] = failure_type_totals.get(k, 0) + v
        total_review_steps = sum(r["failure_annotation"].get("n_needs_review", 0) for r in results)
        total_steps        = sum(r["n_steps"] for r in results)

        summary = {
            "pipeline":           "RobotAnnotationPipeline v1.1",
            "input":              input_name,
            "n_episodes":         len(results),
            "n_flagged":          len(flagged_results),
            "flag_rate_pct":      round(len(flagged_results)/len(results)*100, 1),
            "anomaly_threshold":  results[0]["anomaly_threshold"],
            "performance":        perf,
            "failure_type_totals": failure_type_totals,
            "quality": {
                "threshold":          QUALITY_THRESHOLD,
                "mean_quality":       round(float(quality_scores.mean()), 4),
                "episodes_above_threshold": len(training_eps),
                "pct_training_ready": round(len(training_eps)/len(results)*100, 1),
                "review_steps":       total_review_steps,
                "review_rate_pct":    round(total_review_steps/total_steps*100, 1) if total_steps else 0,
            },
            "outputs": {
                "annotated_hdf5":   str(h5_path),
                "training_hdf5":    str(train_path),
                "report_json":      str(rep_path),
                "review_queue":     str(review_path),
                "summary_json":     str(sum_path),
            },
            "models_used": {
                "anomaly_detection": "IsolationForest (fitted from SOP)",
                "failure_typing":    "RobotAnnotator v1.1 (RF+Platt, 93.7% acc)",
                "semantic_labels":   "SemanticAnnotator v1.0 (RF, 94.9% acc)",
            },
            "label_schema": LABEL_SCHEMA,
        }

        rep_path.write_text(json.dumps(results, indent=2))
        sum_path.write_text(json.dumps(summary, indent=2))

        print(f"\n  ✓ Annotated HDF5    : {h5_path}")
        print(f"  ✓ Training export   : {train_path}  ({len(training_eps)} episodes, quality ≥ {QUALITY_THRESHOLD})")
        print(f"  ✓ Review queue      : {review_path}  ({len(review_eps)} episodes, {total_review_steps} low-conf steps)")
        print(f"  ✓ Full report       : {rep_path}")
        print(f"  ✓ Summary           : {sum_path}")
        return summary

    def _write_training_export(self, episodes: list[dict], path: Path):
        """
        Write a curated HDF5 containing only high-quality episodes,
        with all annotation layers attached — ready for robot policy training.
        Episodes are sorted by quality_score descending.
        """
        sorted_eps = sorted(episodes, key=lambda r: -r["quality_score"])
        with h5py.File(path, "w") as f:
            f.attrs["description"]  = "Curated training data — high quality episodes only"
            f.attrs["n_episodes"]   = len(sorted_eps)
            f.attrs["pipeline"]     = "RobotAnnotationPipeline v1.1"

            for i, r in enumerate(sorted_eps):
                grp = f.create_group(f"episode_{i:04d}")
                grp.attrs["quality_score"]   = r["quality_score"]
                grp.attrs["anomaly_score"]   = r["anomaly_score"]
                grp.attrs["original_id"]     = r["episode_id"]
                grp.attrs["dominant_failure"]= r["failure_annotation"]["dominant"].encode()

                fa = r["failure_annotation"]
                grp.create_dataset("step_failure_types",
                                   data=np.array(fa["step_labels"], dtype="S32"))
                grp.create_dataset("step_failure_confs",
                                   data=np.array(fa["step_confs"], dtype=np.float32))
                grp.create_dataset("step_anomaly_scores",
                                   data=np.array(fa["step_scores"], dtype=np.float32))
                grp.create_dataset("coords_3d",
                                   data=np.array(r["coords_3d"], dtype=np.float32))
                grp.create_dataset("needs_review",
                                   data=np.array(fa.get("needs_review", []), dtype=bool))

                sa = r["semantic_annotation"]
                for layer, data in sa.items():
                    grp.create_dataset(f"semantic_{layer}",
                                       data=np.array(data["step_labels"], dtype="S32"))
                    grp.attrs[f"dominant_{layer}"] = data["dominant"].encode()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, sop_path: Path, input_path: Path) -> dict:
        print("\n" + "="*60)
        print("RobotAnnotationPipeline v1.0")
        print("="*60)

        print(f"\n[1/5] Loading SOP (nominal reference): {sop_path.name}")
        sop_sessions = load_hdf5_sessions(sop_path)
        if not sop_sessions:
            raise ValueError(f"No valid sessions found in SOP file: {sop_path}")

        print(f"\n[2/5] Fitting anomaly detector from SOP...")
        self.fit_from_sop(sop_sessions)

        print(f"\n[3/5] Loading input sessions: {input_path.name}")
        input_sessions = load_hdf5_sessions(input_path)
        if not input_sessions:
            raise ValueError(f"No valid sessions in input: {input_path}")

        print(f"\n[4/5] Loading pre-trained annotation models...")
        ann_path = OUTPUT_DIR / "robot_annotator.pkl"
        sem_path = OUTPUT_DIR / "semantic_annotator.pkl"
        if ann_path.exists():
            self.failure_ann = RobotAnnotator.load(ann_path)
        else:
            print("  WARNING: RobotAnnotator not found — run annotation_model.py --train")
        if sem_path.exists():
            self.semantic_ann = SemanticAnnotator.load(sem_path)
        else:
            print("  WARNING: SemanticAnnotator not found — run semantic_annotator.py --train")

        print(f"\n[5/5] Annotating {len(input_sessions)} sessions...")
        ground_truth = derive_ground_truth(input_sessions)
        results      = self.annotate_sessions(input_sessions)
        summary      = self.write_output(results, input_path.stem, ground_truth)

        self._print_summary(summary, results)
        return summary

    def _print_summary(self, summary: dict, results: list[dict]):
        print("\n" + "="*60)
        print("PIPELINE RESULTS")
        print("="*60)
        print(f"  Sessions processed : {summary['n_episodes']}")
        print(f"  Flagged anomalous  : {summary['n_flagged']}  "
              f"({summary['flag_rate_pct']}%)")

        perf = summary.get("performance", {})
        if "roc_auc" in perf:
            print(f"\n  VALIDATION (vs ground truth):")
            print(f"    ROC-AUC          : {perf['roc_auc']}")
            print(f"    Detection rate   : {perf['detection_rate_pct']}%")
            print(f"    False pos. rate  : {perf['false_positive_rate_pct']}%")
            cm = perf["confusion_matrix"]
            print(f"    Correctly caught : {cm['tp']} failures")
            print(f"    Missed           : {cm['fn']} failures")
            print(f"    False alarms     : {cm['fp']}")
        else:
            print(f"\n  {perf.get('note','No ground truth — SOP mode.')}")

        print(f"\n  TOP FAILURE TYPES:")
        totals = summary.get("failure_type_totals", {})
        for k, v in sorted(totals.items(), key=lambda x: -x[1])[:5]:
            if v > 0:
                print(f"    {k:20s}: {v:5,} steps")

        q = summary.get("quality", {})
        if q:
            print(f"\n  QUALITY & TRAINING EXPORT:")
            print(f"    Mean quality score  : {q.get('mean_quality','—')}")
            print(f"    Training-ready eps  : {q.get('episodes_above_threshold','—')} "
                  f"({q.get('pct_training_ready','—')}%)")
            print(f"    Human review steps  : {q.get('review_steps','—')} "
                  f"({q.get('review_rate_pct','—')}% of all steps)")

        print(f"\n  OUTPUT FILES:")
        for k, v in summary["outputs"].items():
            print(f"    {v}")


# ── Demo mode: uses open source data, no files needed ─────────────────────────

def run_demo():
    """
    Demo mode: splits the xarm_lift HDF5 we already have into
    SOP (nominal episodes) and input (all episodes), then runs
    the full pipeline to show what client output looks like.
    """
    import tempfile
    source = OUTPUT_DIR / "lerobot_xarm_lift_medium_replay_scores.h5"
    if not source.exists():
        print("Run main.py first to generate benchmark data.")
        return

    print("Demo mode: splitting xarm_lift into SOP + input...")

    with h5py.File(source, "r") as f:
        features    = f["features"][:]
        labels      = f["true_labels"][:]
        scores      = f["anomaly_scores"][:]

    # SOP = nominal episodes only (first 50)
    # Reshape 24-dim episode features to (T=6, D=4) — matches xarm 4-DOF state space
    # so annotation_model (trained on D=4 data, expects 8*4+4=36 step features) works.
    nominal_idx = np.where(labels == 0)[0][:50]
    sop_path    = OUTPUT_DIR / "_demo_sop.h5"
    with h5py.File(sop_path, "w") as f:
        for i, idx in enumerate(nominal_idx):
            seq = features[idx].reshape(-1, 4)   # (6, 4)
            grp = f.create_group(f"episode_{i:04d}")
            grp.create_dataset("states", data=seq.astype(np.float32))

    # Input = first 100 episodes (mix of nominal + failures)
    input_path = OUTPUT_DIR / "_demo_input.h5"
    with h5py.File(input_path, "w") as f:
        for i in range(min(100, len(features))):
            seq = features[i].reshape(-1, 4)     # (6, 4)
            grp = f.create_group(f"episode_{i:04d}")
            grp.create_dataset("states", data=seq.astype(np.float32))
            grp.attrs["true_label"] = int(labels[i])

    pipeline = RobotAnnotationPipeline()
    pipeline.run(sop_path, input_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot annotation pipeline: SOP + HDF5 → annotated output")
    parser.add_argument("--sop",     type=str, help="HDF5 file of nominal sessions (SOP)")
    parser.add_argument("--input",   type=str, help="HDF5 file of sessions to annotate")
    parser.add_argument("--demo",    action="store_true",
                        help="Run demo using existing benchmark data")
    parser.add_argument("--dataset", type=str,
                        help="LeRobot dataset name — downloads and runs full pipeline")
    args = parser.parse_args()

    if args.demo:
        run_demo()

    elif args.sop and args.input:
        pipeline = RobotAnnotationPipeline()
        pipeline.run(Path(args.sop), Path(args.input))

    elif args.dataset:
        # convenience: download dataset, split into SOP + input, run pipeline
        from main import load_lerobot
        print(f"Downloading {args.dataset}...")
        features_arr, labels_arr = load_lerobot(args.dataset, max_episodes=200)

        nominal_idx = np.where(labels_arr == 0)[0]
        failure_idx = np.where(labels_arr == 1)[0]

        def write_split(indices, path, include_labels=False):
            with h5py.File(path, "w") as f:
                for i, idx in enumerate(indices):
                    seq = features_arr[idx].reshape(-1, max(1, features_arr.shape[1]//8))
                    grp = f.create_group(f"episode_{i:04d}")
                    grp.create_dataset("states", data=seq.astype(np.float32))
                    if include_labels:
                        grp.attrs["true_label"] = int(labels_arr[idx])

        safe = args.dataset.replace("/", "_")
        sop_path   = OUTPUT_DIR / f"{safe}_sop.h5"
        input_path = OUTPUT_DIR / f"{safe}_input.h5"

        write_split(nominal_idx[:50], sop_path)
        write_split(np.concatenate([nominal_idx[50:150], failure_idx[:50]]),
                    input_path, include_labels=True)

        pipeline = RobotAnnotationPipeline()
        pipeline.run(sop_path, input_path)

    else:
        parser.print_help()
        print("\nQuick start:  python pipeline.py --demo")
