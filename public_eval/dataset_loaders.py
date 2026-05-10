"""
public_eval/dataset_loaders.py
================================
Loaders for 5 public robotics datasets, each normalized to the common
internal schema defined in PRODUCT_TRAINING_PLAN.md.

Common schema (all Optional fields may be None):
    dataset_name      str
    episode_id        str
    timesteps         int
    state_seq         np.ndarray  (T, D_s)
    action_seq        np.ndarray | None  (T, D_a)
    video_frames      list[np.ndarray] | None
    image_paths       list[str] | None
    language_task     str | None
    episode_label     str | None   "nominal" | "failure" | class name
    step_labels       np.ndarray | None  (T,) int or str
    semantic_labels   dict | None
    failure_category  str | None
    source_label_type str   "human" | "reward" | "weak" | "synthetic" | "vqa"
    metadata          dict

Usage:
    from public_eval.dataset_loaders import load_all_datasets
    datasets = load_all_datasets(max_episodes_per_dataset=200)
    # datasets: dict[dataset_name -> list[episode_dict]]
"""

import os
import io
import csv
import json
import warnings
import hashlib
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np

warnings.filterwarnings("ignore")

CACHE_DIR = Path("benchmark_output/public_dataset_eval/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("benchmark_output/public_dataset_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Schema builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_episode(
    dataset_name: str,
    episode_id: str,
    state_seq: np.ndarray,
    source_label_type: str,
    action_seq=None,
    episode_label=None,
    step_labels=None,
    failure_category=None,
    language_task=None,
    semantic_labels=None,
    image_paths=None,
    video_frames=None,
    metadata=None,
) -> dict:
    """Construct a schema-compliant episode dict."""
    return {
        "dataset_name":     dataset_name,
        "episode_id":       episode_id,
        "timesteps":        len(state_seq),
        "state_seq":        np.asarray(state_seq, dtype=np.float32),
        "action_seq":       np.asarray(action_seq, dtype=np.float32) if action_seq is not None else None,
        "video_frames":     video_frames,
        "image_paths":      image_paths,
        "language_task":    language_task,
        "episode_label":    episode_label,
        "step_labels":      np.asarray(step_labels) if step_labels is not None else None,
        "semantic_labels":  semantic_labels,
        "failure_category": failure_category,
        "source_label_type": source_label_type,
        "metadata":         metadata or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1 — BotFails (HuggingFace: kantine/BotFails)
# Label type: human (episode-level failure/success)
# ─────────────────────────────────────────────────────────────────────────────

def load_botfails(max_episodes: int = 300) -> tuple[list[dict], dict]:
    """
    Load BotFails from HuggingFace datasets.
    Falls back to a graceful failure with metadata if datasets lib not available.
    Returns (episodes, access_report).
    """
    report = {
        "dataset": "botfails",
        "url": "https://huggingface.co/datasets/kantine/BotFails",
        "label_type": "human",
        "attempted": True,
        "success": False,
        "n_episodes": 0,
        "failure_reason": None,
        "caveats": [
            "Human-labeled episode-level failure/success",
            "Step-level labels not confirmed — may require parsing annotations field",
        ],
    }

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        report["failure_reason"] = "datasets library not installed — run: pip install datasets"
        print(f"[BotFails] SKIP: {report['failure_reason']}")
        return [], report

    try:
        print("[BotFails] Loading via direct parquet download (HF Hub)…")
        # BotFails has Video feature type unsupported by datasets<=2.x.
        # Structure: normal_train/{task}_expert/*.parquet (nominal)
        #            test/{task}_anomaly/*.parquet       (failure)
        #            labels/{task}_anomaly/*.csv          (step labels, col '0')
        try:
            from huggingface_hub import list_repo_files, hf_hub_download  # type: ignore
            import pandas as pd  # type: ignore
        except ImportError as ie:
            raise Exception(f"huggingface_hub or pandas not available: {ie}")

        all_files = list(list_repo_files("kantine/BotFails", repo_type="dataset"))
        nominal_pqs = [f for f in all_files if "normal_train" in f and f.endswith(".parquet")]
        anomaly_pqs = [f for f in all_files if "/test/" in f and "anomaly" in f and f.endswith(".parquet")]
        label_csvs  = {f.split("/")[-1].replace("_labels.csv", ""): f
                       for f in all_files if f.endswith("_labels.csv")}

        # Sample up to max_episodes/2 from each class
        per_class = max_episodes // 2
        nominal_pqs  = nominal_pqs[:per_class]
        anomaly_pqs  = anomaly_pqs[:per_class]

        episodes = []
        def load_botfails_parquet(pq_path, ep_label, label_csv_path=None, ep_idx=0):
            local = hf_hub_download("kantine/BotFails", pq_path, repo_type="dataset")
            df = pd.read_parquet(local)
            states = np.array(df["observation.state"].tolist(), dtype=np.float32)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            actions = None
            if "action" in df.columns:
                try:
                    actions = np.array(df["action"].tolist(), dtype=np.float32)
                    if actions.ndim == 1:
                        actions = actions.reshape(-1, 1)
                except Exception:
                    pass
            step_labels = None
            if label_csv_path:
                try:
                    llocal = hf_hub_download("kantine/BotFails", label_csv_path, repo_type="dataset")
                    ldf = pd.read_csv(llocal)
                    step_labels = ldf.iloc[:, 0].values.astype(int)
                    if len(step_labels) != len(states):
                        step_labels = None  # length mismatch — skip
                except Exception:
                    pass
            ep_name = pq_path.split("/")[-1].replace(".parquet", "")
            return make_episode(
                dataset_name="botfails",
                episode_id=f"botfails_{ep_label}_{ep_idx:04d}_{ep_name}",
                state_seq=states,
                action_seq=actions,
                episode_label=ep_label,
                step_labels=step_labels,
                failure_category=None if ep_label == "nominal" else "unspecified_anomaly",
                source_label_type="human",
                metadata={"parquet_path": pq_path, "has_step_labels": step_labels is not None},
            )

        for i, pq in enumerate(nominal_pqs):
            try:
                ep = load_botfails_parquet(pq, "nominal", None, i)
                episodes.append(ep)
            except Exception as e:
                print(f"[BotFails] Nominal parquet {i} failed: {e}")

        for i, pq in enumerate(anomaly_pqs):
            ep_key = pq.split("/")[-1].replace(".parquet", "") + "_labels"
            csv_path = label_csvs.get(pq.split("/")[-1].replace(".parquet", ""))
            # Match: labels/{task}_anomaly/episode_XXXXXX_labels.csv
            task_dir = "/".join(pq.split("/")[1:3])  # e.g. test/domotic_dishTidyUp_anomaly
            label_key = pq.split("/")[-1].replace(".parquet", "")
            csv_path = label_csvs.get(label_key)
            try:
                ep = load_botfails_parquet(pq, "failure", csv_path, i)
                episodes.append(ep)
            except Exception as e:
                print(f"[BotFails] Anomaly parquet {i} failed: {e}")

        if not episodes:
            raise Exception("No episodes loaded from parquet files")

        report["success"] = True
        report["n_episodes"] = len(episodes)
        report["source"] = "direct parquet download via huggingface_hub"
        report["caveats"].append(
            "Loaded via direct parquet download — datasets library Video feature workaround")
        label_counts = {}
        for ep in episodes:
            lbl = ep["episode_label"] or "unknown"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        report["label_distribution"] = label_counts
        step_label_count = sum(1 for ep in episodes if ep["step_labels"] is not None)
        report["has_step_labels"] = step_label_count
        print(f"[BotFails] Loaded {len(episodes)} episodes ({step_label_count} with step labels). "
              f"Labels: {label_counts}")
        return episodes, report

    except Exception as e:
        report["failure_reason"] = str(e)
        print(f"[BotFails] Direct parquet load FAILED: {e}")
        # No fallback — BotFails is the highest-priority human-labeled dataset
        return [], report


def _load_botfails_hf_streaming_legacy(report, max_episodes):
    """Legacy streaming path — kept for reference but not called."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return [], report

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("kantine/BotFails", split="train", streaming=True, trust_remote_code=True)
        episodes = []
        for i, row in enumerate(ds):
            if i >= max_episodes:
                break
            state = None
            for k in ["observation.state", "state", "obs", "joint_pos"]:
                if k in row and row[k] is not None:
                    state = np.array(row[k], dtype=np.float32)
                    break
            if state is None:
                continue
            if state.ndim == 1:
                state = state.reshape(-1, 1)

            action = None
            for k in ["action", "actions", "cmd"]:
                if k in row and row[k] is not None:
                    action = np.array(row[k], dtype=np.float32)
                    if action.ndim == 1:
                        action = action.reshape(-1, 1)
                    break

            # Label
            label_raw = None
            for k in ["label", "success", "failure", "episode_label", "annotation"]:
                if k in row:
                    label_raw = row[k]
                    break

            if label_raw is None:
                episode_label = None
                failure_cat = None
            elif isinstance(label_raw, bool) or (isinstance(label_raw, int) and label_raw in (0, 1)):
                episode_label = "nominal" if label_raw else "failure"
                failure_cat = None if label_raw else "unspecified"
            elif isinstance(label_raw, str):
                episode_label = label_raw.lower()
                failure_cat = label_raw if "fail" in label_raw.lower() else None
            else:
                episode_label = str(label_raw)
                failure_cat = None

            ep = make_episode(
                dataset_name="botfails",
                episode_id=f"botfails_ep{i:05d}",
                state_seq=state,
                action_seq=action,
                episode_label=episode_label,
                failure_category=failure_cat,
                source_label_type="human",
                metadata={"row_index": i, "raw_label": str(label_raw)},
            )
            episodes.append(ep)

        report["success"] = len(episodes) > 0
        report["n_episodes"] = len(episodes)
        if episodes:
            label_counts = {}
            for ep in episodes:
                lbl = ep["episode_label"] or "unknown"
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            report["label_distribution"] = label_counts
        print(f"[BotFails] Loaded {len(episodes)} episodes.")
        return episodes, report

    except Exception as e:
        report["failure_reason"] = str(e)
        print(f"[BotFails] FAILED: {e}")
        return [], report


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2 — RoboFAC (HuggingFace: MINT-SJTU/RoboFAC-dataset)
# Label type: vqa (failure analysis Q&A)
# ─────────────────────────────────────────────────────────────────────────────

def load_robofac(max_episodes: int = 300) -> tuple[list[dict], dict]:
    """
    Load RoboFAC dataset. Primarily video + QA annotations.
    Extracts failure category from QA text where possible.
    """
    report = {
        "dataset": "robofac",
        "url": "https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset",
        "label_type": "vqa",
        "attempted": True,
        "success": False,
        "n_episodes": 0,
        "failure_reason": None,
        "caveats": [
            "VQA labels: human-written QA about failure causes — not structured failure taxonomy",
            "Video data may not be available in HF dataset (could be metadata only)",
            "Failure category extracted from QA text via keyword matching — noisy",
            "No proprioceptive state sequences — visual-only dataset",
        ],
    }

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        report["failure_reason"] = "datasets library not installed"
        print(f"[RoboFAC] SKIP: {report['failure_reason']}")
        return [], report

    FAILURE_KEYWORDS = {
        "grasp": "grasp_failure",
        "slip": "grasp_slip",
        "drop": "grasp_slip",
        "collision": "collision",
        "stuck": "stuck",
        "miss": "missed_target",
        "wrong": "wrong_object",
        "fall": "object_fall",
        "tilt": "object_tilt",
        "occlus": "occlusion",
        "place": "placement_error",
    }

    def extract_failure_category(text: str) -> Optional[str]:
        text_lower = text.lower()
        for kw, cat in FAILURE_KEYWORDS.items():
            if kw in text_lower:
                return cat
        return "unspecified_failure"

    try:
        print("[RoboFAC] Loading via direct JSON download (simulation_data)…")
        # RoboFAC's HF dataset loader fails due to JSON format issues.
        # We load simulation JSON files directly: each has episodes with 'success' labels.
        # ⚠ IMPORTANT CAVEAT: No proprioceptive state data is available in these JSON files.
        # Only metadata is accessible: elapsed_steps, episode_seed, success.
        # This means tabular models have near-zero signal from these features.
        # A visual encoder (CLIP on the video files) would be needed for real performance.
        # We load anyway to preserve label structure and document the limitation.
        try:
            from huggingface_hub import list_repo_files, hf_hub_download  # type: ignore
        except ImportError as ie:
            raise Exception(f"huggingface_hub not available: {ie}")

        all_files = list(list_repo_files("MINT-SJTU/RoboFAC-dataset", repo_type="dataset"))
        sim_jsons = [f for f in all_files if f.startswith("simulation_data") and f.endswith(".json")]
        # Limit files to avoid excessive downloads
        sim_jsons = sim_jsons[:20]  # 20 files × ~60 episodes = ~1200 episodes max

        episodes = []
        for json_file in sim_jsons:
            if len(episodes) >= max_episodes:
                break
            try:
                local = hf_hub_download("MINT-SJTU/RoboFAC-dataset", json_file, repo_type="dataset")
                with open(local) as fh:
                    data = json.load(fh)
                eps_raw = data.get("episodes", [])
                task_name = json_file.split("/")[1] if "/" in json_file else "unknown_task"

                for ep_raw in eps_raw:
                    if len(episodes) >= max_episodes:
                        break
                    success = ep_raw.get("success", None)
                    elapsed = ep_raw.get("elapsed_steps", 0) or 0
                    seed = float(ep_raw.get("episode_seed", 0) or 0)
                    ep_id = str(ep_raw.get("episode_id", len(episodes)))

                    # Minimal features: elapsed_steps (proxy for run duration) + seed
                    # This is explicitly documented as insufficient for meaningful ML
                    state = np.array([[elapsed / 500.0, (seed % 1000) / 1000.0]],
                                     dtype=np.float32)

                    if success is None:
                        episode_label = None
                    else:
                        episode_label = "nominal" if success else "failure"

                    # Failure category from task name keyword matching
                    failure_cat = extract_failure_category(task_name) if episode_label == "failure" else None

                    ep = make_episode(
                        dataset_name="robofac",
                        episode_id=f"robofac_{task_name}_{ep_id}",
                        state_seq=state,
                        episode_label=episode_label,
                        failure_category=failure_cat,
                        language_task=task_name,
                        source_label_type="vqa",
                        metadata={
                            "json_file": json_file,
                            "task": task_name,
                            "elapsed_steps": elapsed,
                            "no_proprioceptive_data": True,
                            "caveat": "Only elapsed_steps used as feature — no state sequence available",
                        },
                    )
                    episodes.append(ep)
            except Exception as ef:
                print(f"[RoboFAC] File {json_file} failed: {ef}")
                continue

        if not episodes:
            raise Exception("No episodes loaded from RoboFAC JSON files")

        report["success"] = True
        report["n_episodes"] = len(episodes)
        report["source"] = "simulation JSON files (direct download)"
        report["caveats"].append(
            "⚠ NO STATE DATA: Only elapsed_steps used as feature. "
            "Tabular models will have near-random performance. "
            "Visual model (CLIP on videos) required for meaningful evaluation."
        )
        label_counts = {}
        for ep in episodes:
            lbl = ep["episode_label"] or "unknown"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        report["label_distribution"] = label_counts
        print(f"[RoboFAC] Loaded {len(episodes)} episodes (minimal features only). Labels: {label_counts}")
        return episodes, report

    except Exception as e:
        report["failure_reason"] = str(e)
        print(f"[RoboFAC] FAILED: {e}")
        return [], report


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3 — ViFailback (HuggingFace: sii-rhos-ai/ViFailback-Dataset)
# Label type: human/weak (failure+correction pairs)
# ─────────────────────────────────────────────────────────────────────────────

def load_vifailback(max_episodes: int = 300) -> tuple[list[dict], dict]:
    """
    Load ViFailback dataset. Paired failure/correction trajectories.
    """
    report = {
        "dataset": "vifailback",
        "url": "https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset",
        "label_type": "human",
        "attempted": True,
        "success": False,
        "n_episodes": 0,
        "failure_reason": None,
        "caveats": [
            "Paired failure/correction structure — each failure has a correction episode",
            "Episode labels assumed: 'failure' for failure episodes, 'nominal' for corrections",
            "Step-level labels not confirmed — derived from episode label",
            "Label type classified as 'human' but may include heuristic elements",
        ],
    }

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        report["failure_reason"] = "datasets library not installed"
        print(f"[ViFailback] SKIP: {report['failure_reason']}")
        return [], report

    try:
        print("[ViFailback] Attempting HuggingFace load…")
        # ViFailback may only have 'test' split
        for split_try in ["train", "test", "validation"]:
            try:
                ds = load_dataset("sii-rhos-ai/ViFailback-Dataset", split=split_try,
                                  streaming=True, trust_remote_code=True)
                _ = next(iter(ds))  # confirm it works
                print(f"[ViFailback] Using split: {split_try}")
                break
            except Exception:
                continue
        else:
            raise Exception("No accessible split found for ViFailback-Dataset")
        episodes = []
        for i, row in enumerate(ds):
            if i >= max_episodes:
                break

            # Try state extraction
            state = None
            for k in ["state", "observation.state", "joint_positions", "robot_obs", "obs"]:
                if k in row and row[k] is not None:
                    try:
                        arr = np.array(row[k], dtype=np.float32)
                        if arr.size > 0:
                            state = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)
                            break
                    except Exception:
                        pass

            if state is None:
                state = np.zeros((1, 1), dtype=np.float32)

            # Determine if failure or correction episode
            is_failure = True  # default
            for k in ["is_failure", "failure", "label", "split", "type"]:
                if k in row:
                    val = row[k]
                    if isinstance(val, bool):
                        is_failure = val
                    elif isinstance(val, str):
                        is_failure = "fail" in val.lower()
                    elif isinstance(val, int):
                        is_failure = bool(val)
                    break

            episode_label = "failure" if is_failure else "nominal"
            failure_cat = row.get("failure_type", row.get("failure_category", "unspecified")) if is_failure else None
            if isinstance(failure_cat, str) and not is_failure:
                failure_cat = None

            action = None
            for k in ["action", "actions"]:
                if k in row and row[k] is not None:
                    try:
                        action = np.array(row[k], dtype=np.float32)
                        if action.ndim == 1:
                            action = action.reshape(-1, 1)
                    except Exception:
                        pass
                    break

            ep = make_episode(
                dataset_name="vifailback",
                episode_id=f"vifailback_ep{i:05d}",
                state_seq=state,
                action_seq=action,
                episode_label=episode_label,
                failure_category=failure_cat if isinstance(failure_cat, str) else None,
                source_label_type="human",
                metadata={"row_index": i, "is_failure": is_failure},
            )
            episodes.append(ep)

        report["success"] = len(episodes) > 0
        report["n_episodes"] = len(episodes)
        if episodes:
            lcounts = {}
            for ep in episodes:
                lbl = ep["episode_label"] or "unknown"
                lcounts[lbl] = lcounts.get(lbl, 0) + 1
            report["label_distribution"] = lcounts
        print(f"[ViFailback] Loaded {len(episodes)} episodes.")
        return episodes, report

    except Exception as e:
        report["failure_reason"] = str(e)
        print(f"[ViFailback] FAILED: {e}")
        return [], report


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 4 — LeRobot reward-derived (DROID/xarm — already partially cached)
# Label type: reward (success derived from reward signal)
# ─────────────────────────────────────────────────────────────────────────────

def load_lerobot_reward(max_episodes: int = 300) -> tuple[list[dict], dict]:
    """
    Load LeRobot datasets (xarm / droid_100) using HuggingFace datasets lib.
    Episode label derived from reward: success if reward[-1] > 0.5.
    Also checks local cache in benchmark_output/.
    """
    report = {
        "dataset": "lerobot_reward",
        "url": "https://huggingface.co/datasets/lerobot/xarm_lift_medium_replay",
        "label_type": "reward",
        "attempted": True,
        "success": False,
        "n_episodes": 0,
        "failure_reason": None,
        "caveats": [
            "Label derived from reward signal: success = reward[-1] > 0.5",
            "Reward threshold 0.5 is arbitrary — different thresholds give different class balance",
            "No explicit failure category labels — only success/failure binary",
            "Hard negatives may exist: low reward episodes that look nominal visually",
            "xarm_lift is a simulation dataset — sim-to-real gap applies",
        ],
    }

    # ── Try local pickle cache first ──────────────────────────────────────────
    local_paths = [
        Path("benchmark_output/lerobot_xarm_lift_medium_replay_episodes.pkl"),
        Path("benchmark_output/lerobot_xarm_push_medium_replay_episodes.pkl"),
        Path("benchmark_output/lerobot_droid_100_episodes.pkl"),
    ]
    for pkl in local_paths:
        if pkl.exists():
            try:
                import pickle
                with open(pkl, "rb") as f:
                    raw = pickle.load(f)
                if isinstance(raw, list) and len(raw) > 0:
                    episodes = []
                    for i, ep_raw in enumerate(raw[:max_episodes]):
                        # raw episodes may be dict with states/actions/rewards
                        if isinstance(ep_raw, dict):
                            states = ep_raw.get("states", ep_raw.get("state", None))
                            actions = ep_raw.get("actions", ep_raw.get("action", None))
                            rewards = ep_raw.get("rewards", ep_raw.get("reward", None))
                        else:
                            continue

                        if states is None:
                            continue
                        states = np.asarray(states, dtype=np.float32)
                        if states.ndim == 1:
                            states = states.reshape(-1, 1)

                        reward_val = None
                        if rewards is not None:
                            rew_arr = np.asarray(rewards, dtype=np.float32).flatten()
                            reward_val = float(rew_arr[-1])

                        episode_label = None
                        if reward_val is not None:
                            episode_label = "nominal" if reward_val > 0.5 else "failure"

                        ep = make_episode(
                            dataset_name="lerobot_reward",
                            episode_id=f"lerobot_{pkl.stem}_ep{i:05d}",
                            state_seq=states,
                            action_seq=np.asarray(actions, dtype=np.float32) if actions is not None else None,
                            episode_label=episode_label,
                            failure_category=None,
                            source_label_type="reward",
                            metadata={"source_file": pkl.name, "reward_last": reward_val},
                        )
                        episodes.append(ep)

                    if episodes:
                        report["success"] = True
                        report["n_episodes"] = len(episodes)
                        report["source"] = f"local cache: {pkl.name}"
                        label_counts = {}
                        for ep in episodes:
                            lbl = ep["episode_label"] or "unlabeled"
                            label_counts[lbl] = label_counts.get(lbl, 0) + 1
                        report["label_distribution"] = label_counts
                        print(f"[LeRobot] Loaded {len(episodes)} episodes from {pkl.name}")
                        return episodes, report
            except Exception as e:
                print(f"[LeRobot] Cache load failed ({pkl.name}): {e}")

    # ── Try HuggingFace download ──────────────────────────────────────────────
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        report["failure_reason"] = "datasets library not installed and no local cache found"
        print(f"[LeRobot] SKIP: {report['failure_reason']}")
        return [], report

    for hf_name in ["lerobot/xarm_lift_medium_replay", "lerobot/pusht"]:
        try:
            print(f"[LeRobot] Attempting HuggingFace load: {hf_name}…")
            ds = load_dataset(hf_name, split="train", streaming=True, trust_remote_code=True)

            # LeRobot HF datasets are STEP-level (one row per timestep).
            # We must group rows by episode_index to build episode-level data.
            episodes = []
            current_ep_idx = None
            current_states = []
            current_actions = []
            current_rewards = []
            rows_consumed = 0
            MAX_ROWS = max_episodes * 200  # guard against infinite stream

            def flush_episode(ep_idx, states, actions, rewards, hf_name, ep_count):
                """Convert accumulated step rows → one episode dict."""
                if not states:
                    return None
                state_arr = np.array(states, dtype=np.float32)
                if state_arr.ndim == 1:
                    state_arr = state_arr.reshape(-1, 1)

                action_arr = None
                if actions:
                    try:
                        action_arr = np.array(actions, dtype=np.float32)
                        if action_arr.ndim == 1:
                            action_arr = action_arr.reshape(-1, 1)
                    except Exception:
                        pass

                # Episode success = max reward in episode > 0.5
                reward_arr = np.array(rewards, dtype=np.float32) if rewards else None
                reward_max = float(reward_arr.max()) if reward_arr is not None and len(reward_arr) else None
                episode_label = None
                if reward_max is not None:
                    episode_label = "nominal" if reward_max > 0.5 else "failure"

                return make_episode(
                    dataset_name="lerobot_reward",
                    episode_id=f"lerobot_{hf_name.replace('/', '_')}_ep{ep_count:05d}",
                    state_seq=state_arr,
                    action_seq=action_arr,
                    episode_label=episode_label,
                    source_label_type="reward",
                    metadata={"hf_dataset": hf_name, "reward_max": reward_max,
                              "n_steps": len(states), "episode_index": ep_idx},
                )

            for row in ds:
                rows_consumed += 1
                if rows_consumed > MAX_ROWS:
                    break
                if len(episodes) >= max_episodes:
                    break

                # Get episode index from row
                ep_idx = row.get("episode_index", row.get("episode_id", 0))
                if isinstance(ep_idx, (list, np.ndarray)):
                    ep_idx = int(ep_idx[0]) if len(ep_idx) else 0
                else:
                    ep_idx = int(ep_idx)

                # New episode → flush previous
                if current_ep_idx is not None and ep_idx != current_ep_idx:
                    ep = flush_episode(current_ep_idx, current_states, current_actions,
                                       current_rewards, hf_name, len(episodes))
                    if ep:
                        episodes.append(ep)
                    current_states, current_actions, current_rewards = [], [], []

                current_ep_idx = ep_idx

                # Accumulate state
                for k in ["observation.state", "state", "obs"]:
                    if k in row and row[k] is not None:
                        try:
                            val = np.array(row[k], dtype=np.float32).flatten()
                            current_states.append(val)
                            break
                        except Exception:
                            pass

                # Accumulate action
                for k in ["action", "actions"]:
                    if k in row and row[k] is not None:
                        try:
                            val = np.array(row[k], dtype=np.float32).flatten()
                            current_actions.append(val)
                            break
                        except Exception:
                            pass

                # Accumulate reward
                for k in ["reward", "next.reward"]:
                    if k in row and row[k] is not None:
                        try:
                            current_rewards.append(float(row[k]))
                            break
                        except Exception:
                            pass

            # Flush last episode
            if current_states and len(episodes) < max_episodes:
                ep = flush_episode(current_ep_idx, current_states, current_actions,
                                   current_rewards, hf_name, len(episodes))
                if ep:
                    episodes.append(ep)

            if episodes:
                report["success"] = True
                report["n_episodes"] = len(episodes)
                report["source"] = hf_name
                report["rows_consumed"] = rows_consumed
                label_counts = {}
                for ep in episodes:
                    lbl = ep["episode_label"] or "unlabeled"
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                report["label_distribution"] = label_counts
                print(f"[LeRobot] Loaded {len(episodes)} episodes ({rows_consumed} rows) from {hf_name}.")
                return episodes, report

        except Exception as e:
            print(f"[LeRobot] HF load failed ({hf_name}): {e}")
            continue

    report["failure_reason"] = "All LeRobot sources failed"
    return [], report


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 5 — UCI Robot Execution Failures
# Label type: human (5 failure classes + nominal, clean tabular)
# ─────────────────────────────────────────────────────────────────────────────

UCI_URL = "https://archive.ics.uci.edu/static/public/138/robot+execution+failures.zip"
UCI_FILES = {
    "lp1.data": "normal",
    "lp2.data": "normal",
    "lp3.data": "normal",
    "lp4.data": "normal",
    "lp5.data": "normal",
}
# UCI task names → failure class mapping
UCI_TASK_LABEL = {
    "lp1": "normal",
    "lp2": "normal",
    "lp3": "normal",
    "lp4": "normal",
    "lp5": "normal",
}
UCI_CLASSES = [
    "normal",
    "collision",
    "obstruction",
    "fr_collision",    # frontal_collision
    "back_col_obstacle",
    "moving_obstacle",
    "slipping",
]


def _download_uci(cache_path: Path) -> Optional[bytes]:
    """Download UCI zip, return bytes or None on failure."""
    if cache_path.exists():
        return cache_path.read_bytes()
    try:
        print(f"[UCI] Downloading from {UCI_URL}…")
        with urllib.request.urlopen(UCI_URL, timeout=30) as r:
            data = r.read()
        cache_path.write_bytes(data)
        print(f"[UCI] Downloaded {len(data)//1024} KB → {cache_path}")
        return data
    except Exception as e:
        print(f"[UCI] Download failed: {e}")
        return None


def _parse_uci_data(text: str, task_name: str) -> list[dict]:
    """
    Parse UCI Robot Execution Failures .data file.
    Format: class_label, f1, f2, ..., fN (space/tab separated, multiple readings per class block)
    Each block = one episode (multiple timesteps).
    """
    episodes = []
    current_label = None
    current_steps = []
    ep_idx = 0

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        parts = line.split()
        if not parts:
            continue

        # If first token is a class label string
        first = parts[0]
        if first in UCI_CLASSES or any(c.isalpha() for c in first[:3]):
            # Save previous episode
            if current_steps and current_label is not None:
                episodes.append({
                    "label": current_label,
                    "steps": np.array(current_steps, dtype=np.float32),
                    "idx": ep_idx,
                })
                ep_idx += 1
                current_steps = []
            current_label = first.lower()
            # Rest of line may have first step data
            try:
                step_data = [float(x) for x in parts[1:]]
                if step_data:
                    current_steps.append(step_data)
            except ValueError:
                pass
        else:
            # Numeric step row
            try:
                step_data = [float(x) for x in parts]
                if step_data and current_label is not None:
                    current_steps.append(step_data)
            except ValueError:
                pass

    # Flush last episode
    if current_steps and current_label is not None:
        episodes.append({
            "label": current_label,
            "steps": np.array(current_steps, dtype=np.float32),
            "idx": ep_idx,
        })

    return episodes


def load_uci_failures(max_episodes: int = 500) -> tuple[list[dict], dict]:
    """
    Load UCI Robot Execution Failures dataset.
    Returns (episodes, access_report).
    Label type: human (gold standard tabular).
    """
    report = {
        "dataset": "uci_failures",
        "url": UCI_URL,
        "label_type": "human",
        "attempted": True,
        "success": False,
        "n_episodes": 0,
        "failure_reason": None,
        "caveats": [
            "Human-labeled failure classes — gold standard for tabular models",
            "Proprioceptive only (force/torque from PUMA-560 arm) — no video",
            "Small dataset: typically 88 episodes across 5 task files",
            "Sim-to-real: collected on real robot but 1980s-90s hardware",
            "Cross-dataset transfer to modern robots may be poor",
        ],
    }

    cache_path = CACHE_DIR / "uci_robot_failures.zip"
    data_bytes = _download_uci(cache_path)

    if data_bytes is None:
        # Try simple fallback URL
        for fallback_url in [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/robotfailure/",
        ]:
            try:
                print(f"[UCI] Trying fallback: {fallback_url}lp1.data")
                with urllib.request.urlopen(fallback_url + "lp1.data", timeout=15) as r:
                    text = r.read().decode("utf-8", errors="ignore")
                    raw_eps = _parse_uci_data(text, "lp1")
                    if raw_eps:
                        data_bytes = b"__direct__"
                        report["source"] = "direct file download"
                        # Build episodes from this single file
                        episodes = []
                        for raw_ep in raw_eps[:max_episodes]:
                            steps = raw_ep["steps"]
                            if steps.ndim == 1:
                                steps = steps.reshape(1, -1)
                            ep_label = "nominal" if raw_ep["label"] == "normal" else "failure"
                            episodes.append(make_episode(
                                dataset_name="uci_failures",
                                episode_id=f"uci_lp1_ep{raw_ep['idx']:03d}",
                                state_seq=steps,
                                episode_label=ep_label,
                                failure_category=raw_ep["label"] if ep_label == "failure" else None,
                                source_label_type="human",
                                metadata={"task": "lp1", "raw_label": raw_ep["label"]},
                            ))
                        if episodes:
                            report["success"] = True
                            report["n_episodes"] = len(episodes)
                            print(f"[UCI] Loaded {len(episodes)} episodes from direct download")
                            return episodes, report
                        break
            except Exception as ex:
                print(f"[UCI] Fallback failed: {ex}")

    if data_bytes is None:
        report["failure_reason"] = "Could not download UCI dataset from any source"
        print("[UCI] FAILED: could not download")
        return _load_uci_synthetic_fallback(max_episodes, report)

    # Parse zip
    episodes = []
    try:
        if data_bytes != b"__direct__":
            zf = zipfile.ZipFile(io.BytesIO(data_bytes))
            file_list = zf.namelist()
            data_files = [f for f in file_list if f.endswith(".data")]
            if not data_files:
                # Try looking for any text file
                data_files = [f for f in file_list if not f.endswith("/")]

            for fname in data_files:
                task_name = Path(fname).stem.lower()
                try:
                    text = zf.read(fname).decode("utf-8", errors="ignore")
                    raw_eps = _parse_uci_data(text, task_name)
                    for raw_ep in raw_eps:
                        if len(episodes) >= max_episodes:
                            break
                        steps = raw_ep["steps"]
                        if steps.ndim == 1:
                            steps = steps.reshape(1, -1)
                        ep_label = "nominal" if raw_ep["label"] == "normal" else "failure"
                        episodes.append(make_episode(
                            dataset_name="uci_failures",
                            episode_id=f"uci_{task_name}_ep{raw_ep['idx']:03d}",
                            state_seq=steps,
                            episode_label=ep_label,
                            failure_category=raw_ep["label"] if ep_label == "failure" else None,
                            source_label_type="human",
                            metadata={"task": task_name, "raw_label": raw_ep["label"]},
                        ))
                except Exception as fe:
                    print(f"[UCI] Error parsing {fname}: {fe}")
                    continue

    except Exception as e:
        report["failure_reason"] = f"ZIP parse error: {e}"
        print(f"[UCI] ZIP parse failed: {e}")
        return _load_uci_synthetic_fallback(max_episodes, report)

    if not episodes:
        return _load_uci_synthetic_fallback(max_episodes, report)

    report["success"] = True
    report["n_episodes"] = len(episodes)
    label_counts = {}
    for ep in episodes:
        lbl = ep["episode_label"] or "unknown"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    report["label_distribution"] = label_counts
    print(f"[UCI] Loaded {len(episodes)} episodes.")
    return episodes, report


def _load_uci_synthetic_fallback(max_episodes: int, report: dict) -> tuple[list[dict], dict]:
    """
    If UCI download fails, generate a realistic synthetic stand-in with the
    same class structure (6 classes: normal + 5 failure types).
    Clearly marked as synthetic in report and episode metadata.
    """
    print("[UCI] Using synthetic fallback (same class structure as UCI)")
    rng = np.random.RandomState(42)
    CLASSES = {
        "normal":             (0,   0.5),
        "collision":          (3.0, 1.5),
        "obstruction":        (2.0, 1.0),
        "fr_collision":       (3.5, 2.0),
        "back_col_obstacle":  (2.5, 1.2),
        "slipping":           (1.5, 0.8),
    }
    N_FEATURES = 6  # torque channels (matches UCI format)
    T_RANGE = (15, 30)

    episodes = []
    per_class = max_episodes // len(CLASSES)
    for cls_name, (mean_offset, noise_scale) in CLASSES.items():
        for i in range(per_class):
            T = rng.randint(*T_RANGE)
            base = rng.randn(T, N_FEATURES).astype(np.float32)
            state = base * noise_scale + mean_offset
            # Add class-specific patterns
            if cls_name == "collision":
                spike_t = rng.randint(T//2, T)
                state[spike_t:, 0] += rng.uniform(5, 10)
            elif cls_name == "slipping":
                state[:, 2] += np.sin(np.linspace(0, 4*np.pi, T)).astype(np.float32) * 2
            elif cls_name == "obstruction":
                state[T//2:, 1] = 0.01  # stuck joint

            ep_label = "nominal" if cls_name == "normal" else "failure"
            episodes.append(make_episode(
                dataset_name="uci_failures",
                episode_id=f"uci_synth_{cls_name}_ep{i:03d}",
                state_seq=state,
                episode_label=ep_label,
                failure_category=cls_name if cls_name != "normal" else None,
                source_label_type="synthetic",
                metadata={"synthetic": True, "class": cls_name,
                          "note": "UCI download failed; synthetic stand-in with same structure"},
            ))

    rng.shuffle(episodes)
    report["success"] = True
    report["n_episodes"] = len(episodes)
    report["synthetic_fallback"] = True
    report["caveats"].append("⚠ SYNTHETIC FALLBACK: UCI download failed — results on this data are NOT from real UCI labels")
    label_counts = {}
    for ep in episodes:
        lbl = ep["episode_label"] or "unknown"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    report["label_distribution"] = label_counts
    return episodes, report


# ─────────────────────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_datasets(max_episodes_per_dataset: int = 300) -> tuple[dict, dict]:
    """
    Load all 5 datasets. Returns:
        episodes:  dict[dataset_name -> list[episode_dict]]
        reports:   dict[dataset_name -> access_report_dict]
    """
    loaders = [
        ("botfails",       load_botfails),
        ("robofac",        load_robofac),
        ("vifailback",     load_vifailback),
        ("lerobot_reward", load_lerobot_reward),
        ("uci_failures",   load_uci_failures),
    ]

    all_episodes = {}
    all_reports  = {}

    for name, loader_fn in loaders:
        print(f"\n{'='*60}")
        print(f" Loading: {name}")
        print(f"{'='*60}")
        try:
            eps, rpt = loader_fn(max_episodes=max_episodes_per_dataset)
        except Exception as e:
            eps = []
            rpt = {"dataset": name, "success": False, "failure_reason": str(e), "n_episodes": 0}
            print(f"[{name}] Unexpected error: {e}")

        all_episodes[name] = eps
        all_reports[name]  = rpt

    # ── Save access reports ───────────────────────────────────────────────────
    report_path = OUT_DIR / "dataset_access_report.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n[Loaders] Access report saved → {report_path}")

    return all_episodes, all_reports


# ─────────────────────────────────────────────────────────────────────────────
# Utility: EDA summary
# ─────────────────────────────────────────────────────────────────────────────

def dataset_eda(episodes: list[dict], name: str) -> dict:
    """Generate basic EDA statistics for a loaded dataset."""
    if not episodes:
        return {"dataset": name, "n_episodes": 0}

    labels = [ep["episode_label"] or "unknown" for ep in episodes]
    label_counts = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    timesteps = [ep["timesteps"] for ep in episodes]
    dims = [ep["state_seq"].shape[-1] for ep in episodes]
    has_actions = sum(1 for ep in episodes if ep["action_seq"] is not None)
    has_steps   = sum(1 for ep in episodes if ep["step_labels"] is not None)
    has_video   = sum(1 for ep in episodes if ep["video_frames"] or ep["image_paths"])

    return {
        "dataset": name,
        "n_episodes": len(episodes),
        "label_distribution": label_counts,
        "label_type": episodes[0]["source_label_type"],
        "timesteps_mean": float(np.mean(timesteps)),
        "timesteps_min":  int(np.min(timesteps)),
        "timesteps_max":  int(np.max(timesteps)),
        "state_dim":      int(np.median(dims)),
        "has_actions_pct": round(has_actions / len(episodes) * 100, 1),
        "has_step_labels_pct": round(has_steps / len(episodes) * 100, 1),
        "has_video_pct":  round(has_video / len(episodes) * 100, 1),
        "class_balance": {
            k: round(v / len(episodes), 3) for k, v in label_counts.items()
        },
    }


if __name__ == "__main__":
    all_episodes, all_reports = load_all_datasets(max_episodes_per_dataset=300)
    print("\n\n=== DATASET SUMMARY ===")
    for name, eps in all_episodes.items():
        eda = dataset_eda(eps, name)
        print(f"\n{name}: {eda['n_episodes']} episodes, label_type={eda.get('label_type','?')}")
        print(f"  labels: {eda.get('label_distribution', {})}")
        print(f"  state_dim={eda.get('state_dim','?')}, T_mean={eda.get('timesteps_mean','?'):.1f}")
