"""
exporters.py — Haptal AI Robot Annotation Pipeline
====================================================

Converts annotated episode data produced by the Haptal annotation pipeline into
standard robot learning formats used by downstream training frameworks.

Supported export formats
------------------------
1. LeRobot HDF5 — the LeRobot dataset schema (Hugging Face robotics library).
2. ACT / Diffusion Policy HDF5 — the schema used by Action Chunking Transformer
   and Diffusion Policy training scripts.
3. RLDS manifest + loader stub — a JSON manifest and a companion Python file
   that shows how to wrap the data as a tf.data.Dataset when TensorFlow is
   available.

Typical usage
-------------
    from exporters import RobotDataExporter
    import numpy as np

    exporter = RobotDataExporter(output_dir=Path("exports"), quality_threshold=0.65)
    paths = exporter.export_all(results, state_seqs, output_name="run_001")

CLI
---
    python exporters.py --help
    python exporters.py --demo                    # run synthetic demo
    python exporters.py --format lerobot --output my_run
    python exporters.py --format act     --output my_run
    python exporters.py --format rlds    --output my_run
"""

from __future__ import annotations

import argparse
import json
import textwrap
import warnings
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

# ── Optional heavy imports ────────────────────────────────────────────────────

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    _H5PY_AVAILABLE = False
    warnings.warn(
        "h5py is not installed. LeRobot and ACT/Diffusion Policy HDF5 export "
        "will raise ImportError at call time. Install with: pip install h5py",
        ImportWarning,
        stacklevel=2,
    )

# TensorFlow is intentionally *not* imported here; the RLDS exporter only
# writes JSON + a plain Python stub file.

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("benchmark_output")

#: Full failure taxonomy shared with annotation_model.py.
#: The 10-class extended set covers additional classes produced by newer model
#: versions that clients may receive.
FAILURE_CLASSES: list[str] = [
    "nominal",
    "velocity_spike",
    "position_jerk",
    "stuck_joint",
    "gripper_event",
    "high_anomaly",
    "self_collision",
    "overshoot",
    "trajectory_deviation",
    "perception_failure",
]

# Default state dimensionality for xArm episodes (6-DOF joints + 1 gripper).
XARM_STATE_DIM = 4  # as used by the Haptal pipeline benchmark data


# ── Helper utilities ──────────────────────────────────────────────────────────


def run_length_encode(labels: list[str]) -> list[dict[str, Any]]:
    """Compress a flat list of string labels using run-length encoding.

    Consecutive identical labels are collapsed into a single record that stores
    the label and the run length.  This significantly reduces storage for
    annotation manifests where most steps are ``"nominal"``.

    Parameters
    ----------
    labels:
        Flat list of string labels, e.g. per-step failure annotations.

    Returns
    -------
    list of dicts
        Each dict has the keys ``"label"`` (str) and ``"count"`` (int).

    Examples
    --------
    >>> run_length_encode(["nominal", "nominal", "nominal", "velocity_spike", "nominal"])
    [{'label': 'nominal', 'count': 3}, {'label': 'velocity_spike', 'count': 1},
     {'label': 'nominal', 'count': 1}]

    >>> run_length_encode([])
    []
    """
    if not labels:
        return []
    return [
        {"label": key, "count": sum(1 for _ in group)}
        for key, group in groupby(labels)
    ]


def _require_h5py(caller: str) -> None:
    """Raise a helpful ImportError if h5py is unavailable."""
    if not _H5PY_AVAILABLE:
        raise ImportError(
            f"{caller} requires h5py. Install it with: pip install h5py"
        )


def _validate_result(result: dict) -> None:
    """Lightly validate that a result dict has the expected top-level keys."""
    required = {"episode_id", "n_steps", "anomaly_score", "quality_score",
                "failure_annotation"}
    missing = required - result.keys()
    if missing:
        raise ValueError(
            f"Result dict for episode '{result.get('episode_id', '?')}' is "
            f"missing required keys: {missing}"
        )


def _get_state(result: dict, state_seqs: dict) -> np.ndarray:
    """Return the (T, D) state array for *result*, raising clearly if absent."""
    ep_id = result["episode_id"]
    if ep_id not in state_seqs:
        raise KeyError(
            f"state_seqs does not contain an entry for episode '{ep_id}'. "
            "Provide a state_seqs dict mapping every episode_id to its "
            "np.ndarray of shape (T, D)."
        )
    arr = np.asarray(state_seqs[ep_id], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"state_seqs['{ep_id}'] must be 2-D (T, D), got shape {arr.shape}."
        )
    return arr


# ── Main exporter class ───────────────────────────────────────────────────────


class RobotDataExporter:
    """Export annotated episode data into standard robot learning formats.

    Parameters
    ----------
    output_dir:
        Directory where all exported files are written.  Created automatically
        if it does not exist.
    quality_threshold:
        Minimum ``quality_score`` an episode must have to be included in
        quality-filtered exports (currently ACT / Diffusion Policy).
        Episodes below this threshold are silently skipped with a warning.
    """

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        quality_threshold: float = 0.65,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_threshold = quality_threshold

    # ------------------------------------------------------------------
    # 1. LeRobot HDF5
    # ------------------------------------------------------------------

    def export_lerobot(
        self,
        results: list[dict],
        state_seqs: dict[str, np.ndarray],
        output_name: str,
    ) -> Path:
        """Export annotated episodes to LeRobot HDF5 format.

        The output file follows the LeRobot dataset schema: all episodes are
        concatenated into flat arrays along the time axis, and per-step
        metadata is stored as parallel datasets.

        Datasets written
        ~~~~~~~~~~~~~~~~
        ``observation.state``   — (T_total, D) float32 joint states
        ``action``              — (T_total, D) float32 (copy of state)
        ``episode_index``       — (T_total,) int64 episode IDs (0-based)
        ``frame_index``         — (T_total,) int64 0..T-1 within each episode
        ``timestamp``           — (T_total,) float32 fake timestamps (step index)
        ``next.done``           — (T_total,) bool True only at last step
        ``haptal/failure_label``— (T_total,) bytes per-step failure class
        ``haptal/confidence``   — (T_total,) float32 per-step confidence
        ``haptal/needs_review`` — (T_total,) bool per-step review flag
        ``haptal/anomaly_score``— (T_total,) float32 episode anomaly score broadcast
        ``haptal/quality_score``— (T_total,) float32 episode quality score broadcast

        Parameters
        ----------
        results:
            List of annotation result dicts (all episodes to export).
        state_seqs:
            Mapping from ``episode_id`` to (T, D) numpy array of joint states.
        output_name:
            Base name for the output file (without extension).

        Returns
        -------
        Path
            Absolute path to the written ``.hdf5`` file.
        """
        _require_h5py("export_lerobot")

        out_path = self.output_dir / f"{output_name}_lerobot.hdf5"

        # ── Accumulate arrays across all episodes ──────────────────────────
        states_all: list[np.ndarray] = []
        ep_indices: list[np.ndarray] = []
        frame_indices: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        done_flags: list[np.ndarray] = []
        failure_labels: list[list[bytes]] = []
        confidences: list[np.ndarray] = []
        needs_review: list[np.ndarray] = []
        anomaly_scores: list[np.ndarray] = []
        quality_scores: list[np.ndarray] = []

        for ep_idx, result in enumerate(results):
            _validate_result(result)
            state = _get_state(result, state_seqs)
            T = state.shape[0]
            fa = result["failure_annotation"]

            # Sanity-check lengths
            if len(fa["step_labels"]) != T:
                warnings.warn(
                    f"Episode '{result['episode_id']}': step_labels length "
                    f"({len(fa['step_labels'])}) != state length ({T}). "
                    "Truncating/padding to match state length.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            step_labels = fa["step_labels"][:T]
            step_confs = np.array(fa["step_confs"][:T], dtype=np.float32)
            step_needs_review = np.array(fa["needs_review"][:T], dtype=bool)

            # Build parallel arrays for this episode
            states_all.append(state)
            ep_indices.append(np.full(T, ep_idx, dtype=np.int64))
            frame_indices.append(np.arange(T, dtype=np.int64))
            timestamps.append(np.arange(T, dtype=np.float32))

            done = np.zeros(T, dtype=bool)
            done[-1] = True
            done_flags.append(done)

            failure_labels.append([lbl.encode("utf-8") for lbl in step_labels])
            confidences.append(step_confs)
            needs_review.append(step_needs_review)
            anomaly_scores.append(
                np.full(T, result["anomaly_score"], dtype=np.float32)
            )
            quality_scores.append(
                np.full(T, result["quality_score"], dtype=np.float32)
            )

        if not states_all:
            raise ValueError("results is empty; nothing to export.")

        # ── Concatenate and write ──────────────────────────────────────────
        state_cat = np.concatenate(states_all, axis=0)
        T_total = state_cat.shape[0]

        # Flatten label list
        labels_flat: list[bytes] = []
        for ep_lbls in failure_labels:
            labels_flat.extend(ep_lbls)
        labels_np = np.array(labels_flat, dtype=object)  # variable-length bytes

        with h5py.File(out_path, "w") as f:
            f.attrs["format"] = "lerobot"
            f.attrs["haptal_version"] = "1.0"
            f.attrs["n_episodes"] = len(results)
            f.attrs["total_steps"] = T_total

            f.create_dataset("observation.state",    data=state_cat)
            f.create_dataset("action",               data=state_cat.copy())
            f.create_dataset("episode_index",        data=np.concatenate(ep_indices))
            f.create_dataset("frame_index",          data=np.concatenate(frame_indices))
            f.create_dataset("timestamp",            data=np.concatenate(timestamps))
            f.create_dataset("next.done",            data=np.concatenate(done_flags))

            # Haptal annotation group
            hg = f.create_group("haptal")
            # Variable-length UTF-8 strings encoded as bytes
            str_dtype = h5py.special_dtype(vlen=bytes)
            hg.create_dataset(
                "failure_label",
                data=labels_np,
                dtype=str_dtype,
            )
            hg.create_dataset("confidence",    data=np.concatenate(confidences))
            hg.create_dataset("needs_review",  data=np.concatenate(needs_review))
            hg.create_dataset("anomaly_score", data=np.concatenate(anomaly_scores))
            hg.create_dataset("quality_score", data=np.concatenate(quality_scores))

            # Per-episode metadata table for quick lookup without scanning arrays
            ep_grp = f.create_group("episodes")
            for ep_idx, result in enumerate(results):
                eg = ep_grp.create_group(f"ep_{ep_idx:05d}")
                eg.attrs["episode_id"]       = result["episode_id"]
                eg.attrs["n_steps"]          = result["n_steps"]
                eg.attrs["anomaly_score"]    = float(result["anomaly_score"])
                eg.attrs["quality_score"]    = float(result["quality_score"])
                eg.attrs["flagged"]          = bool(result.get("flagged", False))
                eg.attrs["dominant_failure"] = result["failure_annotation"]["dominant"]

        print(f"[LeRobot] Wrote {len(results)} episodes ({T_total} steps) → {out_path}")
        return out_path.resolve()

    # ------------------------------------------------------------------
    # 2. ACT / Diffusion Policy HDF5
    # ------------------------------------------------------------------

    def export_act(
        self,
        results: list[dict],
        state_seqs: dict[str, np.ndarray],
        output_name: str,
    ) -> Path:
        """Export annotated episodes to ACT / Diffusion Policy HDF5 format.

        Only episodes with ``quality_score >= self.quality_threshold`` are
        included.  Episodes below the threshold are logged as warnings.

        The output follows the ``robomimic`` / ACT demo convention where each
        demonstration is stored under ``data/demo_N/``.

        Datasets written per demo
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        ``data/demo_N/obs/qpos``       — (T, D) float32 joint positions
        ``data/demo_N/obs/qvel``       — (T, D) float32 velocities (finite diff)
        ``data/demo_N/actions``        — (T, D) float32 (copy of qpos)
        ``data/demo_N/haptal_labels``  — (T,) bytes failure class per step
        ``data/demo_N/haptal_quality`` — scalar float32 episode quality score

        Top-level attributes record the total number of demos and steps.

        Parameters
        ----------
        results:
            List of annotation result dicts.
        state_seqs:
            Mapping from ``episode_id`` to (T, D) numpy array.
        output_name:
            Base name for the output file (without extension).

        Returns
        -------
        Path
            Absolute path to the written ``.hdf5`` file.
        """
        _require_h5py("export_act")

        out_path = self.output_dir / f"{output_name}_act.hdf5"

        kept: list[dict] = []
        skipped: list[str] = []
        for result in results:
            _validate_result(result)
            if result["quality_score"] >= self.quality_threshold:
                kept.append(result)
            else:
                skipped.append(result["episode_id"])

        if skipped:
            warnings.warn(
                f"[ACT export] Skipped {len(skipped)} episode(s) below quality "
                f"threshold {self.quality_threshold}: {skipped}",
                UserWarning,
                stacklevel=2,
            )

        if not kept:
            raise ValueError(
                f"No episodes meet quality_threshold={self.quality_threshold}. "
                "Lower the threshold or check your annotation results."
            )

        total_steps = 0
        with h5py.File(out_path, "w") as f:
            f.attrs["format"] = "act_diffusion_policy"
            f.attrs["haptal_version"] = "1.0"
            f.attrs["quality_threshold"] = self.quality_threshold

            data_grp = f.create_group("data")
            str_dtype = h5py.special_dtype(vlen=bytes)

            for demo_idx, result in enumerate(kept):
                state = _get_state(result, state_seqs)
                T = state.shape[0]
                fa = result["failure_annotation"]

                # Finite-difference velocities; pad first step with zeros
                qvel = np.zeros_like(state)
                qvel[1:] = state[1:] - state[:-1]

                step_labels = fa["step_labels"][:T]
                labels_bytes = np.array(
                    [lbl.encode("utf-8") for lbl in step_labels], dtype=object
                )

                demo_grp = data_grp.create_group(f"demo_{demo_idx}")
                demo_grp.attrs["episode_id"]       = result["episode_id"]
                demo_grp.attrs["n_steps"]          = T
                demo_grp.attrs["dominant_failure"] = fa["dominant"]
                demo_grp.attrs["flagged"]          = bool(result.get("flagged", False))

                obs_grp = demo_grp.create_group("obs")
                obs_grp.create_dataset("qpos", data=state)
                obs_grp.create_dataset("qvel", data=qvel)

                demo_grp.create_dataset("actions", data=state.copy())
                demo_grp.create_dataset(
                    "haptal_labels", data=labels_bytes, dtype=str_dtype
                )
                demo_grp.create_dataset(
                    "haptal_quality",
                    data=np.float32(result["quality_score"]),
                )

                total_steps += T

            f.attrs["n_demos"] = len(kept)
            f.attrs["total_steps"] = total_steps

        print(
            f"[ACT] Wrote {len(kept)} demos ({total_steps} steps) → {out_path} "
            f"(skipped {len(skipped)} below threshold)"
        )
        return out_path.resolve()

    # ------------------------------------------------------------------
    # 3. RLDS manifest + loader stub
    # ------------------------------------------------------------------

    def export_rlds_manifest(
        self,
        results: list[dict],
        output_name: str,
    ) -> Path:
        """Write an RLDS-compatible manifest JSON and a TF loader stub.

        Because TensorFlow may not be installed, this exporter writes two
        plain files:

        ``<output_name>_rlds_manifest.json``
            A machine-readable catalogue of all episodes with per-episode
            metadata.  Step labels are stored as run-length–encoded sequences
            to keep file sizes small.

        ``<output_name>_rlds_loader.py``
            A standalone Python module with a ``build_rlds_dataset()`` function
            that turns the manifest + state arrays into a real
            ``tf.data.Dataset`` of RLDS episodes when TensorFlow is present.

        Manifest schema (per episode)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``episode_id``        — str
        ``n_steps``           — int
        ``quality_score``     — float
        ``anomaly_score``     — float
        ``flagged``           — bool
        ``dominant_failure``  — str
        ``step_labels_rle``   — list of {"label": str, "count": int}

        Parameters
        ----------
        results:
            List of annotation result dicts.
        output_name:
            Base name for the output files (without extension).

        Returns
        -------
        Path
            Absolute path to the written ``_rlds_manifest.json`` file.
        """
        manifest_path = self.output_dir / f"{output_name}_rlds_manifest.json"
        stub_path = self.output_dir / f"{output_name}_rlds_loader.py"

        episodes_meta: list[dict] = []
        for result in results:
            _validate_result(result)
            fa = result["failure_annotation"]
            rle = run_length_encode(fa["step_labels"])
            episodes_meta.append(
                {
                    "episode_id":       result["episode_id"],
                    "n_steps":          result["n_steps"],
                    "quality_score":    float(result["quality_score"]),
                    "anomaly_score":    float(result["anomaly_score"]),
                    "flagged":          bool(result.get("flagged", False)),
                    "dominant_failure": fa["dominant"],
                    "step_labels_rle":  rle,
                }
            )

        manifest = {
            "format":          "rlds_haptal_manifest",
            "haptal_version":  "1.0",
            "n_episodes":      len(episodes_meta),
            "failure_classes": FAILURE_CLASSES,
            "episodes":        episodes_meta,
        }

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        # ── Write companion loader stub ────────────────────────────────────
        stub_code = textwrap.dedent(f"""\
            \"\"\"
            RLDS dataset loader stub — generated by Haptal AI exporters.py
            ================================================================
            This file shows how to load the manifest produced by
            RobotDataExporter.export_rlds_manifest() as a real tf.data.Dataset
            of RLDS episodes when TensorFlow is available.

            Requirements
            ------------
                pip install tensorflow tensorflow-datasets

            Usage
            -----
                from {output_name}_rlds_loader import build_rlds_dataset
                import numpy as np

                # state_seqs: dict[episode_id, np.ndarray of shape (T, D)]
                ds = build_rlds_dataset("path/to/{output_name}_rlds_manifest.json",
                                        state_seqs)
                for episode in ds:
                    steps = list(episode["steps"])
                    print(steps[0]["observation"]["state"])
            \"\"\"

            from __future__ import annotations
            import json
            from itertools import repeat
            from pathlib import Path
            from typing import Iterator

            import numpy as np


            def _rle_decode(rle: list[dict]) -> list[str]:
                \"\"\"Expand a run-length–encoded label list back to a flat list.\"\"\"
                labels: list[str] = []
                for seg in rle:
                    labels.extend(repeat(seg["label"], seg["count"]))
                return labels


            def _episode_generator(
                manifest: dict,
                state_seqs: dict[str, np.ndarray],
            ) -> Iterator[dict]:
                \"\"\"Yield one RLDS-style episode dict per entry in *manifest*.\"\"\"
                for ep_meta in manifest["episodes"]:
                    ep_id = ep_meta["episode_id"]
                    state = np.asarray(state_seqs[ep_id], dtype=np.float32)
                    T = state.shape[0]

                    labels = _rle_decode(ep_meta["step_labels_rle"])
                    labels = (labels + ["nominal"] * T)[:T]  # ensure length T

                    steps = []
                    for t in range(T):
                        steps.append({{
                            "observation": {{
                                "state": state[t],
                            }},
                            "action":    state[t],
                            "reward":    np.float32(0.0),
                            "is_terminal": np.bool_(t == T - 1),
                            "haptal": {{
                                "failure_label":  labels[t].encode("utf-8"),
                                "anomaly_score":  np.float32(ep_meta["anomaly_score"]),
                                "quality_score":  np.float32(ep_meta["quality_score"]),
                            }},
                        }})

                    yield {{
                        "episode_metadata": {{
                            "episode_id":       ep_id,
                            "n_steps":          np.int64(T),
                            "quality_score":    np.float32(ep_meta["quality_score"]),
                            "dominant_failure": ep_meta["dominant_failure"].encode("utf-8"),
                        }},
                        "steps": steps,
                    }}


            def build_rlds_dataset(
                manifest_path: str | Path,
                state_seqs: dict[str, np.ndarray],
            ):
                \"\"\"Build a ``tf.data.Dataset`` of RLDS episodes from a Haptal manifest.

                Parameters
                ----------
                manifest_path:
                    Path to the ``*_rlds_manifest.json`` produced by
                    ``RobotDataExporter.export_rlds_manifest()``.
                state_seqs:
                    Mapping from episode_id to (T, D) numpy array of joint states.

                Returns
                -------
                tf.data.Dataset
                    Each element is a dict with keys ``episode_metadata`` and
                    ``steps``.  The dataset mirrors the RLDS episode structure
                    used by tensorflow_datasets robotics datasets.
                \"\"\"
                try:
                    import tensorflow as tf
                except ImportError as exc:
                    raise ImportError(
                        "TensorFlow is required to build an RLDS dataset. "
                        "Install it with: pip install tensorflow"
                    ) from exc

                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json.load(fh)

                def generator():
                    yield from _episode_generator(manifest, state_seqs)

                # Infer output signature from the first episode
                first = next(_episode_generator(manifest, state_seqs))
                state_dim = next(iter(state_seqs.values())).shape[1]

                step_spec = {{
                    "observation": {{
                        "state": tf.TensorSpec(shape=(state_dim,), dtype=tf.float32),
                    }},
                    "action":       tf.TensorSpec(shape=(state_dim,), dtype=tf.float32),
                    "reward":       tf.TensorSpec(shape=(),           dtype=tf.float32),
                    "is_terminal":  tf.TensorSpec(shape=(),           dtype=tf.bool),
                    "haptal": {{
                        "failure_label":  tf.TensorSpec(shape=(), dtype=tf.string),
                        "anomaly_score":  tf.TensorSpec(shape=(), dtype=tf.float32),
                        "quality_score":  tf.TensorSpec(shape=(), dtype=tf.float32),
                    }},
                }}

                output_signature = {{
                    "episode_metadata": {{
                        "episode_id":       tf.TensorSpec(shape=(), dtype=tf.string),
                        "n_steps":          tf.TensorSpec(shape=(), dtype=tf.int64),
                        "quality_score":    tf.TensorSpec(shape=(), dtype=tf.float32),
                        "dominant_failure": tf.TensorSpec(shape=(), dtype=tf.string),
                    }},
                    "steps": tf.RaggedTensorSpec(
                        shape=(None,), dtype=tf.variant, ragged_rank=1
                    ),
                }}

                # Build a flat-steps dataset then re-group into episodes.
                # This avoids nested-dataset complexity for most use cases.
                episodes = list(_episode_generator(manifest, state_seqs))
                steps_flat = [step for ep in episodes for step in ep["steps"]]

                steps_ds = tf.data.Dataset.from_generator(
                    lambda: iter(steps_flat),
                    output_signature=step_spec,
                )
                return steps_ds  # callers can window/batch as needed
        """)

        with open(stub_path, "w", encoding="utf-8") as fh:
            fh.write(stub_code)

        print(
            f"[RLDS] Wrote manifest ({len(episodes_meta)} episodes) → {manifest_path}"
        )
        print(f"[RLDS] Wrote loader stub → {stub_path}")
        return manifest_path.resolve()

    # ------------------------------------------------------------------
    # 4. Convenience: run all exporters
    # ------------------------------------------------------------------

    def export_all(
        self,
        results: list[dict],
        state_seqs: dict[str, np.ndarray],
        output_name: str,
    ) -> dict[str, Path]:
        """Run all three exporters and return a mapping of format → output path.

        Parameters
        ----------
        results:
            List of annotation result dicts.
        state_seqs:
            Mapping from episode_id to (T, D) numpy array of joint states.
        output_name:
            Base name used for all output files.

        Returns
        -------
        dict
            Keys are ``"lerobot"``, ``"act"``, ``"rlds"``; values are the
            resolved ``Path`` objects of each output file.  If an exporter
            raises (e.g. h5py not installed), that key maps to ``None`` and a
            warning is emitted — the remaining exporters still run.
        """
        outputs: dict[str, Path | None] = {}

        for fmt, fn in [
            ("lerobot", lambda: self.export_lerobot(results, state_seqs, output_name)),
            ("act",     lambda: self.export_act(results, state_seqs, output_name)),
            ("rlds",    lambda: self.export_rlds_manifest(results, output_name)),
        ]:
            try:
                outputs[fmt] = fn()
            except Exception as exc:
                warnings.warn(
                    f"[export_all] '{fmt}' exporter failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                outputs[fmt] = None

        return outputs


# ── Synthetic demo ────────────────────────────────────────────────────────────


def _make_synthetic_data(
    n_episodes: int = 4,
    T: int = 120,
    D: int = XARM_STATE_DIM,
    seed: int = 0,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Generate synthetic annotation results and state sequences for testing.

    Parameters
    ----------
    n_episodes:
        Number of episodes to generate.
    T:
        Number of timesteps per episode.
    D:
        State dimensionality (joints).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    results:
        List of annotation result dicts in the Haptal pipeline output format.
    state_seqs:
        Dict mapping episode_id to (T, D) float32 array.
    """
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    state_seqs: dict[str, np.ndarray] = {}

    failure_pool = FAILURE_CLASSES[:6]  # use the original 6 for synthetic data

    for i in range(n_episodes):
        ep_id = f"ep_{i:03d}"
        state = rng.standard_normal((T, D)).astype(np.float32) * 0.1

        # Inject a short failure burst in ~half the episodes
        dominant = "nominal"
        if i % 2 == 1:
            burst_start = T // 3
            burst_len = T // 10
            state[burst_start: burst_start + burst_len] += rng.standard_normal(
                (burst_len, D)
            ).astype(np.float32) * 0.8
            dominant = failure_pool[1 + (i % (len(failure_pool) - 1))]

        step_labels = ["nominal"] * T
        step_confs = rng.uniform(0.7, 1.0, T).astype(float).tolist()
        step_scores = rng.uniform(0.0, 0.3, T).astype(float).tolist()
        needs_review = [False] * T

        if dominant != "nominal":
            burst_start = T // 3
            burst_len = T // 10
            for t in range(burst_start, min(burst_start + burst_len, T)):
                step_labels[t] = dominant
                step_confs[t] = float(rng.uniform(0.6, 0.9))
                step_scores[t] = float(rng.uniform(0.5, 0.9))
                needs_review[t] = True

        quality = float(rng.uniform(0.55, 0.95))
        anomaly = float(rng.uniform(0.05, 0.45))

        counts: dict[str, int] = {}
        for lbl in step_labels:
            counts[lbl] = counts.get(lbl, 0) + 1

        results.append(
            {
                "episode_id": ep_id,
                "n_steps": T,
                "anomaly_score": anomaly,
                "flagged": anomaly > 0.35,
                "quality_score": quality,
                "failure_annotation": {
                    "step_labels": step_labels,
                    "step_confs": step_confs,
                    "step_scores": step_scores,
                    "needs_review": needs_review,
                    "dominant": dominant,
                    "counts": counts,
                },
                "coords_3d": rng.standard_normal((T, 3)).tolist(),
            }
        )
        state_seqs[ep_id] = state

    return results, state_seqs


def _run_demo(output_dir: Path = Path("benchmark_output/export_demo")) -> None:
    """Run a quick end-to-end demo with synthetic data."""
    print("=" * 60)
    print("Haptal AI — RobotDataExporter synthetic demo")
    print("=" * 60)

    results, state_seqs = _make_synthetic_data(n_episodes=4, T=120)
    print(f"\nGenerated {len(results)} synthetic episodes, "
          f"{results[0]['n_steps']} steps each, D={XARM_STATE_DIM}\n")

    exporter = RobotDataExporter(output_dir=output_dir, quality_threshold=0.65)
    paths = exporter.export_all(results, state_seqs, output_name="demo_run")

    print("\nExport summary:")
    for fmt, path in paths.items():
        status = str(path) if path else "FAILED (see warnings)"
        print(f"  {fmt:<10} {status}")

    # Demo: run_length_encode
    sample_labels = results[0]["failure_annotation"]["step_labels"]
    rle = run_length_encode(sample_labels)
    print(f"\nRLE demo (first episode, {len(sample_labels)} steps → "
          f"{len(rle)} segments):")
    for seg in rle[:6]:
        print(f"  {seg}")
    if len(rle) > 6:
        print(f"  ... ({len(rle) - 6} more segments)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exporters.py",
        description=textwrap.dedent("""\
            Haptal AI — Robot Annotation Pipeline Exporters
            ------------------------------------------------
            Convert annotated episode data into standard robot learning formats.

            Available formats
              lerobot   LeRobot HDF5 (Hugging Face robotics schema)
              act       ACT / Diffusion Policy HDF5 (robomimic demo schema)
              rlds      RLDS JSON manifest + TensorFlow loader stub

            Typical workflow
              1. Run the Haptal annotation pipeline to get 'results' and
                 'state_seqs' (see pipeline.py / annotate.py).
              2. Import RobotDataExporter and call the desired export method,
                 or use --demo to test with synthetic data.

            Examples
              python exporters.py --demo
              python exporters.py --demo --output-dir /tmp/haptal_exports
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick end-to-end demo with synthetic data and exit.",
    )
    parser.add_argument(
        "--format",
        choices=["lerobot", "act", "rlds", "all"],
        default="all",
        help="Which format to export (default: all). Only used with --demo.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_output/export_demo",
        metavar="DIR",
        help="Directory to write exported files into (default: benchmark_output/export_demo).",
    )
    parser.add_argument(
        "--output",
        default="demo_run",
        metavar="NAME",
        help="Base name for output files (default: demo_run).",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.65,
        metavar="FLOAT",
        help="Minimum quality score for ACT export (default: 0.65).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=4,
        metavar="N",
        help="Number of synthetic episodes to generate for --demo (default: 4).",
    )
    return parser


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.demo:
        _run_demo(output_dir=Path(args.output_dir))
    else:
        # When called without --demo, print help — the script is primarily a
        # library; real usage goes through RobotDataExporter in Python code.
        parser.print_help()
