# Haptal Engine — Product & Training Plan

> **Document status:** Living plan. Committed before any training code is written.  
> **Date:** 2026-05-10  
> **Author:** Haptal Engineering (generated with Claude)

---

## Table of Contents

1. [Current Repo Architecture](#1-current-repo-architecture)
2. [Current Model & Data Flow](#2-current-model--data-flow)
3. [Datasets Selected](#3-datasets-selected)
4. [Label Quality Caveats](#4-label-quality-caveats)
5. [Common Internal Schema](#5-common-internal-schema)
6. [Training & Evaluation Plan](#6-training--evaluation-plan)
7. [Metrics to Report](#7-metrics-to-report)
8. [Score Definitions & Thresholding](#8-score-definitions--thresholding)
9. [Alternative Training Mechanisms](#9-alternative-training-mechanisms)
10. [Visual / Video Model Architecture](#10-visual--video-model-architecture)
11. [Productized Architecture Proposals](#11-productized-architecture-proposals)
12. [Customer Onboarding & Data Flow](#12-customer-onboarding--data-flow)
13. [Deliverables Checklist](#13-deliverables-checklist)

---

## 1. Current Repo Architecture

```
benchmarking/
├── pipeline.py              # Main client-facing pipeline (IsolationForest + annotators)
├── annotation_model.py      # Step-level RobotAnnotator (RF, 6-class weak-supervision)
├── semantic_annotator.py    # SemanticAnnotator (4-layer labels: phase/zone/contact/motion)
├── models.py                # IsolationForestModel + LSTMAEModel base classes
├── lstm_annotator.py        # LSTM autoencoder for temporal anomaly detection
├── active_learning.py       # Active-learning query strategies (entropy, margin, etc.)
├── feedback_loop.py         # Human correction ingestion + model retraining trigger
├── exporters.py             # Export to LeRobot/ACT/RLDS/HDF5 formats
├── preprocessing.py         # Feature extraction, sliding windows, normalisation
├── augmentation.py          # Data augmentation for imbalanced failure classes
├── benchmark/
│   ├── failure_injector.py  # Physics-based synthetic failure injection
│   ├── evaluate.py          # Benchmark runner (internal synthetic benchmark)
│   ├── data/metadata.json   # Synthetic benchmark metadata (1,800 episodes)
│   └── results.json         # Current best results (RF 93.6% accuracy)
├── benchmark_output/        # All run artifacts and client-specific outputs
├── robot_viewer/            # FastAPI + Three.js SPA for human-in-the-loop review
│   ├── server.py            # REST API (episodes, reviews, datasets)
│   └── static/index.html   # Full SPA (~1,600 lines)
└── requirements.txt
```

### Key existing capabilities
| Capability | Implementation | Status |
|---|---|---|
| Episode anomaly scoring | IsolationForest on 204-dim features (mean+std+max of 68 step features) | ✅ Production |
| Step failure classification | RandomForestClassifier, 6 classes, weak labels | ✅ Production |
| Semantic labelling | 4-layer per-step labels trained on DROID+ALOHA | ✅ Production |
| LSTM autoencoder | Reconstruction-loss anomaly scoring | ✅ Prototype |
| Active learning | Entropy/margin/core-set query strategies | ✅ Prototype |
| Human feedback loop | Correction ingestion, weight boosting (10×), retraining trigger | ✅ Prototype |
| Export | LeRobot HDF5, ACT HDF5, RLDS JSON, custom HDF5 | ✅ Production |
| HITL review UI | Three.js 3D replay, accept/reject/flag, live log, failure cards | ✅ Production |
| Synthetic benchmark | 1,800 episodes, 6 classes, physics injection, 93.6% RF accuracy | ✅ Done |

---

## 2. Current Model & Data Flow

```
Customer HDF5 (states/actions/rewards)
        │
        ▼
  preprocessing.py  ─── sliding window (W=10) ──► 68 step features
        │                                           (pos Δ, vel, acc, jerk,
        │                                            gripper, angular vel …)
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 1 — Episode anomaly (IsolationForest)            │
  │  • Fit on SOP/nominal reference HDF5                    │
  │  • Score each episode: anomaly_score ∈ [0,1]            │
  │  • threshold @ 75th percentile of SOP scores            │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 2 — Step failure type (RobotAnnotator / RF)      │
  │  • Weak labels from velocity/jerk/stuck-joint rules     │
  │  • 6-class per-step: nominal/vel_spike/pos_jerk/        │
  │    stuck_joint/gripper_event/high_anomaly               │
  │  • confidence ∈ [0,1] from predict_proba                │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 3 — Semantic labels (SemanticAnnotator)          │
  │  • 4 axes: task_phase / workspace_zone /                │
  │    contact_state / motion_type                          │
  │  • Trained on 177k steps (DROID + ALOHA, 4 datasets)   │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  Review queue  ──► HITL review (robot_viewer SPA)
        │
        ▼
  Human corrections ──► feedback_loop.py ──► retrain
        │
        ▼
  Annotated export (LeRobot / ACT / RLDS / HDF5)
```

**Existing benchmark results (synthetic data):**
- RandomForest (204-dim, n=300, depth=20): accuracy 93.6%, macro-F1 93.7%, Cohen κ 0.923
- IsolationForest on real xarm_lift: ROC-AUC 0.943, detection rate 92.2%
- IsolationForest on real xarm_push: ROC-AUC 0.871, detection rate 55.0%

---

## 3. Datasets Selected

Five public datasets are selected in priority order. Access, format, and size are assessed during implementation; fallbacks are documented.

### Dataset 1 — BotFails (PRIMARY)
| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/kantine/BotFails |
| Format | LeRobot Parquet + HDF5 |
| Content | Real robot manipulation failures, episode-level annotations |
| Label type | **Human-labeled episode-level failure/success** (LeRobot-style) |
| Why selected | Only dataset in list with direct human-labeled failure labels |
| Expected size | Small–medium (evaluate after download) |
| Fallback | UCI Robot Execution Failures if access blocked |

### Dataset 2 — RoboFAC
| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset |
| Format | Video + QA JSON annotations |
| Content | Robot failure analysis and correction videos with QA labels |
| Label type | **VQA/correction labels** (human-written QA about failure causes) |
| Why selected | Unique failure-cause explanation labels; complements step-level detection |
| Expected size | Medium (video data) |
| Fallback | ViFailback if format blocked |

### Dataset 3 — ViFailback
| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset |
| Format | Video frames + trajectory JSON |
| Content | Manipulation failure trajectories, visual failure + correction data |
| Label type | **Heuristic + human failure/correction labels** |
| Why selected | Paired failure/correction structure enables contrastive training |
| Fallback | LiRAnomaly or LeRobot reward-derived |

### Dataset 4 — LeRobot reward-derived (DROID/xarm)
| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/lerobot/droid_100 and lerobot/xarm_lift_medium_replay |
| Format | LeRobot HDF5 (already partially cached in benchmark_output/) |
| Content | Manipulation trajectories with reward/success signals |
| Label type | **Reward-derived episode labels** (success=1 if reward > threshold) |
| Why selected | Already partially integrated in repo; provides nominal/failure split baseline |
| Notes | reward signal ≠ explicit failure label; threshold choice affects class balance |

### Dataset 5 — UCI Robot Execution Failures
| Field | Value |
|---|---|
| URL | https://archive.ics.uci.edu/dataset/138/robot+execution+failures |
| Format | CSV, tabular proprioceptive (torque/force sensors) |
| Content | 5 failure classes + nominal from a PUMA-560 robot |
| Label type | **Human-labeled failure classes** (gold standard tabular) |
| Why selected | Clean human labels, small size, ideal for cross-dataset generalization test |
| Notes | Proprioceptive only; no video; useful for pure tabular model comparison |

### Fallback Dataset — Haptal Synthetic Benchmark
If any of the above cannot be loaded, the internal synthetic benchmark (1,800 episodes, `benchmark/data/`) is used as a stand-in, clearly labeled as synthetic injected.

---

## 4. Label Quality Caveats

This section is **critical for honest evaluation**. Every metric in this project must be annotated with its label source.

### Label Type Taxonomy

| Label Type | Symbol | Description | Reliability |
|---|---|---|---|
| Human-labeled failure | 🟢 H | A human explicitly labeled each episode/step as failure or success | Highest — gold standard |
| VQA / correction label | 🔵 V | Human wrote a natural-language Q&A or correction about a failure | High — but category ambiguity |
| Reward-derived label | 🟡 R | `success = reward[-1] > threshold`; no explicit failure annotation | Medium — threshold-sensitive |
| Heuristic / weak label | 🟠 W | Rule-based: velocity spike > 3σ, joint stuck for N steps, etc. | Medium-low — noisy, systematic bias |
| Synthetic injected label | 🔴 S | Failure artificially injected into a nominal trajectory | Low for real-world transfer — controlled but not real |
| Semantic label | 🔷 Se | Phase/zone/contact/motion type labeled by model or coarse annotation | Moderate — model-in-the-loop |

### Per-Dataset Label Quality

| Dataset | Episode Label | Step Label | Label Type |
|---|---|---|---|
| BotFails | ✅ failure/success | ❓ unknown until inspected | 🟢 H |
| RoboFAC | ✅ QA about failure | ❌ none explicit | 🔵 V |
| ViFailback | ✅ failure/correction | ❓ partial | 🟠 W / 🟢 H (mixed) |
| DROID/xarm | ✅ success (reward) | ❌ none | 🟡 R |
| UCI Failures | ✅ 5-class failure | ✅ all steps labeled | 🟢 H |
| Haptal Synthetic | ✅ 6-class injected | ✅ all steps labeled | 🔴 S |

### Key Caveats

1. **Reward ≠ failure label.** A low reward means the episode did not succeed, but does not identify *why* or *where* the failure occurred. Step-level failure labels cannot be reliably derived from episode-level reward without additional assumptions.

2. **Weak label inflation.** Rule-based (heuristic) labels on velocity spikes/stuck joints systematically mislabel near-threshold events. Models trained only on weak labels may learn the rule rather than the failure.

3. **Synthetic-to-real gap.** The existing 93.6% benchmark accuracy is on synthetic data. Synthetic failures (injected velocity spikes) may not match the distribution of real failures in BotFails or UCI.

4. **VQA labels are unstructured.** RoboFAC QA text must be parsed to extract structured failure categories; parsing errors propagate into labels.

5. **Class imbalance.** Most real datasets have ≥10:1 nominal:failure ratio. PR-AUC is more informative than ROC-AUC under extreme imbalance.

6. **Cross-dataset transfer.** Models trained on arm manipulation data may not transfer to mobile AMR data. Robot embodiment, DOF, and sensor modalities all differ.

---

## 5. Common Internal Schema

All datasets are normalized to this schema before any model training. Optional fields are set to `None` if unavailable.

```python
{
    "dataset_name":     str,                   # e.g. "botfails", "uci_failures"
    "episode_id":       str,                   # unique episode identifier
    "timesteps":        int,                   # number of steps T
    "state_seq":        np.ndarray,            # (T, D_s) joint/pose states
    "action_seq":       np.ndarray | None,     # (T, D_a) actions
    "video_frames":     list[np.ndarray] | None,  # list of (H,W,3) frames
    "image_paths":      list[str] | None,      # paths if frames not loaded in memory
    "language_task":    str | None,            # task description / instruction
    "episode_label":    str | None,            # "nominal", "failure", or class name
    "step_labels":      np.ndarray | None,     # (T,) per-step failure class int or str
    "semantic_labels":  dict | None,           # {axis: (T,) array}
    "failure_category": str | None,            # e.g. "grasp_slip", "velocity_spike"
    "source_label_type": str,                  # "human", "reward", "weak", "synthetic", "vqa"
    "metadata":         dict,                  # any extra fields from source
}
```

---

## 6. Training & Evaluation Plan

### Phase 1 — Data Loading & Schema Normalization
- Attempt download of each dataset in priority order
- Document access issues and fallbacks immediately
- Convert each dataset to the common schema
- Generate per-dataset EDA: class balance, state dimensionality, step-count distribution

### Phase 2 — Feature Extraction
- Use existing `preprocessing.py` sliding-window feature extractor (W=10, 68 features)
- Compute episode-level aggregates: mean, std, max → 204-dim episode vector
- For video datasets: extract CLIP ViT-B/32 frame embeddings (mean-pooled per episode)
- For UCI tabular: use raw force/torque readings directly

### Phase 3 — Model Training per Dataset

#### Model 1: IsolationForest (episode-level anomaly baseline)
- Fit on nominal episodes only (if nominal/failure split exists)
- If no split: fit on 80% of all episodes, score all
- Hyperparameters: n_estimators=200, contamination='auto'
- Threshold: 75th percentile of training anomaly scores
- Score: `-score_samples(X)` (higher = more anomalous)

#### Model 2: RandomForest classifier
- Episode-level: 204-dim features, n_estimators=200, max_depth=15
- Step-level (where step labels exist): 68-dim window features
- If only episode labels: propagate episode label to all steps (weak assumption — documented)
- Train/test: 80/20 stratified split
- Class weights: balanced

#### Model 3: HistGradientBoosting classifier
- Same features and splits as RF
- Handles NaN natively (useful for sparse sensor datasets)
- Early stopping: n_iter_no_change=10, validation_fraction=0.1
- Compare macro-F1 and PR-AUC against RF

#### Model 4: LSTM Autoencoder (where T ≥ 30 and enough episodes)
- Use existing `lstm_annotator.py` / `models.py` LSTMAEModel
- Input: (T, D_s) state sequence, normalized
- Reconstruction loss → episode anomaly score
- Only run where n_episodes ≥ 200 to avoid severe overfitting

#### Model 5: CLIP visual encoder (where video/images available)
- Use `openai/clip-vit-base-patch32` (frozen, no fine-tuning)
- Frame embedding: 512-dim per frame
- Episode embedding: mean pool over frames
- Classifier: LogisticRegression on 512-dim embedding
- Compare with proprioceptive-only RF

### Phase 4 — Evaluation
- For each dataset × model: compute all applicable metrics
- Where labels are weak/synthetic: note this prominently in every metric table
- Cross-dataset: train on BotFails, test on UCI and vice-versa (where classes overlap)
- Save all metrics to `benchmark_output/public_dataset_eval/{dataset}/{model}_metrics.json`

### Phase 5 — Reporting
- Aggregate into `FINAL_PUBLIC_DATASET_EVAL.md`
- Per-dataset result tables with label-type caveats
- Cross-dataset transfer table
- Recommendations for production architecture

---

## 7. Metrics to Report

All metrics are conditioned on label availability. Metrics on weak/synthetic labels are flagged.

| Metric | When | Notes |
|---|---|---|
| ROC-AUC | Binary or OvR multiclass | Use only when label quality ≥ 🟡 R |
| PR-AUC | Binary, class imbalance > 5:1 | More informative than ROC under imbalance |
| Accuracy | Multiclass with balanced classes | Misleading under imbalance — always pair with macro-F1 |
| Macro-F1 | Multiclass | Weights all classes equally — preferred primary metric |
| Per-class precision/recall/F1 | Multiclass | Report for every class |
| Confusion matrix | Multiclass | JSON + text table |
| Cohen's κ | Multiclass | Agreement above chance |
| False positive rate | Binary | Critical for review-queue sizing |
| Detection rate (recall on failures) | Binary | Primary safety metric |
| PR-curve at operating point | Binary | Show 3 operating points |
| Brier score | Probabilistic | Calibration quality |
| Review rate % | All | % of episodes routed to human review |
| Dataset coverage | All | N episodes, N steps, N labeled, label type |
| Cross-dataset Δ macro-F1 | Transfer | Drop from in-distribution to cross-dataset |

---

## 8. Score Definitions & Thresholding

### Episode Anomaly Score
```
anomaly_score(ep) = -IsolationForest.score_samples(episode_features)
```
- Range: approximately [−0.5, 0.5]; rescaled to [0, 1] after dataset-level min-max
- Interpretation: 0 = very normal, 1 = highly anomalous
- Threshold: `τ_ep = quantile(training_scores, q=0.75)` by default
  - q=0.75 → 25% of training episodes flagged (conservative for high-recall use case)
  - q=0.90 → 10% flagged (conservative for low-volume human review)
  - Customer-adjustable via `review_rate_target` parameter

### Step Anomaly Score
```
step_anomaly_score(t) = 1 - P(class=nominal | window_features_t)
```
- Range: [0, 1], from RandomForest `predict_proba`
- Aggregated to episode: `max(step_anomaly_scores)` and `mean(step_anomaly_scores)`
- Threshold: `τ_step = 0.5` (majority-vote) by default

### Confidence Score
```
confidence(ep) = max(P(class=k)) for k in failure_classes
```
- Range: [0, 1]
- Low confidence (< 0.6) → episode routed to human review queue
- High confidence (> 0.85) → auto-label without review (opt-in, customer configurable)

### Review Score / needs_review Flag
```
needs_review = (anomaly_score > τ_ep) OR (confidence < confidence_threshold)
```
- Combines anomaly signal and model uncertainty
- Drives the review queue in robot_viewer SPA
- Review rate = fraction of episodes with `needs_review = True`

### Quality Score
```
quality_score(ep) = 1 - anomaly_score(ep)
```
- Used in data curation: select high-quality nominal episodes for training downstream policies
- Range: [0, 1]; 1 = high-quality nominal episode

### Semantic Label Confidence
```
semantic_confidence(t, axis) = max(P(label | step_features_t)) for axis in {phase, zone, contact, motion}
```
- Per-step, per-axis confidence
- Episode-level: mean over T steps

### Visual Anomaly Score (if video available)
```
visual_anomaly(ep) = IsolationForest.score_samples(mean_clip_embedding(ep))
```
- Or: cosine distance from centroid of nominal episode CLIP embeddings
- Combined with proprioceptive anomaly score:
  `fusion_score = α * visual_anomaly + (1-α) * anomaly_score`, α = 0.5 default

### Fusion Score
```
fusion_score = weighted_average([visual_anomaly, prop_anomaly, step_anomaly_max])
```
- Weights learned from labeled validation set if available
- Default equal weighting

### Threshold Selection Strategy
1. **If labeled validation set exists:** sweep threshold, maximize F1 on validation
2. **If only review_rate_target is known:** set τ = quantile(1 - review_rate_target)
3. **If no labels:** use 75th percentile of nominal training scores (conservative)
4. All thresholds are stored in the model card and version-controlled per customer

---

## 9. Alternative Training Mechanisms

The following mechanisms are discussed; those marked ✅ are implemented or prototyped in this benchmark run.

### Supervised Classification on Labeled Failures ✅
Train RF/GBM directly on human-labeled failure classes. Best when labels are plentiful and high-quality. Currently deployed for UCI and BotFails (where labels exist).

### Semi-supervised Learning
Train on labeled episodes + unlabeled episodes jointly. Use self-training: train supervised model, pseudo-label high-confidence unlabeled episodes, retrain. Applicable when labeled set is small (< 100 episodes). Not implemented in this run; planned for Proposal C.

### Positive-Unlabeled (PU) Learning
When only nominal examples are labeled and all failures are "unlabeled." Standard PU learning: treat nominal as positives, sample negatives from unlabeled pool. More principled than IsolationForest for one-class classification. Planned as future work.

### Contrastive Learning
Train an encoder where failure episodes are far from nominal episodes in embedding space. Use pairs (nominal, failure) or triplets (anchor, positive, negative). Useful pre-training before downstream classification. Foundation for Proposal F.

### Change-Point Detection
Detect the step where the trajectory distribution shifts. CUSUM or Bayesian online change-point detection on the step feature sequence. Directly outputs the failure onset step — more informative than episode-level labels. Partially implemented in `lstm_annotator.py`.

### Temporal Segmentation
Segment the trajectory into phases (approach, grasp, lift, place). Failures are anomalous phases. HMM or CRF over step features. Related to SemanticAnnotator's task_phase axis.

### Dynamic Time Warping (DTW) against SOP
Compute DTW distance between a production episode and the nearest SOP episode. High DTW distance → anomalous. Parameter-free, interpretable. Slow for long sequences (use FastDTW). Proposed in Proposal A.

### Conformal Prediction
Produce guaranteed coverage prediction sets: "the true class is in this set with ≥ 95% probability." Threshold-free uncertainty quantification. Applicable on top of any trained model. Planned for confidence score calibration.

### Programmatic Weak Supervision (Snorkel-style) ✅ (partial)
Define labeling functions (LFs) for known fault signatures:
- `LF_vel_spike`: max(|dv/dt|) > 3σ
- `LF_stuck`: max(|Δpos|) < 0.001 for N consecutive steps
- `LF_contact`: gripper force > threshold unexpectedly
- `LF_reward_drop`: reward decreases monotonically
Combine LF outputs with a label model (majority vote or generative model). Currently used as the weak-label source for `annotation_model.py`.

### Active Learning ✅
Select the most informative unlabeled episodes for human review. Strategies: entropy sampling, margin sampling, core-set. Implemented in `active_learning.py`. Drives review queue ordering in Proposal C.

### Retrieval-based Nearest-Neighbor Failure Matching
Given a new episode, find the K nearest episodes in feature space from a labeled failure library. Assign failure class by majority vote of neighbors. Zero-shot for new failure types; degrades gracefully as library grows.

### Multi-modal VLM-based Failure Explanation (Video)
Feed video frames + failure flag to a frozen VLM (e.g. LLaVA, GPT-4V). Prompt: "The robot failed at step N. Describe what went wrong." Output: structured failure explanation. Useful for generating human-readable failure cards in the HITL UI. Already partially reflected in `robot_viewer` failure cards.

### Hidden Markov Model (HMM)
Model the robot trajectory as a sequence of hidden states (task phases). Anomalous episodes have unusual state sequences or unexpected transitions. Viterbi decoding identifies the anomalous phase. Complementary to LSTM autoencoder.

---

## 10. Visual / Video Model Architecture

Applied to datasets with video frames (RoboFAC, ViFailback).

### Stage 1 — Frame Encoding (Frozen CLIP)
```
frames (T, H, W, 3)
    │
    ▼
CLIP ViT-B/32 encode_image()  ─── 512-dim per frame
    │
    ▼
temporal aggregation:
  - mean pool → 512-dim episode embedding
  - std pool  → 512-dim episode embedding
  - max pool  → 512-dim episode embedding
  → concat → 1536-dim multi-pool embedding
```

### Stage 2 — Episode Classifier
```
1536-dim CLIP embedding
    │
    ▼
LogisticRegression (L2, C=1.0)   ─── primary visual classifier
    │
    ▼
failure probability per class
```

Alternative: `sklearn.MLPClassifier(hidden=(256,128))` if LR underfits.

### Stage 3 — Multi-modal Fusion (where proprioceptive + video available)
```
CLIP episode embedding (1536-dim)  ─── visual branch
Proprioceptive episode features (204-dim)  ─── state branch
Language task embedding (512-dim, CLIP text) ─── task branch (optional)
    │              │                │
    ▼              ▼                ▼
    ├──────────────┴────────────────┘
    │     concat → 2252-dim
    ▼
StandardScaler → LogisticRegression or MLP
    ▼
episode failure class + confidence
```

### Stage 4 — Frame-level Anomaly (optional)
```
per-frame CLIP embedding (512-dim)
    │
    ▼
IsolationForest (fit on nominal frames)
    │
    ▼
per-frame anomaly score → temporal smoothing (Gaussian σ=3)
    │
    ▼
failure onset step = argmax(smoothed_anomaly_score)
```

### Key Design Decisions
- CLIP is **frozen** — no fine-tuning in this benchmark run (insufficient labeled data)
- If ≥ 500 labeled episodes with video: consider fine-tuning final CLIP layer with LoRA
- Video model only reported where video data is confirmed accessible (not assumed)
- All visual model results are flagged with the label type of the training data

---

## 11. Productized Architecture Proposals

### Proposal A — SOP-based Anomaly Detection (Soonest to market)

**Target customer:** Manufacturer with an established nominal SOP and unlabeled production data.

```
Customer nominal SOP data ──► fit IsolationForest + DTW library
Customer production data  ──► score against normal model
                                     │
                          ┌──────────┴──────────┐
                     low score              high score
                    (normal)               (anomalous)
                          │
                          ▼
                   Review queue (robot_viewer)
                          │
                    Human confirms/corrects
                          │
                          ▼
                   Annotated export
```

**Pros:** No labeled failures needed. Fast to deploy. Interpretable.  
**Cons:** Cannot classify *type* of failure. Performance degrades if SOP itself is not clean.  
**Time to deploy:** 1–2 weeks per customer.

---

### Proposal B — Foundation/Benchmark Pretrained Annotation Model

**Target customer:** Robotics lab that wants zero-shot failure labeling on new data.

```
Pretrain on BotFails + UCI + synthetic ──► foundation annotator
Customer unlabeled data ──► zero-shot episode + step labels
                                     │
                          Human review corrects labels
                                     │
                          Fine-tune customer adapter layer
                                     │
                          Customer-specific model
```

**Pros:** Works without customer SOP. Improves with review.  
**Cons:** Zero-shot performance limited by domain gap. Requires public dataset coverage.  
**Time to deploy:** 2–4 weeks (foundation training) + 1 week per customer.

---

### Proposal C — Human-in-the-Loop Active Learning

**Target customer:** Customer with some labeled data willing to invest in ongoing labeling.

```
Weak model (Proposal A or B)
        │
        ▼
Active learning: select top-K uncertain episodes ──► review queue
        │
  Human reviews K episodes/day
        │
        ▼
Corrections ──► retrain ──► improved model
        │
        ▼
Track: model version, customer taxonomy, label counts, model performance
```

**Active learning query strategy:** entropy sampling (default) or core-set for diversity.  
**Expected improvement:** 5–15% macro-F1 gain per 100 human-labeled episodes (empirical estimate).  
**Implementation:** `active_learning.py` + `feedback_loop.py` already scaffolded.

---

### Proposal D — Multi-modal Visual + Proprioceptive System

**Target customer:** Customer with video-equipped robots where visual failures dominate.

```
Robot video  ──► CLIP frame embeddings ──► visual branch
Robot states ──► sliding window features ──► prop branch
Task language ──► CLIP text embedding ──► task branch
        │
        ▼
Fusion layer (concat + MLP or logistic)
        │
        ▼
Episode failure class + visual anomaly heatmap
```

**Use cases:** Wrong object grasped, occlusion, poor grasp pose, scene disturbance — all invisible in joint states alone.  
**Requires:** Synchronized video + proprioception.

---

### Proposal E — Programmatic Labeling (Snorkel-style)

**Target customer:** High-volume customer (millions of steps/day) who cannot afford human labeling at scale.

```
Domain expert writes labeling functions (LFs)
        │
        ▼
LF_vel_spike, LF_stuck, LF_contact, LF_reward, LF_gripper_mismatch, …
        │
        ▼
Label model (generative or majority vote) ──► probabilistic episode labels
        │
        ▼
Train discriminative model on probabilistic labels
        │
        ▼
Human labels calibrate LF accuracy
```

**Pros:** Scales to millions of episodes. Expert knowledge encoded as code.  
**Cons:** LF coverage gaps; noisy labels for edge cases.

---

### Proposal F — Self-supervised Contrastive Representation Learning

**Target customer:** Customer with large unlabeled dataset and small labeled set.

```
Large unlabeled dataset
        │
        ▼
SimCLR/MoCo on state sequences (augmentation: noise, crop, warp)
        │
        ▼
Pretrained trajectory encoder
        │
small labeled set (100–500 episodes)
        │
        ▼
Fine-tune linear head: failure class
        │
        ▼
Compare to: pure supervised RF on same 100–500 episodes
```

**Expected outcome:** Contrastive pretrain + linear probe outperforms supervised RF when N_labeled < 200.

---

## 12. Customer Onboarding & Data Flow

```
Step 1: Upload
  Customer uploads robot logs (HDF5/ROS bag/CSV) + optionally videos.
  Haptal ingestion API validates schema (states, actions, timestamps).

Step 2: Schema Mapping
  Auto-detect state dimensions, action dimensions, sensor channels.
  Map to internal schema. Flag missing required fields.

Step 3: Embodiment Detection
  Classify robot type from state dimensionality + DOF:
    6-DOF arm, 7-DOF arm, mobile base, quadruped, bimanual, etc.
  Select appropriate pretrained model (arm vs mobile vs manipulation).

Step 4: Nominal Model (if SOP exists)
  Customer provides SOP/reference HDF5.
  Fit IsolationForest on SOP features.
  Store customer-specific threshold τ_ep.

Step 5: Annotate
  Run pretrained annotator (Proposal B foundation model).
  Compute: episode_anomaly_score, step_failure_types, semantic_labels,
           confidence, quality_score, needs_review.

Step 6: Review Queue
  Build prioritized queue: sort by (needs_review=True, confidence ASC).
  Surface in robot_viewer HITL SPA.
  Human reviews accept/reject/flag + correction notes.

Step 7: Corrections → Training Data
  Confirmed labels → corrections.json.
  Human labels weighted 10× vs weak labels in retraining.
  Trigger retraining when N_corrections ≥ 50 or weekly.

Step 8: Fine-tune Customer Adapter
  Retrain final classification layer on customer corrections.
  Version-control: model_v1, model_v2, … per customer.
  Track: macro-F1, review_rate, correction_count per version.

Step 9: Export
  Annotated data exported in customer's preferred format:
    LeRobot HDF5, ACT HDF5, RLDS JSON, custom HDF5, CSV.
  Annotations: episode_label, step_labels, anomaly_score,
               failure_category, confidence, quality_score.

Step 10: Production Monitoring
  Track: anomaly_score distribution over time (drift alert if mean shifts > 2σ).
  Track: review rate trending up → model may need retraining.
  Track: correction rate → measure human-model agreement.
  Monthly model health report to customer.
```

---

## 13. Deliverables Checklist

| Deliverable | Status |
|---|---|
| `PRODUCT_TRAINING_PLAN.md` (this file) | ✅ Done |
| Dataset loader / normalizer (`public_eval/dataset_loaders.py`) | 🔲 Pending |
| Benchmark runner (`public_eval/benchmark_runner.py`) | 🔲 Pending |
| Metrics JSON per dataset (`benchmark_output/public_dataset_eval/`) | 🔲 Pending |
| Cross-dataset transfer report | 🔲 Pending |
| Video model experiment (if video accessible) | 🔲 Conditional |
| `FINAL_PUBLIC_DATASET_EVAL.md` | 🔲 Pending |

---

*End of PRODUCT_TRAINING_PLAN.md*
