# Haptal vs. Human Operators
## Automating Robot Training Data Quality at Scale

---

### The problem every robotics team has

Before a robot episode goes into training, someone has to decide: is this a good demonstration or a bad one?

Right now that someone is a human. They watch the video, they make a judgment, they move on. At small scale this works. At the scale needed to train production manipulation systems — thousands of episodes per week — it breaks down.

The teams building at this scale today have staffed up operator pools. They have managers who sample operator ratings and assign ELO scores. They have review queues and escalation paths. It is expensive, it is slow, and the labels are inconsistent between annotators.

Haptal automates this process. Here is exactly how it compares.

---

### Side-by-side comparison

| | Human operator process | Haptal |
|---|---|---|
| **Episode verdict** | Human watches video, marks pass/fail | Model scores episode in < 1 second |
| **Failure granularity** | Binary: pass or fail | 6 failure classes with failure timestep |
| **Operator consistency** | κ ≈ 0.60–0.75 (typical human IAA) | κ = 0.66 supervised head; 0.92 on synthetic benchmark |
| **False alarm rate** | ~15–25% (varies by operator fatigue) | 15.3% (matched to human baseline) |
| **Miss rate** | ~15–20% | 18.7% (matched to human baseline) |
| **Throughput** | 1 operator reviews ~50–100 episodes/hour | Unlimited — parallel, no queue |
| **Operator reliability tracking** | Manager samples + ELO score | Automated correction-rate ELO (same scale) |
| **Retraining on errors** | Operator coaching session | Auto-retrain when correction queue hits threshold |
| **Cost** | $X / hour per operator, scales linearly | Fixed SaaS, scales to zero marginal cost |
| **Audit trail** | Spreadsheet / manual log | Full correction history, version-controlled model |
| **Works at 3am** | No | Yes |

---

### The numbers

**Human parity study** — 160 real robot episodes, 2 platforms (xArm, DROID), human labels as ground truth:

| Method | Cohen's κ | Accuracy | False Alarm Rate | Miss Rate |
|---|---|---|---|---|
| Haptal supervised head | **0.66** | **83.1%** | **15.3%** | **18.7%** |
| Human–human typical range | 0.60–0.75 | — | ~15–25% | ~15–20% |
| Majority baseline | 0.00 | 53.1% | 0% | 100% |

Haptal sits squarely inside the human-human agreement range — with 160 labelled episodes. Agreement improves as more human-labelled data is added (standard supervised learning curve).

**What Haptal adds on top of binary pass/fail:**

- Failure type classification (6 classes): grasp slip, velocity spike, trajectory deviation, stuck joint, overcorrect, nominal
- Failure timestep: pinpoints exactly when in the episode the failure occurred
- Confidence score per step: low-confidence steps are flagged for human review
- Unknown failure detection: episodes with no confident class are surfaced, not silently misclassified

Human operators cannot produce any of these. They mark pass/fail and move on.

---

### The reliability loop

The MIT team and others give human operators ELO scores based on manager sampling. If an operator's labels diverge from ground truth, their score drops and they get coaching or reassignment.

Haptal runs the same process automatically:

1. **Every human correction** is logged with reviewer ID, original prediction, corrected label, and timestamp
2. **Correction rate** is computed on a rolling 7-day window
3. **ELO score** updates in real time: base 1500, rises when corrections are rare, falls when they spike
4. **Auto-retrain triggers** when the ELO drops below 1300 — equivalent to replacing a poor-performing operator — no human escalation required
5. **Model version bumps** on each retrain, with full history and accuracy tracking

```
Model ELO 1600+ → excellent  (< 5% correction rate)
Model ELO 1500  → baseline   (fresh model, no feedback)
Model ELO 1400  → acceptable (~15% correction rate)
Model ELO 1300  → degraded   (auto-retrain triggered)
Model ELO <1200 → critical   (immediate retrain)
```

Human operator ELO programs typically operate in the 1400–1600 range. Haptal targets the same band.

---

### What we trained on

The production model (RobotAnnotator v1.1) was trained on real robot trajectories from four LeRobot datasets:

- `lerobot/xarm_lift_medium_replay` — xArm manipulation
- `lerobot/xarm_push_medium_replay` — xArm pushing
- `lerobot/aloha_sim_transfer_cube_human` — ALOHA bimanual
- `lerobot/aloha_sim_insertion_human` — ALOHA insertion

Validation accuracy on held-out real data: **89.9%**  
Brier score (calibrated confidence): **0.017**

The synthetic benchmark (Haptal Robotics Failure Benchmark v1.1) adds cross-platform evaluation:

- Train: 4 datasets → Test: held-out platform (ALOHA insertion)
- In-distribution accuracy: **93.6%**
- OOD accuracy: **90.8%**
- Generalisation gap: **0.03** (industry standard is < 0.15)

---

### The two-stage architecture

```
Episode (state sequence)
        │
        ▼
┌───────────────────┐
│  Step-level RF    │  68-dim physics features per step
│  (10 classes)     │  velocity, jerk, acceleration, rolling stats
│  89.9% accuracy   │  → failure type + timestep per step
└────────┬──────────┘
         │  204-dim episode vector (mean + std + max across steps)
         ▼
┌───────────────────┐
│  Episode-level    │  Supervised on human pass/fail labels
│  classifier       │  κ = 0.66 vs human operators
│  (pass/fail)      │  → episode verdict + confidence
└────────┬──────────┘
         │
         ▼
  ┌──────────────────────────────┐
  │ Output                       │
  │  • Episode: PASS / FAIL      │
  │  • Failure type (if fail)    │
  │  • Failure timestep          │
  │  • Confidence score          │
  │  • Low-confidence → review   │
  └──────────────────────────────┘
```

Human operators produce only the top box (PASS / FAIL). Haptal produces all four.

---

### Benchmark

The Haptal Robotics Failure Benchmark v1.1 is the first public benchmark for robot training data annotation quality.

- Fixed test set: 360 episodes, 6 failure classes, 5 robot platforms
- Public leaderboard: `HaptalAI/robotics-failure-benchmark` on HuggingFace
- Score your model: `python score.py your_predictions.csv`
- Submit results to `aarav@haptal.ai` to appear on the leaderboard

No comparable benchmark exists. Open X-Embodiment, BridgeData V2, DROID, and LeRobot are data collections. None measure annotation quality or failure detection accuracy.

---

### What this means for your team

If you are running human operators today:

- Every operator-hour spent on pass/fail review can be redirected to the 17.5% of episodes that Haptal flags as low-confidence (the hard cases that genuinely need a human)
- Failure type labels that currently require a robotics engineer to investigate are generated automatically with timestep precision
- Operator ELO and model ELO run on the same scale — you can compare them directly

If you are scaling up data collection:

- Human review throughput does not scale. Haptal does.
- The model improves every time a human corrects it — corrections are fuel, not overhead

---

*Haptal AI · aarav@haptal.ai · haptal.ai*  
*RobotAnnotator v1.1 · Benchmark v1.1 · huggingface.co/datasets/HaptalAI/robotics-failure-benchmark*
