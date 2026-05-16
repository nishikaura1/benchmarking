---
title: Haptal Robotics 3D Trajectory Visualizer
emoji: 🤖
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---

# Haptal Robotics 3D Trajectory Visualizer

[Haptal](https://haptal.ai) builds failure intelligence for robot training data. We automatically detect, label, and attribute failure modes in raw robot trajectories — so your training pipeline gets clean, structured, physics-grounded failure labels instead of unlabeled noise. Our benchmark dataset is available at [HaptalAI/robotics-failure-benchmark](https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark).

---

## What This Tool Does

This Space has two tabs:

**Tab 1 — Quick Demo**
Select one of four supported LeRobot datasets, enter an episode number, and click Visualize. The app streams that episode's joint state data and renders an interactive 3D Plotly trajectory. Color is a continuous gradient from blue (start of episode) to red (end of episode). Below the plot you'll see the episode number, dataset name, total timesteps, and which columns were used.

**Tab 2 — Analyze Your Data**
Upload your own robot trajectory file (CSV or Parquet). Enter your email — required before results are shown. Click Analyze. The app parses your file, finds trajectory columns, renders the same 3D plot, and returns a quality summary including total episodes, total timesteps, velocity spike detection, and a quality score out of 100. A CTA at the bottom links to our team for full failure attribution.

---

## Supported Datasets for Quick Demo

| Dataset | HuggingFace Link |
|---------|-----------------|
| lerobot/xarm_lift_medium_replay | [Link](https://huggingface.co/datasets/lerobot/xarm_lift_medium_replay) |
| lerobot/xarm_push_medium_replay | [Link](https://huggingface.co/datasets/lerobot/xarm_push_medium_replay) |
| lerobot/aloha_sim_transfer_cube_human | [Link](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) |
| lerobot/aloha_sim_insertion_human | [Link](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) |

---

## About Episode Numbers

Episode numbers correspond directly to `episode_index` in the LeRobot dataset, starting from 0. Exact episode counts per dataset:

| Dataset | Valid Episode Range |
|---------|-------------------|
| lerobot/xarm_lift_medium_replay | 0 – 199 (200 episodes) |
| lerobot/xarm_push_medium_replay | 0 – 199 (200 episodes) |
| lerobot/aloha_sim_transfer_cube_human | 0 – 49 (50 episodes) |
| lerobot/aloha_sim_insertion_human | 0 – 49 (50 episodes) |

If you enter an episode number outside the valid range the app will show a clear error with the correct range — it will not crash.

---

## Uploading Your Own Data

**Accepted formats:** CSV (`.csv`) or Parquet (`.parquet`, `.pq`)

**Expected column names:** The app looks for columns containing any of these keywords: `pos`, `joint`, `state`, `obs`, `action`. Column names are case-insensitive. Examples that work: `joint_pos_0`, `observation.state`, `obs_position_x`, `action_0`.

**What the quality summary shows:**
- Total episodes detected (via `episode_index` column if present)
- Total timesteps in the file
- Which columns were used for visualization
- Velocity spike detection — timesteps where any column exceeds 2 standard deviations from its mean
- Quality score out of 100 — the percentage of timesteps with no detected velocity anomalies. Higher is better.

**Email is required** before results are shown. We use it to send you the full failure attribution report.

---

## What Is Not Included

This tool provides trajectory visualization and basic quality scoring only. It does not perform failure detection, failure attribution, or training data curation.

Full failure detection — including per-timestep failure class labels (grasp slip, velocity spike, stuck joint, trajectory deviation), physics-grounded root cause analysis, and training-ready annotations — is available by contacting [aarav@haptal.ai](mailto:aarav@haptal.ai).

---

## FAQ

**Q: Which datasets are supported in Quick Demo?**
lerobot/xarm_lift_medium_replay, lerobot/xarm_push_medium_replay, lerobot/aloha_sim_transfer_cube_human, lerobot/aloha_sim_insertion_human.

**Q: What column names should my uploaded file have?**
Columns containing `pos`, `joint`, `state`, `obs`, or `action` anywhere in the name. For example: `joint_pos_0`, `observation.state`, `action_0`.

**Q: What does the quality score mean?**
The percentage of timesteps with no detected velocity anomalies (no reading exceeds 2 standard deviations from the column mean). Higher is better. A score above 85 is generally clean data.

**Q: What do the colors in the 3D plot mean?**
Blue = start of episode, red = end of episode. The color gradient runs continuously through the trajectory so you can see the direction and progression of motion.

**Q: How do I get full failure detection on my data?**
Contact [aarav@haptal.ai](mailto:aarav@haptal.ai). We run our physics-grounded failure detection pipeline on your dataset and return labeled annotations ready to go back into your training pipeline.

**Q: Why do you ask for my email?**
To send you the full failure attribution report with per-timestep labels and root cause analysis for your uploaded dataset.

---

## Contact

- Email: [aarav@haptal.ai](mailto:aarav@haptal.ai)
- Website: [haptal.ai](https://haptal.ai)
- Benchmark: [HaptalAI/robotics-failure-benchmark](https://huggingface.co/datasets/HaptalAI/robotics-failure-benchmark)
