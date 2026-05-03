"""Run IsolationForest vs LSTM-AE comparison using cached HDF5 features."""
import sys, h5py
sys.stdout.reconfigure(line_buffering=True)

from models import compare_models

with h5py.File('benchmark_output/lerobot_xarm_lift_medium_replay_scores.h5') as f:
    features = f['features'][:]
    labels   = f['true_labels'][:]

print(f'Loaded {len(features)} episodes, {int(labels.sum())} failures', flush=True)
compare_models(features, labels, 'xarm_lift_medium_replay')
print("DONE", flush=True)
