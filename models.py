"""
Anomaly detection models with a unified interface.

Both models expose:
    .fit(nominal_features)          — train on nominal episodes only
    .score(features)                — returns anomaly scores (higher = more anomalous)
    .benchmark(features, labels)    — returns a benchmark card dict

Usage:
    from models import IsolationForestModel, LSTMAEModel, compare_models
"""

import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Shared base ──────────────────────────────────────────────────────────────

class BaseAnomalyModel:
    name = "base"

    def fit(self, nominal_features: np.ndarray):
        raise NotImplementedError

    def score(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def benchmark(self, features: np.ndarray, labels: np.ndarray,
                  threshold_quantile: float = 0.75) -> dict:
        scores = self.score(features)
        auc = roc_auc_score(labels, scores)
        thresh = np.quantile(scores, threshold_quantile)
        preds = (scores >= thresh).astype(int)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        return {
            "model": self.name,
            "roc_auc": round(float(auc), 4),
            "detection_rate_pct": round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0,
            "false_positive_rate_pct": round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "threshold_quantile": threshold_quantile,
        }


# ── Model 1: IsolationForest (fast baseline) ─────────────────────────────────

class IsolationForestModel(BaseAnomalyModel):
    name = "IsolationForest"

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.clf = IsolationForest(contamination=contamination,
                                   random_state=random_state, n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(self, nominal_features: np.ndarray):
        scaled = self.scaler.fit_transform(nominal_features)
        self.clf.fit(scaled)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return -self.clf.score_samples(scaled)


# ── Model 2: LSTM Autoencoder (sequence-aware) ───────────────────────────────

class _LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])          # (batch, latent_dim)


class _LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.fc   = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # repeat latent vector across time steps
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        return self.out(out)            # (batch, seq_len, output_dim)


class _LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, num_layers):
        super().__init__()
        self.encoder = _LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = _LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAEModel(BaseAnomalyModel):
    """
    LSTM Autoencoder trained on nominal episode sequences.
    Anomaly score = per-episode reconstruction error (MSE).
    Captures temporal patterns that IsolationForest misses.
    """
    name = "LSTM-Autoencoder"

    def __init__(self, seq_len: int = None, hidden_dim: int = 32,
                 latent_dim: int = 8, num_layers: int = 1,
                 epochs: int = 40, batch_size: int = 64, lr: float = 1e-3):
        self.seq_len    = seq_len   # auto-derived from feature dim if None
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.scaler     = StandardScaler()
        self.model      = None
        # force CPU — avoids MPS segfaults on Apple Silicon
        self.device     = torch.device("cpu")

    def _to_sequences(self, features: np.ndarray) -> torch.Tensor:
        """
        Reshape (N, D) episode vectors into (N, seq_len, step_dim).
        seq_len is chosen so step_dim >= 2 (LSTM needs meaningful per-step input).
        """
        N, D = features.shape
        # auto pick seq_len: largest divisor of D that gives step_dim >= 2
        if self.seq_len is None:
            self.seq_len = next(
                (s for s in range(min(D // 2, 8), 0, -1) if D % s == 0), 1
            )
        remainder = D % self.seq_len
        if remainder:
            features = np.pad(features, ((0, 0), (0, self.seq_len - remainder)))
        step_dim = features.shape[1] // self.seq_len
        seqs = features.reshape(N, self.seq_len, step_dim)
        return torch.tensor(seqs, dtype=torch.float32)

    def fit(self, nominal_features: np.ndarray):
        scaled = self.scaler.fit_transform(nominal_features)
        seqs   = self._to_sequences(scaled).to(self.device)
        _, _, step_dim = seqs.shape

        self.model = _LSTMAE(step_dim, self.hidden_dim, self.latent_dim,
                             self.seq_len, self.num_layers).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loader = DataLoader(TensorDataset(seqs), batch_size=self.batch_size,
                            shuffle=True, num_workers=0)

        self.model.train()
        print(f"  Training LSTM-AE on {len(seqs)} nominal episodes "
              f"({self.epochs} epochs)...", flush=True)
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in loader:
                optimiser.zero_grad()
                recon = self.model(batch)
                loss  = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.5f}", flush=True)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        seqs   = self._to_sequences(scaled).to(self.device)
        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(seqs), 64):
                batch = seqs[i:i+64]
                recon = self.model(batch)
                mse   = ((recon - batch) ** 2).mean(dim=(1, 2))
                scores.extend(mse.cpu().numpy())
        return np.array(scores)


# ── Cross-dataset validation ─────────────────────────────────────────────────

def cross_dataset_validate(
    train_features: np.ndarray, train_labels: np.ndarray,
    test_features:  np.ndarray, test_labels:  np.ndarray,
    model_class, dataset_train: str, dataset_test: str, **model_kwargs
) -> dict:
    """
    Train on nominal episodes from dataset A, score episodes from dataset B.
    Strong generalisation = the model learned failure physics, not dataset quirks.
    """
    nominal = train_features[train_labels == 0]
    m = model_class(**model_kwargs)
    m.fit(nominal)
    card = m.benchmark(test_features, test_labels)
    card["train_dataset"] = dataset_train
    card["test_dataset"]  = dataset_test
    card["cross_dataset"] = True
    print(f"\n  Cross-dataset  ({dataset_train} → {dataset_test})")
    print(f"  {card['model']}  AUC={card['roc_auc']}  "
          f"detect={card['detection_rate_pct']}%  "
          f"FPR={card['false_positive_rate_pct']}%")
    return card


# ── Side-by-side comparison ───────────────────────────────────────────────────

def compare_models(features: np.ndarray, labels: np.ndarray,
                   dataset_label: str, output_dir: Path = Path("benchmark_output")):
    """
    Fit both models on nominal episodes, benchmark both, save comparison JSON.
    """
    nominal = features[labels == 0]
    print(f"\n{'='*60}")
    print(f"Model comparison — {dataset_label}")
    print(f"  Nominal: {nominal.shape[0]}   Failures: {labels.sum()}")

    results = []
    for ModelClass, kwargs in [
        (IsolationForestModel, {}),
        (LSTMAEModel,          {"epochs": 20}),
    ]:
        m = ModelClass(**kwargs)
        m.fit(nominal)
        card = m.benchmark(features, labels)
        card["dataset"] = dataset_label
        results.append(card)
        print(f"\n  [{card['model']}]")
        print(f"    ROC-AUC        : {card['roc_auc']}")
        print(f"    Detection rate : {card['detection_rate_pct']}%")
        print(f"    FPR            : {card['false_positive_rate_pct']}%")

    safe = dataset_label.replace("/", "_").replace(" ", "_")
    out  = output_dir / f"{safe}_model_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")
    return results
