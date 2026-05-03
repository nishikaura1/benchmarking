"""
huggingface_upload/prepare_upload.py — Prepare benchmark data for HuggingFace upload.

Reads benchmark/data/train.parquet and benchmark/data/test.parquet,
strips internal feature engineering columns, and saves clean public
versions to huggingface_upload/data/.

Public columns only — no feature vectors, no model internals.

Usage
-----
  python huggingface_upload/prepare_upload.py
"""

import pandas as pd
from pathlib import Path


def prepare_for_upload():
    # Load existing benchmark data
    train = pd.read_parquet("benchmark/data/train.parquet")
    test  = pd.read_parquet("benchmark/data/test.parquet")

    print(f"Loaded train: {len(train):,} episodes  |  test: {len(test):,} episodes")
    print(f"Raw columns: {list(train.columns)}")

    # Keep only public columns — no internal feature engineering
    public_cols = [
        "episode_id",
        "failure_class",
        "action",
        "observation_state",
        "failure_timestep",
        "synthetic",
        "base_dataset",
    ]

    # Only keep columns that actually exist in the data
    train_public = train[[c for c in public_cols if c in train.columns]]
    test_public  = test[[c for c in public_cols if c in test.columns]]

    # Save clean versions
    Path("huggingface_upload/data").mkdir(parents=True, exist_ok=True)
    train_public.to_parquet("huggingface_upload/data/train.parquet", index=False)
    test_public.to_parquet("huggingface_upload/data/test.parquet",  index=False)

    print(f"\nPublic columns kept: {list(train_public.columns)}")
    print(f"Train episodes : {len(train_public):,}")
    print(f"Test episodes  : {len(test_public):,}")
    print(f"\nSaved to huggingface_upload/data/")
    print("Done. Ready to upload.")


if __name__ == "__main__":
    prepare_for_upload()
