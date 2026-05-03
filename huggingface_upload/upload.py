"""
huggingface_upload/upload.py — Upload prepared benchmark data to HuggingFace.

Requires HUGGINGFACE_TOKEN environment variable to be set.

Usage
-----
  export HUGGINGFACE_TOKEN=hf_...
  python huggingface_upload/upload.py
"""

import os
from pathlib import Path


def upload():
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "No HuggingFace token found.\n"
            "Set it with:  export HUGGINGFACE_TOKEN=hf_...\n"
            "Get one at:   https://huggingface.co/settings/tokens"
        )

    upload_dir = Path(__file__).parent
    data_dir   = upload_dir / "data"

    # Verify data has been prepared
    for f in ["train.parquet", "test.parquet"]:
        if not (data_dir / f).exists():
            raise FileNotFoundError(
                f"Missing {f} — run prepare_upload.py first:\n"
                f"  python huggingface_upload/prepare_upload.py"
            )

    repo_id = "haptal-ai/robotics-failure-benchmark"

    api = HfApi(token=token)

    print(f"Creating dataset repo: {repo_id} ...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )

    print(f"Uploading from {upload_dir} ...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add Haptal Robotics Failure Benchmark v1.1",
        ignore_patterns=["*.py", "__pycache__", "*.pyc"],
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nLive at: {url}")
    return url


if __name__ == "__main__":
    upload()
