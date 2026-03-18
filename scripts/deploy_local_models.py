from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path("/home/user/Project_files/project")
LOCAL_MODELS_DIR = PROJECT_ROOT / "local_models"
MICROVQA_MODELS_DIR = Path("/home/user/Project_files/microvqa/models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen-repo",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--qwen-dir",
        type=str,
        default=str(LOCAL_MODELS_DIR / "qwen3-vl-4b"),
    )
    parser.add_argument(
        "--dino-source-dir",
        type=str,
        default=str(MICROVQA_MODELS_DIR / "dinov3_vit"),
    )
    parser.add_argument(
        "--dino-link",
        type=str,
        default=str(LOCAL_MODELS_DIR / "dinov3_vit"),
    )
    parser.add_argument(
        "--skip-qwen-download",
        action="store_true",
    )
    return parser.parse_args()


def ensure_symlink(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"DINO source directory not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source, target)


def download_qwen(repo_id: str, local_dir: Path) -> Path:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return Path(path)


def main() -> None:
    args = parse_args()

    dino_source = Path(args.dino_source_dir)
    dino_link = Path(args.dino_link)
    ensure_symlink(dino_source, dino_link)

    payload = {
        "dinov3_local_link": str(dino_link),
        "dinov3_source": str(dino_source),
    }

    if not args.skip_qwen_download:
        qwen_dir = download_qwen(args.qwen_repo, Path(args.qwen_dir))
        payload["qwen_dir"] = str(qwen_dir)
        payload["qwen_repo"] = args.qwen_repo

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
