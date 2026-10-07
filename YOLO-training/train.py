#!/usr/bin/env python3
"""Fine-tune YOLO11s on the prepared candy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "storage" / "dataset" / "data.yaml",
    )
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--runs", type=Path, default=project_dir / "storage" / "runs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"dataset configuration not found: {args.data}; run prepare_dataset.py"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. On this Jetson, run training with sudo so Python "
            "can read /dev/nvmap."
        )
    print(f"Training on {torch.cuda.get_device_name(0)}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=0,
        project=str(args.runs),
        name="candy-yolo11s",
        exist_ok=False,
        pretrained=True,
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
