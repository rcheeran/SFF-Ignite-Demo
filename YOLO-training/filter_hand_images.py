#!/usr/bin/env python3
"""Create a YOLO dataset excluding every image where MediaPipe detects a hand."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import mediapipe as mp


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = [
    "MMs_peanut",
    "MMs_regular",
    "airheads",
    "gummy_worms",
    "milky_way",
    "nerds",
    "skittles",
    "snickers",
    "starburst",
    "three_musketeers",
    "twizzlers",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_dir / "storage" / "datasets" / "collected-starburst",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "storage" / "datasets" / "startbirst-training",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="minimum MediaPipe hand detection and presence confidence",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")
    if not args.model.is_file():
        raise FileNotFoundError(f"hand model not found: {args.model}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    image_dir = args.source / "images"
    label_dir = args.source / "labels"
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")
    pairs = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing label for {image.name}: {label}")
        pairs.append((image, label))

    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "images").mkdir(parents=True)
    (temporary / "labels").mkdir()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
    )
    removed = []
    kept: list[tuple[Path, Path]] = []
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            for index, (image, label) in enumerate(pairs, start=1):
                result = detector.detect(mp.Image.create_from_file(str(image)))
                if result.hand_landmarks:
                    removed.append(
                        {
                            "image": image.name,
                            "hand_count": len(result.hand_landmarks),
                            "scores": [
                                float(categories[0].score)
                                for categories in result.handedness
                                if categories
                            ],
                        }
                    )
                else:
                    link_or_copy(image, temporary / "images" / image.name)
                    link_or_copy(label, temporary / "labels" / label.name)
                    kept.append((image, label))
                if index % 100 == 0 or index == len(pairs):
                    print(
                        f"Scanned {index}/{len(pairs)}: "
                        f"kept={len(kept)}, removed={len(removed)}"
                    )

        shuffled = kept.copy()
        random.Random(args.seed).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * args.validation_fraction))
        validation = shuffled[:validation_count]
        training = shuffled[validation_count:]
        for split, samples in (("train", training), ("val", validation)):
            for image, label in samples:
                link_or_copy(image, temporary / split / "images" / image.name)
                link_or_copy(label, temporary / split / "labels" / label.name)

        names = "\n".join(
            f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)
        )
        (temporary / "data.yaml").write_text(
            f"path: {args.output.resolve()}\n"
            "train: train/images\n"
            "val: val/images\n"
            "names:\n"
            f"{names}\n",
            encoding="utf-8",
        )

        report = {
            "source": str(args.source.resolve()),
            "hand_model": str(args.model.resolve()),
            "confidence": args.confidence,
            "validation_fraction": args.validation_fraction,
            "split_seed": args.seed,
            "total_images": len(pairs),
            "kept_images": len(kept),
            "removed_images": len(removed),
            "training_images": len(training),
            "validation_images": len(validation),
            "removed": removed,
        }
        (temporary / "filter-report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"Created {args.output} with {len(training)} training and "
        f"{len(validation)} validation image-label pairs"
    )


if __name__ == "__main__":
    main()