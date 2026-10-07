#!/usr/bin/env python3
"""Validate and split the candy dataset for Ultralytics YOLO training."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_dir.parent
        / "OpenCV-service"
        / "datasets"
        / "candy_data_14DEC24",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "storage" / "dataset"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_labels(path: Path) -> set[int]:
    class_ids: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: coordinates must be in [0, 1]")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(f"{path}:{line_number}: box dimensions must be positive")
        class_ids.add(class_id)
    if not class_ids:
        raise ValueError(f"{path}: label file is empty")
    return class_ids


def find_covered_split(
    samples: list[tuple[Path, Path, set[int]]],
    validation_count: int,
    seed: int,
) -> tuple[list[tuple[Path, Path, set[int]]], list[tuple[Path, Path, set[int]]]]:
    required = set(range(len(CLASS_NAMES)))
    for candidate_seed in range(seed, seed + 10_000):
        shuffled = samples.copy()
        random.Random(candidate_seed).shuffle(shuffled)
        validation = shuffled[:validation_count]
        training = shuffled[validation_count:]
        validation_classes = set().union(*(sample[2] for sample in validation))
        training_classes = set().union(*(sample[2] for sample in training))
        if validation_classes == required and training_classes == required:
            return training, validation
    raise RuntimeError("could not create a split containing every class in train and val")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(
    output: Path,
    name: str,
    samples: list[tuple[Path, Path, set[int]]],
) -> None:
    for image, label, _ in samples:
        link_or_copy(image, output / name / "images" / image.name)
        link_or_copy(label, output / name / "labels" / label.name)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    image_dir = args.source / "images"
    label_dir = args.source / "labels"
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")

    samples: list[tuple[Path, Path, set[int]]] = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing label for {image.name}: {label}")
        samples.append((image, label, read_labels(label)))

    extra_labels = {
        path.stem for path in label_dir.glob("*.txt")
    } - {image.stem for image in images}
    if extra_labels:
        raise ValueError(f"labels without images: {sorted(extra_labels)}")

    validation_count = max(1, round(len(samples) * args.validation_fraction))
    training, validation = find_covered_split(samples, validation_count, args.seed)

    if args.output.exists():
        raise FileExistsError(
            f"output already exists: {args.output}; choose a new --output path"
        )
    write_split(args.output, "train", training)
    write_split(args.output, "val", validation)

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data_yaml = (
        f"path: {args.output.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        f"{names}\n"
    )
    (args.output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Prepared {len(training)} training and {len(validation)} validation images")
    print(f"Dataset configuration: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
