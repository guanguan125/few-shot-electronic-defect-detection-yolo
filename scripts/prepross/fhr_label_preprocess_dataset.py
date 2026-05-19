from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from yolo_preprocess_dataset import (
    CLASS_NAMES,
    PROJECT_ROOT,
    bbox_from_points,
    build_preprocess_channels,
    iter_images,
    prepare_output,
    preview_from_channels,
    read_gray,
    safe_output_stem,
    shape_points,
    write_data_yaml,
    write_manifest,
    yolo_line,
)


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "first_data" / "fhr_LABEL"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "first_data" / "data_fhr_label_with_preprocess"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_fhr_label_with_preprocess.yaml"

FHR_LABEL_TO_CLASS_ID = {
    "0": 3,  # collision
    "1": 1,  # dirt
    "2": 0,  # plain_particle
    "3": 0,  # plain_particle
    "4": 2,  # scratch
    "5": 2,  # scratch
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert flat fhr_LABEL LabelMe data to the same 4-channel "
            "preprocessed YOLO format as yolo_preprocess_dataset.py."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def load_labelme(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def class_id_for_fhr_shape(shape: dict, json_path: Path) -> int:
    raw_label = str(shape.get("label", "")).strip()
    if raw_label not in FHR_LABEL_TO_CLASS_ID:
        expected = ", ".join(sorted(FHR_LABEL_TO_CLASS_ID))
        raise ValueError(f"{json_path}: unsupported fhr label {raw_label!r}; expected one of {expected}")
    return FHR_LABEL_TO_CLASS_ID[raw_label]


def primary_class_for_split(image_path: Path) -> int:
    data = load_labelme(image_path.with_suffix(".json"))
    if data is None:
        return -1
    for shape in data.get("shapes") or []:
        raw_label = str(shape.get("label", "")).strip()
        if raw_label in FHR_LABEL_TO_CLASS_ID:
            return FHR_LABEL_TO_CLASS_ID[raw_label]
    return -1


def split_images(images: list[Path], val_ratio: float, seed: int) -> dict[Path, str]:
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")

    grouped: dict[int, list[Path]] = {}
    for image_path in images:
        grouped.setdefault(primary_class_for_split(image_path), []).append(image_path)

    rng = random.Random(seed)
    split_by_image: dict[Path, str] = {}
    for paths in grouped.values():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if val_ratio > 0 and len(shuffled) > 1:
            val_count = max(1, val_count)
        val_set = set(shuffled[:val_count])
        for path in shuffled:
            split_by_image[path] = "val" if path in val_set else "train"
    return split_by_image


def convert_one(
    image_path: Path,
    source_root: Path,
    output_root: Path,
    split: str,
) -> tuple[dict[str, object], Counter]:
    stats: Counter = Counter()
    gray = read_gray(image_path)
    height, width = gray.shape[:2]
    channels = build_preprocess_channels(gray)

    json_path = image_path.with_suffix(".json")
    data = load_labelme(json_path)
    label_lines: list[str] = []
    if data is None:
        stats["missing_json"] += 1
    else:
        for shape in data.get("shapes") or []:
            bbox = bbox_from_points(shape_points(shape), width, height)
            if bbox is None:
                stats["bad_shape"] += 1
                continue
            class_id = class_id_for_fhr_shape(shape, json_path)
            label_lines.append(yolo_line(class_id, bbox, width, height))
            stats[f"class_{class_id}"] += 1

    stem = safe_output_stem(image_path, source_root)
    out_image = output_root / "images" / split / f"{stem}.jpg"
    out_npy = output_root / "images" / split / f"{stem}.npy"
    out_label = output_root / "labels" / split / f"{stem}.txt"

    cv2.imwrite(str(out_image), preview_from_channels(channels), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    np.save(out_npy, channels, allow_pickle=False)
    out_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    stats["images"] += 1
    stats["objects"] += len(label_lines)
    return (
        {
            "source": str(image_path),
            "split": split,
            "image": str(out_image.relative_to(output_root)),
            "npy": str(out_npy.relative_to(output_root)),
            "label": str(out_label.relative_to(output_root)),
            "objects": len(label_lines),
        },
        stats,
    )


def build_dataset(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    val_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = False,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()

    images = iter_images(source_root)
    if not images:
        raise FileNotFoundError(f"no images found in {source_root}")

    prepare_output(output_root, clean=clean)
    write_data_yaml(config_path, output_root)
    split_by_image = split_images(images, val_ratio=val_ratio, seed=seed)

    total = Counter()
    rows: list[dict[str, object]] = []
    for image_path in images:
        row, stats = convert_one(
            image_path=image_path,
            source_root=source_root,
            output_root=output_root,
            split=split_by_image[image_path],
        )
        rows.append(row)
        total.update(stats)

    write_manifest(output_root, rows)
    return {
        "source": str(source_root),
        "output": str(output_root),
        "config": str(config_path),
        "images": total["images"],
        "objects": total["objects"],
        "train_images": sum(1 for row in rows if row["split"] == "train"),
        "val_images": sum(1 for row in rows if row["split"] == "val"),
        "classes": {CLASS_NAMES[i]: total[f"class_{i}"] for i in CLASS_NAMES},
        "skipped": {
            key: total[key]
            for key in ("missing_json", "bad_shape")
            if total[key]
        },
    }


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        source_root=args.source,
        output_root=args.output,
        config_path=args.config,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean=args.clean,
    )
    print(f"source: {summary['source']}")
    print(f"output: {summary['output']}")
    print(f"config: {summary['config']}")
    print(f"images: {summary['images']} train={summary['train_images']} val={summary['val_images']}")
    print(f"objects: {summary['objects']}")
    for class_name, count in summary["classes"].items():
        print(f"  {class_name}: {count}")
    if summary["skipped"]:
        print("skipped:")
        for reason, count in summary["skipped"].items():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
