from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "first_data" / "train" / "负样本"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "first_data" / "data_train_with_preprocess"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_train_with_preprocess.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PREPROCESS_CHANNELS = ("gray", "clahe", "scharr", "blackhat")

CLASS_NAMES = {
    0: "plain_particle",
    1: "dirt",
    2: "scratch",
    3: "collision",
}

LABEL_TO_CLASS_ID = {
    "plain particle": 0,
    "plain_particle": 0,
    "plain-particle": 0,
    "particle": 0,
    "dent": 0,
    "凹点": 0,
    "颗粒": 0,
    "dirt": 1,
    "scratch": 2,
    "scratch damage": 2,
    "scratch_damage": 2,
    "scratch mark": 2,
    "scratch_mark": 2,
    "划伤": 2,
    "划痕": 2,
    "collision": 3,
}

DIR_TO_CLASS_ID = {
    ("plain particle",): 0,
    ("plain particle", "凹点"): 0,
    ("plain particle", "颗粒"): 0,
    ("dirt",): 1,
    ("scratch",): 2,
    ("scratch", "划伤"): 2,
    ("scratch", "划痕"): 2,
    ("collision",): 3,
}

DIR_PART_ALIASES = {
    "plain particle": "plain_particle",
    "凹点": "dent",
    "颗粒": "particle",
    "划伤": "damage",
    "划痕": "mark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LabelMe grayscale defect images to a YOLO dataset and "
            "save 4-channel preprocessed npy tensors beside each image."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if float(img.max()) == float(img.min()):
        return np.zeros_like(img, dtype=np.uint8)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def read_gray(image_path: Path) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    return gray


def build_preprocess_channels(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    scharr_x = cv2.Scharr(clahe, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(clahe, cv2.CV_64F, 0, 1)
    scharr = normalize_to_uint8(cv2.magnitude(scharr_x, scharr_y))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    blackhat = normalize_to_uint8(cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel))

    return np.stack([gray, clahe, scharr, blackhat], axis=-1).astype(np.uint8)


def preview_from_channels(channels: np.ndarray) -> np.ndarray:
    gray = channels[..., 0]
    clahe = channels[..., 1]
    scharr = channels[..., 2]
    return cv2.merge([scharr, clahe, gray])


def iter_images(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def normalize_label(label: str | None) -> str:
    return (label or "").strip().lower()


def class_id_from_path(image_path: Path, source_root: Path) -> int:
    parts = image_path.parent.relative_to(source_root).parts
    candidates: list[tuple[str, ...]] = []
    if len(parts) >= 2:
        candidates.append((parts[0], parts[1]))
    if parts:
        candidates.append((parts[0],))
    for candidate in candidates:
        if candidate in DIR_TO_CLASS_ID:
            return DIR_TO_CLASS_ID[candidate]
    raise ValueError(f"unknown class directory for image: {image_path}")


def class_id_for_shape(shape: dict, image_path: Path, source_root: Path) -> int:
    label = normalize_label(shape.get("label"))
    if label in LABEL_TO_CLASS_ID:
        return LABEL_TO_CLASS_ID[label]
    return class_id_from_path(image_path, source_root)


def shape_points(shape: dict) -> list[tuple[float, float]]:
    points = shape.get("points") or []
    if shape.get("shape_type") == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [(float(x), float(y)) for x, y in points]


def bbox_from_points(
    points: list[tuple[float, float]], width: int, height: int
) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    x1 = float(np.clip(xs.min(), 0, width - 1))
    y1 = float(np.clip(ys.min(), 0, height - 1))
    x2 = float(np.clip(xs.max(), 0, width - 1))
    y2 = float(np.clip(ys.max(), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_line(class_id: int, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def safe_output_stem(image_path: Path, source_root: Path) -> str:
    rel_parent = image_path.parent.relative_to(source_root).parts
    parts = [DIR_PART_ALIASES.get(part, part).replace(" ", "_") for part in rel_parent]
    parts.append(image_path.stem)
    return "__".join(parts)


def split_images(images: list[Path], source_root: Path, val_ratio: float, seed: int) -> dict[Path, str]:
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")

    grouped: dict[int, list[Path]] = {}
    for image_path in images:
        grouped.setdefault(class_id_from_path(image_path, source_root), []).append(image_path)

    rng = random.Random(seed)
    split_by_image: dict[Path, str] = {}
    for class_id, paths in grouped.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if val_ratio > 0 and len(shuffled) > 1:
            val_count = max(1, val_count)
        val_set = set(shuffled[:val_count])
        for path in shuffled:
            split_by_image[path] = "val" if path in val_set else "train"
    return split_by_image


def prepare_output(output_root: Path, clean: bool) -> None:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for cache_file in output_root.rglob("*.cache"):
        cache_file.unlink()


def write_data_yaml(config_path: Path, output_root: Path) -> None:
    lines = [
        f"path: {output_root}",
        "",
        "train: images/train",
        "val: images/val",
        "",
        "channels: 4",
        "",
        "names:",
    ]
    lines.extend(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    lines.extend(
        [
            "",
            "preprocess_channels:",
            *[f"  {idx}: {name}" for idx, name in enumerate(PREPROCESS_CHANNELS)],
        ]
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    label_lines: list[str] = []
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for shape in data.get("shapes") or []:
            bbox = bbox_from_points(shape_points(shape), width, height)
            if bbox is None:
                stats["bad_shape"] += 1
                continue
            class_id = class_id_for_shape(shape, image_path, source_root)
            label_lines.append(yolo_line(class_id, bbox, width, height))
            stats[f"class_{class_id}"] += 1
    else:
        stats["missing_json"] += 1

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


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    split_by_image = split_images(images, source_root, val_ratio=val_ratio, seed=seed)

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

    summary = {
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
    return summary


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
