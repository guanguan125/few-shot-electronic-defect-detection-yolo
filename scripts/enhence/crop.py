from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "first_data" / "train" / "负样本"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "first_data" / "enhence" / "crop"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_DIRS = {
    ("collision",): "collision",
    ("dirt",): "dirt",
    ("plain particle", "凹点"): "plain_particle_dent",
    ("plain particle", "颗粒"): "plain_particle_particle",
    ("scratch", "划伤"): "scratch_damage",
    ("scratch", "划痕"): "scratch_mark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop LabelMe polygon targets into transparent PNG cutouts."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output crop folder before generating new cutouts.",
    )
    return parser.parse_args()


def class_name_for_image(image_path: Path, source_root: Path) -> str:
    parts = image_path.parent.relative_to(source_root).parts
    candidates = []
    if len(parts) >= 2:
        candidates.append((parts[0], parts[1]))
    if parts:
        candidates.append((parts[0],))

    for key in candidates:
        if key in CLASS_DIRS:
            return CLASS_DIRS[key]
    raise ValueError(f"Unknown class directory for image: {image_path}")


def shape_points(shape: dict) -> list[tuple[float, float]]:
    points = shape.get("points") or []
    if shape.get("shape_type") == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [(float(x), float(y)) for x, y in points]


def crop_shape(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image | None:
    if len(points) < 3:
        return None

    width, height = image.size
    clipped = [
        (min(max(float(x), 0.0), width - 1), min(max(float(y), 0.0), height - 1))
        for x, y in points
    ]

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(clipped, fill=255)
    bbox = mask.getbbox()
    if bbox is None:
        return None

    rgba = image.convert("RGBA")
    crop = rgba.crop(bbox)
    alpha = mask.crop(bbox)
    crop.putalpha(alpha)
    return crop


def iter_images(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    for class_name in CLASS_DIRS.values():
        (output_root / class_name).mkdir(parents=True, exist_ok=True)

    cropped = Counter()
    skipped = Counter()

    for image_path in iter_images(source_root):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            skipped["missing_json"] += 1
            continue

        class_name = class_name_for_image(image_path, source_root)
        data = json.loads(json_path.read_text(encoding="utf-8"))

        with Image.open(image_path) as image:
            for index, shape in enumerate(data.get("shapes") or [], start=1):
                cutout = crop_shape(image, shape_points(shape))
                if cutout is None:
                    skipped["bad_shape"] += 1
                    continue

                output_name = f"{image_path.stem}__shape_{index:03d}.png"
                cutout.save(output_root / class_name / output_name)
                cropped[class_name] += 1

    print(f"output: {output_root}")
    print("cropped:")
    for class_name in CLASS_DIRS.values():
        print(f"  {class_name}: {cropped[class_name]}")
    if skipped:
        print("skipped:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
